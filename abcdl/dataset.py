"""LeRobot-compatible torch Dataset over a directory of abcdl episodes."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from abcdl.format.reader import read_abcdl


@dataclass
class AbcdlDatasetMeta:
    features: dict
    fps: float
    camera_keys: list
    video_keys: list
    tasks: list
    robot_type: str = "yam_bimanual"


class AbcdlDataset(Dataset):
    """Torch Dataset over a root directory of abcdl episode dirs.

    Each episode dir must contain:
      - states_actions.bin
      - combined_camera-images-rgb.mp4
      - episode_metadata.json

    Decodes each episode once via ``read_abcdl`` and keeps the N most-recently
    accessed episodes in an LRU cache so shuffled access does not re-decode
    every item.
    """

    def __init__(
        self,
        root: str,
        delta_timestamps: Optional[dict] = None,
        camera_keys: Optional[list] = None,
        cache_episodes: int = 4,
    ):
        self.root = root
        self.delta_timestamps = delta_timestamps or {}

        # Discover episode dirs (sorted for determinism).
        self._dirs: list[str] = sorted(
            os.path.join(root, d)
            for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
            and os.path.exists(os.path.join(root, d, "states_actions.bin"))
        )
        if not self._dirs:
            raise ValueError(f"no abcdl episodes under {root}")

        # Read per-episode lengths and camera list from metadata JSON.
        self._lengths: list[int] = []
        self._tasks_per_ep: list[str] = []
        self._cams: Optional[list[str]] = None
        for d in self._dirs:
            with open(os.path.join(d, "episode_metadata.json")) as f:
                m = json.load(f)
            self._lengths.append(int(m["num_steps"]))
            self._tasks_per_ep.append(m["task_name"])
            if self._cams is None:
                self._cams = list(m["cameras"])

        self.camera_keys: list[str] = camera_keys or self._cams or []

        # Cumulative start indices; length N+1 where N = num_episodes.
        self._starts: np.ndarray = np.cumsum([0] + self._lengths)
        self.num_frames: int = int(self._starts[-1])
        self.num_episodes: int = len(self._dirs)

        # LRU episode cache: {ep_idx: Episode}
        self._cache: OrderedDict = OrderedDict()
        self._cache_n: int = cache_episodes

        # Probe the first episode for fps and dimensions.
        s_dim, a_dim = self._probe_dims()

        tasks = sorted(set(self._tasks_per_ep))
        feats: dict = {
            "observation.state": {"dtype": "float32", "shape": (s_dim,), "names": None},
            "action": {"dtype": "float32", "shape": (a_dim,), "names": None},
        }
        for cam in self.camera_keys:
            feats[f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": None,
                "names": None,
            }

        self.meta = AbcdlDatasetMeta(
            features=feats,
            fps=self._fps,
            camera_keys=list(self.camera_keys),
            video_keys=[f"observation.images.{c}" for c in self.camera_keys],
            tasks=tasks,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probe_dims(self) -> tuple[int, int]:
        """Read fps and state/action dims from the first episode's metadata.

        Prefers the explicit ``fps`` field written by the writer (exact).
        Falls back to ``1e9 / tick_ns`` for older files that lack the field.
        """
        with open(os.path.join(self._dirs[0], "episode_metadata.json")) as f:
            m = json.load(f)
        if "fps" in m:
            self._fps: float = float(m["fps"])
        elif "tick_ns" in m:
            self._fps = 1e9 / int(m["tick_ns"])
        else:
            self._fps = float(m.get("fps", 30.0))
        return int(m["state_dim"]), int(m["action_dim"])

    def _get_episode(self, ep_idx: int):
        """Return a decoded Episode, loading and caching if necessary (LRU)."""
        if ep_idx in self._cache:
            # Move to end (most recently used).
            self._cache.move_to_end(ep_idx)
            return self._cache[ep_idx]
        ep = read_abcdl(self._dirs[ep_idx])
        self._cache[ep_idx] = ep
        self._cache.move_to_end(ep_idx)
        # Evict least-recently-used episode when cache exceeds capacity.
        if len(self._cache) > self._cache_n:
            self._cache.popitem(last=False)
        return ep

    def _locate(self, idx: int) -> tuple[int, int]:
        """Map a global frame index to (episode_index, frame_within_episode).

        Uses searchsorted on the cumulative start array.  The starts array has
        length N+1: [0, L0, L0+L1, ...].  searchsorted with side='right' on
        value idx gives the first start that is *strictly greater* than idx,
        so subtracting 1 yields the episode whose interval contains idx.

        Boundary check: idx == self._starts[k] is the *first* frame of episode
        k, and searchsorted(..., side='right') returns k+1, so ep_idx = k. ✓
        """
        ep_idx = int(np.searchsorted(self._starts, idx, side="right")) - 1
        frame = idx - int(self._starts[ep_idx])
        return ep_idx, frame

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        ep_idx, frame = self._locate(idx)
        ep = self._get_episode(ep_idx)
        n = ep.num_steps

        out: dict = {
            "observation.state": torch.from_numpy(
                ep.states[frame].astype(np.float32)
            ),
            "task": ep.meta.task,
            "timestamp": torch.tensor(frame / self._fps, dtype=torch.float32),
            "episode_index": torch.tensor(ep_idx, dtype=torch.long),
            "frame_index": torch.tensor(frame, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }

        # Action: single frame or chunked by delta_timestamps offsets.
        offsets = self.delta_timestamps.get("action")
        if offsets:
            # Each offset is a float in seconds; convert to frame row indices,
            # clamped to valid episode range [0, n-1].
            rows = [
                min(max(frame + int(round(o * self._fps)), 0), n - 1)
                for o in offsets
            ]
            # rows is a Python list of ints → fancy indexing → (K, A)
            out["action"] = torch.from_numpy(
                ep.actions[rows].astype(np.float32)
            )
        else:
            out["action"] = torch.from_numpy(ep.actions[frame].astype(np.float32))

        # Camera images: HWC uint8 → CHW float32 in [0, 1].
        for cam in self.camera_keys:
            hwc = ep.cameras[cam].frames[frame]  # (H, W, 3) uint8
            chw = (
                torch.from_numpy(np.ascontiguousarray(hwc))
                .permute(2, 0, 1)
                .float()
                .div(255.0)
            )
            out[f"observation.images.{cam}"] = chw

        return out


class MixtureDataset(Dataset):
    """Dataset that samples from multiple sub-datasets according to weights.

    Builds a static integer index map proportional to the supplied weights so
    that ``len(MixtureDataset)`` equals the sum of all sub-dataset lengths and
    each sub-dataset is sampled at the requested rate.
    """

    def __init__(self, datasets: list, weights: list):
        self.datasets = datasets
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        total = sum(len(d) for d in datasets)
        self._map: list[tuple[int, int]] = []
        for di, (d, wi) in enumerate(zip(datasets, w)):
            k = max(1, int(round(wi * total)))
            local = np.linspace(0, len(d) - 1, k).astype(int)
            self._map.extend((di, int(j)) for j in local)

    def __len__(self) -> int:
        return len(self._map)

    def __getitem__(self, i: int) -> dict:
        di, j = self._map[i]
        return self.datasets[di][j]
