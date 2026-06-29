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

from abcdl import hf
from abcdl.format.reader import open_episode


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

    Keeps an LRU cache of lightweight per-episode handles (an open torchcodec
    decoder + the small states/actions arrays). ``__getitem__`` decodes ONLY the
    requested frame via the analytic index — so shuffled access never decodes a
    whole episode, the access pattern abcdl is built for.
    """

    def __init__(
        self,
        root: str,
        delta_timestamps: Optional[dict] = None,
        camera_keys: Optional[list] = None,
        cache_episodes: int = 32,
        fmt: str = "abcdl",
        version: str = "latest",
        revision_root: Optional[str] = None,
    ):
        # If root is not an existing directory but looks like a Hub repo_id
        # (contains a "/" and is not an OS path), auto-download from the Hub.
        if not os.path.isdir(root) and root.count("/") == 1 and not root.startswith((".", "/", "~")):
            owner_name = root.replace("/", "__")
            dest = revision_root or os.path.join(
                os.path.expanduser("~"), ".cache", "abcdl", owner_name, version
            )
            os.makedirs(dest, exist_ok=True)
            root = hf.pull(repo_id=root, fmt=fmt, version=version, dest=dest)
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

        # LRU cache of per-episode handles {ep_idx: EpisodeHandle}. torchcodec
        # decoders are NOT fork-safe, so the cache is tagged with the owning PID
        # and reset when accessed from a different process (a DataLoader worker).
        self._cache: OrderedDict = OrderedDict()
        self._cache_n: int = cache_episodes
        self._cache_pid: int = os.getpid()

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
    # Hub convenience API (mirrors LeRobotDataset)
    # ------------------------------------------------------------------

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        fmt: str = "abcdl",
        version: str = "latest",
        root: Optional[str] = None,
        **kw,
    ) -> "AbcdlDataset":
        """Download *repo_id* from the Hub and return a ready-to-use dataset."""
        return cls(repo_id, fmt=fmt, version=version, revision_root=root, **kw)

    def push_to_hub(
        self,
        repo_id: str,
        version: str,
        fmt: str = "abcdl",
        message: Optional[str] = None,
        token: Optional[str] = None,
        update_card: bool = True,
        description: Optional[str] = None,
    ) -> str:
        """Upload ``self.root`` to the Hub as *repo_id*, tagged *version*.

        When *update_card* is true (default), a dataset card (``README.md``) is
        auto-generated from this dataset's metadata and uploaded to ``main`` — no
        manual post-processing, mirroring ``LeRobotDataset.push_to_hub``. Pass
        *description* to override the dataset blurb.
        """
        card = self._card_metadata(description) if update_card else None
        return hf.push(
            repo_id=repo_id,
            local_dir=self.root,
            fmt=fmt,
            version=version,
            message=message,
            token=token,
            card=card,
        )

    def _card_metadata(self, description: Optional[str] = None) -> dict:
        """Collect dataset stats for the auto-generated dataset card."""
        resolution = None
        with open(os.path.join(self._dirs[0], "episode_metadata.json")) as f:
            m = json.load(f)
        cr = m.get("camera_resolutions", {})
        if self.camera_keys and self.camera_keys[0] in cr:
            resolution = tuple(cr[self.camera_keys[0]])  # (width, height)
        return {
            "num_episodes": self.num_episodes,
            "num_frames": self.num_frames,
            "camera_keys": list(self.meta.camera_keys),
            "fps": self.meta.fps,
            "resolution": resolution,
            "robot_type": self.meta.robot_type,
            "tasks": list(self.meta.tasks),
            "state_dim": int(self.meta.features["observation.state"]["shape"][0]),
            "action_dim": int(self.meta.features["action"]["shape"][0]),
            "description": description,
        }

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
            self._fps = 30.0
        return int(m["state_dim"]), int(m["action_dim"])

    def _get_handle(self, ep_idx: int):
        """Return a per-episode random-access handle, opening + caching (LRU)."""
        # In a forked DataLoader worker the inherited decoders are invalid; drop them.
        pid = os.getpid()
        if pid != self._cache_pid:
            self._cache = OrderedDict()
            self._cache_pid = pid
        if ep_idx in self._cache:
            self._cache.move_to_end(ep_idx)
            return self._cache[ep_idx]
        h = open_episode(self._dirs[ep_idx])
        self._cache[ep_idx] = h
        self._cache.move_to_end(ep_idx)
        if len(self._cache) > self._cache_n:
            self._cache.popitem(last=False)
        return h

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
        h = self._get_handle(ep_idx)
        n = h.num_steps

        out: dict = {
            "observation.state": torch.from_numpy(h.states[frame].astype(np.float32)),
            "task": h.task,
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
            out["action"] = torch.from_numpy(h.actions[rows].astype(np.float32))
        else:
            out["action"] = torch.from_numpy(h.actions[frame].astype(np.float32))

        # Camera images: decode ONLY this frame, split per camera, HWC→CHW float [0,1].
        cams = h.frame(frame)
        for cam in self.camera_keys:
            chw = (
                torch.from_numpy(np.ascontiguousarray(cams[cam]))
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
