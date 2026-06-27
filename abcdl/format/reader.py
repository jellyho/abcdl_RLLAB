"""Read the abcdl MP4+binary layout back into an Episode (analytic frame index)."""

from __future__ import annotations

import json
import os

import numpy as np

from abcdl.constants import FPS, TICK_NS, TICKS_PER_FRAME
from abcdl.episode import CameraStream, Episode, EpisodeMeta


def _synth_frame_map(num_steps: int) -> str:
    """Synthesized constant-frame-rate map (analytic index; no file probing)."""
    frames = [{"pts": TICKS_PER_FRAME * i, "duration": TICKS_PER_FRAME,
               "key_frame": 1 if i % FPS == 0 else 0} for i in range(num_steps)]
    return json.dumps({"frames": frames})


def _decode_all_frames(mp4_path: str, num_steps: int) -> np.ndarray:
    """Decode every frame via torchcodec with a synthesized CFR frame map."""
    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(mp4_path, custom_frame_mappings=_synth_frame_map(num_steps))
    out = np.empty((num_steps, *dec[0].shape), dtype=np.uint8)  # (C,H,W) per frame
    for i in range(num_steps):
        out[i] = dec[i].numpy()
    return out  # (T, C, Hstack, W)


class EpisodeHandle:
    """Lightweight random-access handle for one abcdl episode.

    Unlike :func:`read_abcdl` (which decodes the whole video up front), this keeps
    a torchcodec decoder open and decodes ONLY the requested frame on demand — the
    access pattern the abcdl analytic frame index is designed for, and what a
    shuffled training DataLoader needs. States/actions are small and kept in memory.
    """

    def __init__(self, decoder, states: np.ndarray, actions: np.ndarray,
                 cam_names: list, cam_h: int, task: str, fps: float, num_steps: int):
        self.decoder = decoder
        self.states = states
        self.actions = actions
        self.cam_names = cam_names
        self.cam_h = cam_h
        self.task = task
        self.fps = fps
        self.num_steps = num_steps

    def frame(self, i: int) -> dict:
        """Decode ONLY frame *i*; return ``{cam_name: (H, W, 3) uint8}``."""
        stacked = np.transpose(self.decoder[i].numpy(), (1, 2, 0))  # (Hstack, W, C)
        h = self.cam_h
        return {name: stacked[k * h:(k + 1) * h, :, :]
                for k, name in enumerate(self.cam_names)}


def open_episode(in_dir: str) -> EpisodeHandle:
    """Open an abcdl episode for efficient random per-frame access (no full decode)."""
    from torchcodec.decoders import VideoDecoder

    with open(os.path.join(in_dir, "episode_metadata.json")) as f:
        meta = json.load(f)
    T = int(meta["num_steps"])
    sd, ad = int(meta["state_dim"]), int(meta["action_dim"])
    sa = np.fromfile(os.path.join(in_dir, "states_actions.bin"), dtype="<f8").reshape(T, sd + ad)
    names = list(meta["cameras"])
    cam_h = int(meta["camera_resolutions"][names[0]][1])  # (width, height) -> height
    tick_ns = int(meta.get("tick_ns", TICK_NS))
    fps = float(meta["fps"]) if "fps" in meta else (1e9 / tick_ns if tick_ns else 30.0)
    dec = VideoDecoder(os.path.join(in_dir, "combined_camera-images-rgb.mp4"),
                       custom_frame_mappings=_synth_frame_map(T))
    return EpisodeHandle(dec, sa[:, :sd], sa[:, sd:], names, cam_h, meta["task_name"], fps, T)


def read_abcdl(in_dir: str) -> Episode:
    with open(os.path.join(in_dir, "episode_metadata.json")) as f:
        meta = json.load(f)
    T = int(meta["num_steps"])
    state_dim = int(meta["state_dim"])
    action_dim = int(meta["action_dim"])
    row = state_dim + action_dim

    sa = np.fromfile(os.path.join(in_dir, "states_actions.bin"), dtype="<f8").reshape(T, row)
    states, actions = sa[:, :state_dim], sa[:, state_dim:]

    stacked = _decode_all_frames(os.path.join(in_dir, "combined_camera-images-rgb.mp4"), T)
    stacked = np.transpose(stacked, (0, 2, 3, 1))  # (T, Hstack, W, C)
    names = list(meta["cameras"])
    h = stacked.shape[1] // len(names)

    tick_ns = int(meta.get("tick_ns", TICK_NS))
    t0 = int(meta.get("t0_ns", 0))
    ts = t0 + tick_ns * np.arange(T, dtype=np.int64)

    cams = {}
    for i, name in enumerate(names):
        frames = stacked[:, i * h:(i + 1) * h, :, :]
        w = frames.shape[2]
        cams[name] = CameraStream(frames=frames, timestamps=ts,
                                  width=w, height=h, codec="h264")
    em = EpisodeMeta(
        task=meta["task_name"], fps=1e9 / tick_ns, cameras=names,
        camera_resolutions={k: tuple(v) for k, v in meta["camera_resolutions"].items()},
        camera_codecs={k: "h264" for k in names},
        operator_id=meta.get("operator_id"), alignment=meta.get("alignment", "native"),
        t0_ns=t0, tick_ns=tick_ns, session_id=meta.get("session_id"))
    return Episode(states=states, actions=actions, timestamps=ts, cameras=cams, meta=em)
