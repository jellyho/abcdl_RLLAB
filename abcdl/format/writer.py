"""Write an Episode to the abcdl MP4+binary on-disk layout."""

from __future__ import annotations

import json
import os

import numpy as np

from abcdl.constants import TICK_NS
from abcdl.episode import Episode
from abcdl.format.encode import encode_strict_h264


def write_abcdl(episode: Episode, out_dir: str) -> None:
    episode.validate()
    os.makedirs(out_dir, exist_ok=True)
    T = episode.num_steps

    sa = np.concatenate([np.asarray(episode.states, np.float64),
                         np.asarray(episode.actions, np.float64)], axis=1)
    sa.astype("<f8").tofile(os.path.join(out_dir, "states_actions.bin"))

    stacks = []
    h = w = None
    for name in episode.meta.cameras:
        frames = np.asarray(episode.cameras[name].frames, np.uint8)
        if frames.shape[0] != T:
            raise ValueError(f"camera {name} has {frames.shape[0]} frames, expected {T}")
        if h is None:
            h, w = frames.shape[1], frames.shape[2]
        elif (frames.shape[1], frames.shape[2]) != (h, w):
            raise ValueError("all cameras must share H,W for vertical stacking")
        stacks.append(frames)
    combined = np.concatenate(stacks, axis=1)  # vstack along height
    encode_strict_h264(combined, os.path.join(out_dir, "combined_camera-images-rgb.mp4"))

    meta = {
        "task_name": episode.meta.task,
        "cameras": list(episode.meta.cameras),
        "camera_resolutions": {k: list(v) for k, v in episode.meta.camera_resolutions.items()},
        "alignment": episode.meta.alignment,
        "t0_ns": int(episode.meta.t0_ns or 0),
        "tick_ns": int(episode.meta.tick_ns or TICK_NS),
        "num_steps": T,
        "state_dim": int(episode.states.shape[1]),
        "action_dim": int(episode.actions.shape[1]),
        "operator_id": episode.meta.operator_id,
        "session_id": episode.meta.session_id,
    }
    with open(os.path.join(out_dir, "episode_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
