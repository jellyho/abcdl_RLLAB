import os
from pathlib import Path

import pytest

_DEFAULT_SAMPLE = Path(
    "~/.cache/huggingface/hub/datasets--XDOF--ABC-130k/snapshots/"
    "071311db1ac281848714bff024f9c6f944837c40/data/val/open_the_umbrella/"
    "episode_ebcbf9d1-d42b-4ef1-b3be-cad29b11edf8/episode.mcap"
).expanduser()


@pytest.fixture
def sample_mcap() -> Path:
    path = Path(os.environ.get("ABCDL_SAMPLE_MCAP", _DEFAULT_SAMPLE)).expanduser()
    if not path.exists():
        pytest.skip(f"sample mcap not found at {path}")
    return path
