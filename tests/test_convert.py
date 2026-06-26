import shutil

import numpy as np
import pytest

from abcdl.convert.mcap_abcdl import mcap_to_abcdl
from abcdl.format.reader import read_abcdl
from abcdl.mcap.reader import read_mcap

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_mcap_to_abcdl_fixed_clock(sample_mcap, tmp_path):
    out = tmp_path / "ep"
    mcap_to_abcdl(str(sample_mcap), str(out))
    ep = read_abcdl(str(out))
    assert ep.meta.alignment == "fixed_clock_30hz_causal"
    assert abs(ep.meta.fps - 30.0) < 1e-6
    # 14-D state preserved; frame count equals state count
    assert ep.states.shape[1] == 14
    for name in ep.meta.cameras:
        assert len(ep.cameras[name].frames) == ep.num_steps
