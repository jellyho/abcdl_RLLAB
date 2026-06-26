from abcdl.mcap.reader import read_mcap


def test_read_sample_episode(sample_mcap):
    ep = read_mcap(str(sample_mcap))
    assert ep.meta.task == "open the umbrella"
    assert ep.states.shape[1] == 14 and ep.actions.shape[1] == 14
    assert ep.num_steps > 100
    assert set(ep.cameras) >= {"top", "left_wrist", "right_wrist"}
    # cameras carry encoded h264 chunks (bytes) at native timing
    top = ep.cameras["top"]
    assert top.codec == "h264" and isinstance(top.frames[0], (bytes, bytearray))
    assert ep.meta.operator_id  # metadata-drift tolerant read populated this
    assert ep.ee_poses is not None and ep.ee_poses["left"].shape[1:] == (4, 4)
