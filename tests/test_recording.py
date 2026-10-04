import json

import h5py
import numpy as np

from robodojosim.controller import BottleController
from robodojosim.image_codec import decode_jpeg
from robodojosim.mock_env import MockBottleEnv
from robodojosim.recording import EpisodeRecorder, validate_episode


def test_atomic_recorder_writes_xpolicylab_layout(tmp_path):
    env = MockBottleEnv(2)
    controller = BottleController()
    controller.reset(env.snapshot())
    recorder = EpisodeRecorder(tmp_path, 4, seed=2, instruction=env.snapshot().instruction)
    while not controller.done:
        planned = controller.next_action(env.snapshot())
        recorder.append(env.observation(), planned.action, teacher=planned.privileged)
        env.step(planned.action)
    final = recorder.close(success=env.success, score=100)

    assert final.exists()
    assert not recorder.partial_path.exists()
    assert validate_episode(final) == []
    with h5py.File(final, "r") as handle:
        assert bool(handle.attrs["complete"])
        assert handle["state/left_ee_poses"].shape == (recorder.length, 7)
        assert handle["action/right_ee_joint_states"].shape == (recorder.length, 1)
        assert handle["vision/cam_head/colors"].shape == (recorder.length, 16, 16, 3)
        assert handle["vision/cam_head/shape"].shape == (2,)
        assert handle["teacher/phase"].shape == (recorder.length,)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["episodes"][0]["success"] is True
    assert EpisodeRecorder.is_complete(tmp_path, 4)


def test_abort_is_not_resumable_as_complete(tmp_path):
    recorder = EpisodeRecorder(tmp_path, 3, seed=0, instruction="test")
    recorder.abort("test abort")
    assert recorder.partial_path.exists()
    assert not EpisodeRecorder.is_complete(tmp_path, 3)


def test_jpeg_recording_uses_xpolicylab_variable_length_stream(tmp_path):
    env = MockBottleEnv()
    controller = BottleController()
    controller.reset(env.snapshot())
    planned = controller.next_action()
    observation = env.observation()
    recorder = EpisodeRecorder(tmp_path, 9, seed=0, instruction="test", jpeg_quality=90)
    recorder.append(observation, planned.action)
    path = recorder.close(success=True)
    with h5py.File(path, "r") as handle:
        encoded = handle["vision/cam_head/colors"][0]
        assert handle["vision/cam_head/colors"].shape == (1,)
        assert handle.attrs["image_encoding"] == "xpolicylab_jpeg"
    assert b"XPL-RGB1" in encoded.tobytes()
    decoded = decode_jpeg(encoded)
    assert decoded.shape == (16, 16, 3)
    assert np.max(np.abs(decoded.astype(int) - observation["vision"]["cam_head"]["color"].astype(int))) <= 2
