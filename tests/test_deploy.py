import importlib.util
from pathlib import Path

import h5py
from test_adapter import FakeSceneManager

from robodojosim.mock_env import MockBottleEnv
from robodojosim.recording import EpisodeRecorder

_DEPLOY_PATH = (
    Path(__file__).parents[1] / "integration" / "xpolicylab" / "policy" / "bottle_scripted" / "deploy.py"
)
_SPEC = importlib.util.spec_from_file_location("bottle_scripted_deploy", _DEPLOY_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
eval_one_episode = _MODULE.eval_one_episode


class FakeModelClient:
    def call(self, func_name, **kwargs):
        assert func_name == "reset"


class FakeTaskEnv:
    def __init__(self, seed=7, allow_success=True):
        self.mock = MockBottleEnv(seed)
        self.scene_manager = FakeSceneManager(self.mock)
        self.current_env_seed_map = {0: seed}
        self.success = [True]
        self.end_flag = [False]
        self.allow_success = allow_success

    def get_obs(self):
        return self.mock.observation()

    def take_action(self, action):
        self.mock.step(action)

    def is_episode_end(self):
        if self.allow_success and self.mock.success:
            self.end_flag[0] = True
        return self.end_flag[0]


def test_deploy_uses_layout_seed_and_records_success(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBODOJOSIM_DATASET_DIR", str(tmp_path))
    task = FakeTaskEnv(seed=7)
    eval_one_episode(task, FakeModelClient())
    path = tmp_path / "episode_000007.hdf5"
    assert path.exists()
    with h5py.File(path, "r") as handle:
        assert bool(handle.attrs["success"])
        assert int(handle.attrs["seed"]) == 7

    # Existing data is left untouched, but the environment must still execute
    # so RoboDojo cannot mistake a skipped controller for success.
    original_mtime = path.stat().st_mtime_ns
    second = FakeTaskEnv(seed=7)
    eval_one_episode(second, FakeModelClient())
    assert second.end_flag[0] and second.mock.success
    assert path.stat().st_mtime_ns == original_mtime


def test_deploy_marks_unrewarded_finished_script_as_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBODOJOSIM_DATASET_DIR", str(tmp_path))
    task = FakeTaskEnv(seed=8, allow_success=False)
    eval_one_episode(task, FakeModelClient())
    assert task.end_flag[0]
    assert task.success[0] is False
    with h5py.File(tmp_path / "episode_000008.hdf5", "r") as handle:
        assert not bool(handle.attrs["success"])
        assert handle.attrs["termination_reason"] == "script_complete_without_full_reward"
    assert EpisodeRecorder.is_complete(tmp_path, 8)
    assert not EpisodeRecorder.is_complete(tmp_path, 8, require_success=True)
