import importlib.util
import sys
import types
from pathlib import Path


class _RewardManager:
    def __init__(self, _num_envs=1):
        self.checked = None
        self.scored = None

    def is_A_on_B_bottom(self, **kwargs):
        return ("inside", kwargs["label_A"], kwargs["label_B"])

    def all_robot_back_to_origin(self):
        return ("home",)

    def is_all_gripper_open(self, open_threshold=0.6):
        return ("grippers_open", open_threshold)

    def check(self, checks):
        self.checked = checks

    def score(self, checks, scores, score_mode):
        self.scored = (checks, scores, score_mode)


def _overlay_module(monkeypatch):
    environment = types.ModuleType("env.environment")
    environment_task = types.ModuleType("env.environment.task_env")
    environment_task.TaskEnv = object
    reward_package = types.ModuleType("env.reward_manager")
    reward_module = types.ModuleType("env.reward_manager.reward_manager")
    reward_module.RewardManager = _RewardManager
    monkeypatch.setitem(sys.modules, "env", types.ModuleType("env"))
    monkeypatch.setitem(sys.modules, "env.environment", environment)
    monkeypatch.setitem(sys.modules, "env.environment.task_env", environment_task)
    monkeypatch.setitem(sys.modules, "env.reward_manager", reward_package)
    monkeypatch.setitem(sys.modules, "env.reward_manager.reward_manager", reward_module)
    path = (
        Path(__file__).parents[1]
        / "integration"
        / "robodojo"
        / "tasks"
        / "put_bottles_into_dustbin.py"
    )
    spec = importlib.util.spec_from_file_location("two_bottle_task_overlay", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_two_bottle_success_requires_released_bottles_but_not_home(monkeypatch):
    module = _overlay_module(monkeypatch)
    task = module.PutBottlesIntoDustbinCommon.__new__(module.PutBottlesIntoDustbinCommon)
    task.reward_manager = _RewardManager()

    task.run_reward()
    assert task.reward_manager.checked == [
        ("inside", "bottle0", "dustbin"),
        ("inside", "bottle1", "dustbin"),
        ("grippers_open", 0.8),
    ]

    task.get_score()
    checks, scores, mode = task.reward_manager.scored
    assert scores == [25, 100]
    assert mode == "transition"
    assert ("grippers_open", 0.8) in checks[-1]
    assert all("home" not in repr(check) for check in checks)
