import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

import robodojosim.inference


class FakeRuntime:
    def __init__(self, checkpoint, device):
        self.checkpoint = checkpoint
        self.device = device
        self.reset_count = 0

    def update(self, observation):
        self.observation = observation

    def get_action(self):
        return {
            "left_ee_pose": np.zeros(7),
            "left_ee_joint_state": np.ones(1),
            "right_ee_pose": np.zeros(7),
            "right_ee_joint_state": np.ones(1),
        }

    def reset(self):
        self.reset_count += 1


def test_xpolicylab_model_returns_one_action_chunk(tmp_path, monkeypatch):
    template = types.ModuleType("XPolicyLab.model_template")
    template.ModelTemplate = object
    package = types.ModuleType("XPolicyLab")
    package.model_template = template
    monkeypatch.setitem(sys.modules, "XPolicyLab", package)
    monkeypatch.setitem(sys.modules, "XPolicyLab.model_template", template)
    monkeypatch.setattr(robodojosim.inference, "LeRobotPolicyRuntime", FakeRuntime)

    path = (
        Path(__file__).parents[1]
        / "integration"
        / "xpolicylab"
        / "policy"
        / "bottle_lerobot"
        / "model.py"
    )
    spec = importlib.util.spec_from_file_location("bottle_lerobot_model", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = module.Model({"ckpt_name": str(checkpoint), "device": "cuda"})
    model.update_obs({"state": {}, "vision": {}})
    actions = model.get_action()
    assert isinstance(actions, list) and len(actions) == 1
    assert set(actions[0]) == {
        "left_ee_pose",
        "left_ee_joint_state",
        "right_ee_pose",
        "right_ee_joint_state",
    }
    model.reset()
    assert model.runtime.reset_count == 1
