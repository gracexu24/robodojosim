from __future__ import annotations

from typing import Any

from XPolicyLab.model_template import ModelTemplate

from robodojosim.inference import LeRobotPolicyRuntime


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


class Model(ModelTemplate):
    """LeRobot ACT/Diffusion checkpoint served through XPolicyLab."""

    def __init__(self, model_cfg):
        checkpoint = _config_value(model_cfg, "ckpt_name")
        if not checkpoint:
            raise ValueError("ckpt_name must point to a LeRobot pretrained_model directory")
        device = _config_value(model_cfg, "device", "cuda")
        self.runtime = LeRobotPolicyRuntime(checkpoint, device=device)

    def update_obs(self, obs):
        self.runtime.update(obs)

    def update_obs_batch(self, obs_list):
        raise NotImplementedError("bottle_lerobot supports only eval_batch=false")

    def get_action(self):
        # XPolicyLab requires one action chunk. LeRobot owns the ACT/Diffusion
        # queue internally, so one select_action call yields the next command.
        return [self.runtime.get_action()]

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("bottle_lerobot supports only eval_batch=false")

    def reset(self):
        self.runtime.reset()
