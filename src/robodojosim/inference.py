from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .lerobot_export import STATE_NAMES

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
WORKSPACE_MIN = np.asarray((-0.85, -0.55, 0.30), dtype=np.float32)
WORKSPACE_MAX = np.asarray((0.75, 0.40, 1.35), dtype=np.float32)


def observation_state(observation: Mapping[str, Any]) -> np.ndarray:
    state = observation.get("state")
    if not isinstance(state, Mapping):
        raise TypeError("observation does not contain a state mapping")
    parts: list[np.ndarray] = []
    for arm in ("left", "right"):
        pose = _numpy(state.get(f"{arm}_ee_pose")).reshape(-1)
        gripper = _numpy(state.get(f"{arm}_ee_joint_state")).reshape(-1)
        if pose.shape != (7,) or gripper.shape != (1,):
            raise ValueError(f"expected {arm} pose/gripper dimensions 7+1, got {pose.shape}+{gripper.shape}")
        parts.extend((pose, gripper))
    result = np.concatenate(parts).astype(np.float32, copy=False)
    if result.shape != (len(STATE_NAMES),) or not np.all(np.isfinite(result)):
        raise ValueError("robot state is not a finite 16-value Cartesian vector")
    return result


def observation_images(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    vision = observation.get("vision")
    if not isinstance(vision, Mapping):
        raise TypeError("observation does not contain a vision mapping")
    result: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        camera_data = vision.get(camera)
        if not isinstance(camera_data, Mapping):
            raise TypeError(f"observation is missing camera {camera!r}")
        value = camera_data.get("color", camera_data.get("colors"))
        image = _numpy(value)
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.moveaxis(image, 0, -1)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"camera {camera!r} has invalid RGB shape {image.shape}")
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and image.size and float(np.nanmax(image)) <= 1.0:
                image = np.rint(image * 255.0)
            image = np.clip(image, 0, 255).astype(np.uint8)
        result[f"observation.images.{camera}"] = np.ascontiguousarray(image)
    return result


def sanitize_action(action: Any) -> np.ndarray:
    value = _numpy(action).astype(np.float32, copy=False).reshape(-1)
    if value.shape != (len(STATE_NAMES),):
        raise ValueError(f"policy returned {value.shape}, expected a 16-value Cartesian action")
    if not np.all(np.isfinite(value)):
        raise ValueError("policy returned NaN or Inf")
    result = value.copy()
    for offset in (0, 8):
        result[offset : offset + 3] = np.clip(result[offset : offset + 3], WORKSPACE_MIN, WORKSPACE_MAX)
        quaternion = result[offset + 3 : offset + 7]
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-6:
            raise ValueError("policy returned a zero-length quaternion")
        result[offset + 3 : offset + 7] = quaternion / norm
        result[offset + 7] = float(np.clip(result[offset + 7], 0.0, 1.0))
    return result


def action_mapping(action: Any) -> dict[str, np.ndarray]:
    value = sanitize_action(action)
    return {
        "left_ee_pose": value[0:7],
        "left_ee_joint_state": value[7:8],
        "right_ee_pose": value[8:15],
        "right_ee_joint_state": value[15:16],
    }


class LeRobotPolicyRuntime:
    """Lazy LeRobot checkpoint loader used by the XPolicyLab policy server."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cuda"):
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {self.checkpoint}")
        self.device = device
        self.torch = _import_torch()
        self.config, self.policy = self._load_policy()
        self.preprocessor, self.postprocessor = self._load_processors()
        self.policy.eval()

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def update(self, observation: Mapping[str, Any]) -> None:
        batch: dict[str, Any] = {
            "observation.state": self.torch.from_numpy(observation_state(observation)),
        }
        for name, image in observation_images(observation).items():
            tensor = self.torch.from_numpy(np.moveaxis(image, -1, 0)).float().div_(255.0)
            batch[name] = tensor
        # LeRobot's saved preprocessor adds the batch dimension, moves tensors to the
        # configured device, and applies the checkpoint's normalization statistics.
        self.observation = batch

    def get_action(self) -> dict[str, np.ndarray]:
        if not hasattr(self, "observation"):
            raise RuntimeError("update() must be called before get_action()")
        with self.torch.inference_mode():
            batch = self.preprocessor(self.observation)
            action = self.policy.select_action(batch)
            action = self.postprocessor(action)
        return action_mapping(action)

    def _load_policy(self) -> tuple[Any, Any]:
        try:
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class
        except ImportError as exc:
            raise RuntimeError("LeRobot is required in the policy environment; install robodojosim[train]") from exc
        config = PreTrainedConfig.from_pretrained(self.checkpoint)
        config.device = self.device
        config.pretrained_path = str(self.checkpoint)
        policy_class = get_policy_class(config.type)
        policy = policy_class.from_pretrained(str(self.checkpoint), config=config)
        return config, policy.to(self.device)

    def _load_processors(self) -> tuple[Any, Any]:
        required = ("preprocessor_config.json", "postprocessor_config.json")
        missing = [name for name in required if not (self.checkpoint / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"checkpoint is missing LeRobot processor configuration: {', '.join(missing)}"
            )
        try:
            from lerobot.policies.factory import make_pre_post_processors
        except ImportError as exc:
            raise RuntimeError("this checkpoint requires LeRobot 0.4.4 processor support") from exc
        return make_pre_post_processors(
            policy_cfg=self.config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )


def _numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required in the policy inference environment") from exc
    return torch
