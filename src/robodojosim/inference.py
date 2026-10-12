from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .lerobot_export import STATE_NAMES

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
WORKSPACE_MIN = np.asarray((-0.85, -0.55, 0.30), dtype=np.float32)
WORKSPACE_MAX = np.asarray((0.75, 0.40, 1.35), dtype=np.float32)


@dataclass(frozen=True)
class PolicySafetyConfig:
    """Stateful safeguards for Cartesian learned-policy commands at 25 Hz."""

    max_position_step_m: float = 0.015
    max_rotation_step_rad: float = math.radians(5.0)
    close_threshold: float = 0.25
    open_threshold: float = 0.80
    close_debounce_steps: int = 2
    open_debounce_steps: int = 8
    minimum_closed_steps: int = 20

    def __post_init__(self) -> None:
        if self.max_position_step_m <= 0 or self.max_rotation_step_rad <= 0:
            raise ValueError("Cartesian rate limits must be positive")
        if not 0 <= self.close_threshold < self.open_threshold <= 1:
            raise ValueError("gripper thresholds must satisfy 0 <= close < open <= 1")
        if min(self.close_debounce_steps, self.open_debounce_steps, self.minimum_closed_steps) < 1:
            raise ValueError("gripper debounce and hold steps must be positive")


class PolicySafetyFilter:
    """Rate-limit learned poses and turn continuous gripper output into latched commands."""

    def __init__(self, config: PolicySafetyConfig | None = None):
        self.config = config or PolicySafetyConfig()
        self.reset()

    def reset(self) -> None:
        self._previous_action: np.ndarray | None = None
        self._gripper_open = [True, True]
        self._close_votes = [0, 0]
        self._open_votes = [0, 0]
        self._closed_steps = [0, 0]
        self._raw_gripper_category = [None, None]
        self._stats = {
            "steps": 0,
            "position_limited": [0, 0],
            "rotation_limited": [0, 0],
            "raw_gripper_category_changes": [0, 0],
            "close_commands": [0, 0],
            "open_commands": [0, 0],
            "suppressed_open_steps": [0, 0],
        }

    def apply(self, action: Any, observed_state: Any) -> np.ndarray:
        result = sanitize_action(action)
        observed = _validated_state_vector(observed_state)
        if self._previous_action is None:
            self._previous_action = observed.copy()
            for arm_index, offset in enumerate((0, 8)):
                self._gripper_open[arm_index] = bool(observed[offset + 7] >= self.config.open_threshold)
                self._previous_action[offset + 7] = 1.0 if self._gripper_open[arm_index] else 0.0

        for arm_index, offset in enumerate((0, 8)):
            result[offset : offset + 3] = self._rate_limit_position(
                self._previous_action[offset : offset + 3], result[offset : offset + 3], arm_index
            )
            result[offset + 3 : offset + 7] = self._rate_limit_rotation(
                self._previous_action[offset + 3 : offset + 7], result[offset + 3 : offset + 7], arm_index
            )
            result[offset + 7] = self._filter_gripper(float(result[offset + 7]), arm_index)

        self._previous_action = result.copy()
        self._stats["steps"] += 1
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "replan_every_step": True,
            "max_position_step_m": self.config.max_position_step_m,
            "max_rotation_step_deg": math.degrees(self.config.max_rotation_step_rad),
            **self._stats,
        }

    def _rate_limit_position(self, previous: np.ndarray, target: np.ndarray, arm_index: int) -> np.ndarray:
        delta = target - previous
        distance = float(np.linalg.norm(delta))
        if distance <= self.config.max_position_step_m:
            return target
        self._stats["position_limited"][arm_index] += 1
        return previous + delta * (self.config.max_position_step_m / distance)

    def _rate_limit_rotation(self, previous: np.ndarray, target: np.ndarray, arm_index: int) -> np.ndarray:
        previous = previous / np.linalg.norm(previous)
        target = target / np.linalg.norm(target)
        dot = float(np.dot(previous, target))
        if dot < 0:
            target = -target
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        angle = 2.0 * math.acos(dot)
        if angle <= self.config.max_rotation_step_rad:
            return target.astype(np.float32, copy=False)
        self._stats["rotation_limited"][arm_index] += 1
        fraction = self.config.max_rotation_step_rad / angle
        if dot > 0.9995:
            interpolated = previous + fraction * (target - previous)
        else:
            half_angle = math.acos(dot)
            denominator = math.sin(half_angle)
            interpolated = (
                math.sin((1.0 - fraction) * half_angle) / denominator * previous
                + math.sin(fraction * half_angle) / denominator * target
            )
        return (interpolated / np.linalg.norm(interpolated)).astype(np.float32, copy=False)

    def _filter_gripper(self, raw: float, arm_index: int) -> float:
        if raw <= self.config.close_threshold:
            category = "close"
        elif raw >= self.config.open_threshold:
            category = "open"
        else:
            category = "hold"
        previous_category = self._raw_gripper_category[arm_index]
        if previous_category is not None and category != previous_category:
            self._stats["raw_gripper_category_changes"][arm_index] += 1
        self._raw_gripper_category[arm_index] = category

        if category == "close":
            self._close_votes[arm_index] += 1
            self._open_votes[arm_index] = 0
        elif category == "open":
            self._open_votes[arm_index] += 1
            self._close_votes[arm_index] = 0
        else:
            self._close_votes[arm_index] = 0
            self._open_votes[arm_index] = 0

        if self._gripper_open[arm_index]:
            if self._close_votes[arm_index] >= self.config.close_debounce_steps:
                self._gripper_open[arm_index] = False
                self._closed_steps[arm_index] = 0
                self._close_votes[arm_index] = 0
                self._stats["close_commands"][arm_index] += 1
        else:
            self._closed_steps[arm_index] += 1
            can_open = self._closed_steps[arm_index] >= self.config.minimum_closed_steps
            if category == "open" and not (
                can_open and self._open_votes[arm_index] >= self.config.open_debounce_steps
            ):
                self._stats["suppressed_open_steps"][arm_index] += 1
            if can_open and self._open_votes[arm_index] >= self.config.open_debounce_steps:
                self._gripper_open[arm_index] = True
                self._open_votes[arm_index] = 0
                self._stats["open_commands"][arm_index] += 1
        return 1.0 if self._gripper_open[arm_index] else 0.0


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
        self.safety_filter = PolicySafetyFilter()
        self._force_replan_each_step()
        self.policy.eval()

    def reset(self) -> None:
        if self.safety_filter.summary()["steps"]:
            print(f"[robodojosim.safety] {json.dumps(self.safety_filter.summary(), sort_keys=True)}", flush=True)
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.safety_filter.reset()

    def update(self, observation: Mapping[str, Any]) -> None:
        state = observation_state(observation)
        batch: dict[str, Any] = {
            "observation.state": self.torch.from_numpy(state),
        }
        for name, image in observation_images(observation).items():
            tensor = self.torch.from_numpy(np.moveaxis(image, -1, 0)).float().div_(255.0)
            batch[name] = tensor
        # LeRobot's saved preprocessor adds the batch dimension, moves tensors to the
        # configured device, and applies the checkpoint's normalization statistics.
        self.observation = batch
        self.observed_state = state

    def get_action(self) -> dict[str, np.ndarray]:
        if not hasattr(self, "observation"):
            raise RuntimeError("update() must be called before get_action()")
        with self.torch.inference_mode():
            batch = self.preprocessor(self.observation)
            action = self.policy.select_action(batch)
            action = self.postprocessor(action)
        safe_action = self.safety_filter.apply(action, self.observed_state)
        return action_mapping(safe_action)

    def _force_replan_each_step(self) -> None:
        """Disable LeRobot's open-loop action queue without changing checkpoint weights."""
        for value in (self.config, getattr(self.policy, "config", None)):
            if value is not None and hasattr(value, "n_action_steps"):
                value.n_action_steps = 1
        if hasattr(self.policy, "reset"):
            self.policy.reset()

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
        # LeRobot 0.4.4 saves these pipelines under the policy-prefixed names
        # consumed by make_pre_post_processors().  The older generic names are
        # used only by the migration utility and are not emitted by training.
        required = ("policy_preprocessor.json", "policy_postprocessor.json")
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


def _validated_state_vector(value: Any) -> np.ndarray:
    state = _numpy(value).astype(np.float32, copy=False).reshape(-1)
    if state.shape != (len(STATE_NAMES),) or not np.all(np.isfinite(state)):
        raise ValueError("observed state is not a finite 16-value Cartesian vector")
    result = state.copy()
    for offset in (0, 8):
        quaternion = result[offset + 3 : offset + 7]
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-6:
            raise ValueError("observed state contains a zero-length quaternion")
        result[offset + 3 : offset + 7] = quaternion / norm
    return result


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required in the policy inference environment") from exc
    return torch
