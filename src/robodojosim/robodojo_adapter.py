from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .types import ObjectState, Pose, SceneSnapshot


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


class RoboDojoSceneAdapter:
    """Small boundary around RoboDojo's simulator-only scene API.

    This module deliberately has no Isaac Sim imports, so its behavior can be
    unit-tested on development machines. The passed environment owns all
    simulator objects.
    """

    def __init__(self, task_env: Any, env_idx: int = 0):
        self.task_env = task_env
        self.env_idx = env_idx

    def snapshot(self, observation: Mapping[str, Any]) -> SceneSnapshot:
        state = observation["state"]
        arms = {
            "left": Pose.from_array(state["left_ee_pose"]),
            "right": Pose.from_array(state["right_ee_pose"]),
        }
        grippers = {
            "left": float(_numpy(state.get("left_ee_joint_state", [1.0])).reshape(-1)[0]),
            "right": float(_numpy(state.get("right_ee_joint_state", [1.0])).reshape(-1)[0]),
        }
        bottles = {label: self._rigid(label) for label in ("bottle0", "bottle1", "bottle2", "bottle3")}
        dustbin = self._geometry("dustbin")
        instruction = observation.get(
            "instruction", "Pick up the bottles and throw them into the dustbin, using handover when needed."
        )
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else ""
        teacher = {
            "bottle_poses": {label: value.pose.as_array().tolist() for label, value in bottles.items()},
            "dustbin_pose": dustbin.pose.as_array().tolist(),
        }
        return SceneSnapshot(arms, grippers, bottles, dustbin, str(instruction), teacher)

    def _rigid(self, label: str) -> ObjectState:
        instance_name = self._instance_name(label)
        objects = self.task_env.scene_manager.get_objects(
            env_ids=[self.env_idx], object_name=instance_name, object_type="rigid"
        )
        if len(objects) != 1:
            raise RuntimeError(
                f"expected one rigid object for label {label!r} "
                f"(instance {instance_name!r}), found {list(objects)}"
            )
        obj = next(iter(objects.values()))
        try:
            position, orientation, bbox = obj.get_bbox(is_relative=True)
            return ObjectState(Pose(position, orientation), bbox)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            position, orientation = obj.get_local_pose()
            return ObjectState(Pose(_numpy(position), _numpy(orientation)))

    def _geometry(self, label: str) -> ObjectState:
        instance_name = self._instance_name(label)
        objects = self.task_env.scene_manager.get_objects(
            env_ids=[self.env_idx], object_name=instance_name, object_type="geometry"
        )
        if len(objects) != 1:
            raise RuntimeError(
                f"expected one geometry object for label {label!r} "
                f"(instance {instance_name!r}), found {list(objects)}"
            )
        obj = next(iter(objects.values()))
        state = obj.get_state(is_relative=True)
        return ObjectState(Pose.from_array(_numpy(state["root_pose"])))

    def _instance_name(self, label: str) -> str:
        layout_manager = getattr(self.task_env.scene_manager, "layout_manager", None)
        if layout_manager is None:
            layout_manager = getattr(self.task_env, "layout_manager", None)
        if layout_manager is None:
            raise RuntimeError("RoboDojo environment does not expose a layout manager")
        instance_name = layout_manager.get_instance_name(env_idx=self.env_idx, label=label)
        if not instance_name:
            raise RuntimeError(f"no scene instance found for task label {label!r} in env {self.env_idx}")
        return str(instance_name)


def teacher_frame(snapshot: SceneSnapshot, phase: str, bottle: str | None, active_arm: str | None) -> dict[str, Any]:
    """Metadata stored under /teacher, separate from policy observations."""

    result: dict[str, Any] = {
        "phase": phase,
        "bottle": bottle or "",
        "active_arm": active_arm or "",
        "dustbin_pose": snapshot.dustbin.pose.as_array(),
    }
    for label, bottle_state in snapshot.bottles.items():
        result[f"{label}_pose"] = bottle_state.pose.as_array()
    return result
