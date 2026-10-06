from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import bbox_top, interpolate_pose
from .types import Action, Pose, SceneSnapshot


class Phase(str, Enum):
    APPROACH = "approach"
    GRASP = "grasp"
    CLOSE = "close"
    LIFT = "lift"
    TRANSIT = "transit"
    HANDOVER = "handover"
    RELEASE = "release"
    RETREAT = "retreat"
    HOME = "home"
    DONE = "done"


class SafetyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ControllerConfig:
    max_translation_step: float = 0.035
    approach_height: float = 0.14
    lift_height: float = 0.20
    grasp_clearance: float = 0.015
    grasp_center_offset: float | None = None
    drop_clearance: float = 0.18
    retreat_height: float = 0.14
    bottle_fallback_half_height: float = 0.045
    dustbin_fallback_half_height: float = 0.25
    direct_left_max_x: float = 0.08
    open_value: float = 1.0
    closed_value: float = 0.0
    gripper_hold_steps: int = 3
    lift_hold_steps: int = 1
    max_actions: int = 650
    bottle_limit: int | None = None
    stop_after_lift: bool = False
    use_overhead_approach: bool = True
    workspace_min: tuple[float, float, float] = (-0.85, -0.55, 0.30)
    workspace_max: tuple[float, float, float] = (0.75, 0.40, 1.35)
    # These are tool-center positions, not bottle positions. They are expected
    # to need a one-time calibration for the exact X5 gripper asset.
    handover_left_position: tuple[float, float, float] = (-0.025, -0.02, 1.02)
    handover_right_position: tuple[float, float, float] = (0.025, -0.02, 1.02)
    left_grasp_quaternion: tuple[float, float, float, float] | None = None
    right_grasp_quaternion: tuple[float, float, float, float] | None = None
    left_handover_quaternion: tuple[float, float, float, float] | None = None
    right_handover_quaternion: tuple[float, float, float, float] | None = None
    vary_bottle_order: bool = False
    height_jitter: float = 0.0
    drop_xy_jitter: float = 0.0

    def __post_init__(self) -> None:
        if self.max_translation_step <= 0:
            raise ValueError("max_translation_step must be positive")
        if self.gripper_hold_steps < 1 or self.lift_hold_steps < 1 or self.max_actions < 1:
            raise ValueError("gripper_hold_steps, lift_hold_steps, and max_actions must be positive")
        if self.bottle_limit is not None and self.bottle_limit < 1:
            raise ValueError("bottle_limit must be positive when provided")
        if np.any(np.asarray(self.workspace_min) >= np.asarray(self.workspace_max)):
            raise ValueError("workspace_min must be below workspace_max")
        if self.height_jitter < 0 or self.drop_xy_jitter < 0:
            raise ValueError("trajectory jitter values cannot be negative")

    @classmethod
    def from_json(cls, path: str | Path) -> ControllerConfig:
        with Path(path).open(encoding="utf-8") as handle:
            data = json.load(handle)
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown controller settings: {unknown}")
        return cls(**data)


@dataclass(frozen=True)
class PlannedAction:
    action: Action
    phase: Phase
    bottle: str | None
    active_arm: str | None
    privileged: Mapping[str, Any]


@dataclass(frozen=True)
class _Event:
    phase: Phase
    poses: Mapping[str, Pose]
    grippers: Mapping[str, float]
    bottle: str | None
    active_arm: str | None
    repeat: int = 1


class BottleController:
    """Deterministic Cartesian teacher for the dual-X5 bottle task.

    The controller emits one RoboDojo EE action per call. It uses privileged
    object poses only to construct the expert plan; those poses are never added
    to the policy observation by the recorder.
    """

    def __init__(self, config: ControllerConfig | None = None, *, trajectory_variant: int = 0):
        self.config = config or ControllerConfig()
        self.trajectory_variant = int(trajectory_variant)
        self._steps: list[PlannedAction] = []
        self._cursor = 0
        self._home: dict[str, Pose] = {}

    @property
    def done(self) -> bool:
        return bool(self._steps) and self._cursor >= len(self._steps)

    @property
    def phase(self) -> Phase:
        if not self._steps or self.done:
            return Phase.DONE
        return self._steps[self._cursor].phase

    @property
    def planned_action_count(self) -> int:
        return len(self._steps)

    @property
    def executed_action_count(self) -> int:
        return self._cursor

    def reset(self, snapshot: SceneSnapshot) -> None:
        self._home = dict(snapshot.arms)
        events = self._build_events(snapshot)
        self._steps = self._compile(events, snapshot)
        self._cursor = 0
        if len(self._steps) > self.config.max_actions:
            raise SafetyError(
                f"plan contains {len(self._steps)} actions, exceeding max_actions={self.config.max_actions}"
            )

    def next_action(self, snapshot: SceneSnapshot | None = None) -> PlannedAction:
        if not self._steps:
            if snapshot is None:
                raise RuntimeError("controller must be reset with a scene snapshot")
            self.reset(snapshot)
        if self.done:
            raise StopIteration("scripted controller is done")
        result = self._steps[self._cursor]
        self._cursor += 1
        return result

    def _orientation(self, arm: str, snapshot: SceneSnapshot, handover: bool = False) -> np.ndarray:
        name = f"{arm}_{'handover' if handover else 'grasp'}_quaternion"
        configured = getattr(self.config, name)
        return snapshot.arms[arm].quaternion if configured is None else np.asarray(configured, dtype=np.float64)

    def _build_events(self, snapshot: SceneSnapshot) -> list[_Event]:
        poses = dict(snapshot.arms)
        grippers = {"left": self.config.open_value, "right": self.config.open_value}
        events: list[_Event] = []
        rng = np.random.default_rng(self.trajectory_variant)
        dustbin_top = bbox_top(
            snapshot.dustbin.pose, snapshot.dustbin.bbox, self.config.dustbin_fallback_half_height
        )

        # Left-side bottles first keeps the bin-side workspace uncluttered.
        bottle_order = sorted(snapshot.bottles, key=lambda name: snapshot.bottles[name].pose.position[0])
        if self.config.vary_bottle_order:
            rng.shuffle(bottle_order)
        if self.config.bottle_limit is not None:
            bottle_order = bottle_order[: self.config.bottle_limit]
        for label in bottle_order:
            height_delta = float(rng.uniform(-self.config.height_jitter, self.config.height_jitter))
            drop_xy = rng.uniform(-self.config.drop_xy_jitter, self.config.drop_xy_jitter, size=2)
            drop_position = np.array(
                [
                    snapshot.dustbin.pose.position[0] + drop_xy[0],
                    snapshot.dustbin.pose.position[1] + drop_xy[1],
                    dustbin_top + self.config.drop_clearance + height_delta,
                ]
            )
            bottle = snapshot.bottles[label]
            pick_arm = "left" if bottle.pose.position[0] <= self.config.direct_left_max_x else "right"
            orientation = self._orientation(pick_arm, snapshot)
            top = bbox_top(bottle.pose, bottle.bbox, self.config.bottle_fallback_half_height)
            grasp_position = bottle.pose.position.copy()
            if self.config.grasp_center_offset is None:
                grasp_position[2] = top + self.config.grasp_clearance
            else:
                # RoboDojo's bottle assets include upright and sideways poses.
                # A tool-center offset from the live bounding-box center is
                # invariant to that orientation, unlike a local-z "top".
                grasp_position[2] = bottle.pose.position[2] + self.config.grasp_center_offset
            pregrasp = Pose(
                grasp_position + np.array([0.0, 0.0, self.config.approach_height + height_delta]), orientation
            )
            grasp = Pose(grasp_position, orientation)
            lift = Pose(grasp_position + np.array([0.0, 0.0, self.config.lift_height + height_delta]), orientation)

            if self.config.use_overhead_approach:
                # First rise at the current XY, then translate above the
                # bottle. A direct diagonal move from the X5 home pose sweeps
                # its long open fingers through tall bottles.
                overhead = Pose(
                    np.array([poses[pick_arm].position[0], poses[pick_arm].position[1], pregrasp.position[2]]),
                    orientation,
                )
                events.append(self._event(Phase.APPROACH, poses, grippers, pick_arm, overhead, label))
            events.extend(
                [
                    self._event(Phase.APPROACH, poses, grippers, pick_arm, pregrasp, label),
                    self._event(Phase.GRASP, poses, grippers, pick_arm, grasp, label),
                ]
            )
            grippers[pick_arm] = self.config.closed_value
            events.append(self._event(Phase.CLOSE, poses, grippers, pick_arm, grasp, label, self.config.gripper_hold_steps))
            events.append(
                self._event(
                    Phase.LIFT,
                    poses,
                    grippers,
                    pick_arm,
                    lift,
                    label,
                    self.config.lift_hold_steps,
                )
            )
            if self.config.stop_after_lift:
                return events

            carrying_arm = pick_arm
            if pick_arm == "right":
                right_handover = Pose(
                    self.config.handover_right_position, self._orientation("right", snapshot, handover=True)
                )
                left_handover = Pose(
                    self.config.handover_left_position, self._orientation("left", snapshot, handover=True)
                )
                events.append(self._event(Phase.HANDOVER, poses, grippers, "right", right_handover, label))
                events.append(self._event(Phase.HANDOVER, poses, grippers, "left", left_handover, label))
                grippers["left"] = self.config.closed_value
                events.append(
                    self._event(
                        Phase.HANDOVER,
                        poses,
                        grippers,
                        "left",
                        left_handover,
                        label,
                        self.config.gripper_hold_steps,
                    )
                )
                grippers["right"] = self.config.open_value
                events.append(
                    self._event(
                        Phase.RELEASE,
                        poses,
                        grippers,
                        "right",
                        right_handover,
                        label,
                        self.config.gripper_hold_steps,
                    )
                )
                right_retreat = right_handover.at(
                    right_handover.position + np.array([0.12, -0.05, self.config.retreat_height])
                )
                events.append(self._event(Phase.RETREAT, poses, grippers, "right", right_retreat, label))
                left_lift = left_handover.at(left_handover.position + np.array([0.0, 0.0, self.config.lift_height]))
                events.append(self._event(Phase.LIFT, poses, grippers, "left", left_lift, label))
                carrying_arm = "left"

            carry_pose = Pose(drop_position, self._orientation(carrying_arm, snapshot))
            events.append(self._event(Phase.TRANSIT, poses, grippers, carrying_arm, carry_pose, label))
            grippers[carrying_arm] = self.config.open_value
            events.append(
                self._event(
                    Phase.RELEASE,
                    poses,
                    grippers,
                    carrying_arm,
                    carry_pose,
                    label,
                    self.config.gripper_hold_steps,
                )
            )
            retreat = carry_pose.at(carry_pose.position + np.array([0.0, 0.0, self.config.retreat_height]))
            events.append(self._event(Phase.RETREAT, poses, grippers, carrying_arm, retreat, label))

        # Full reward requires both grippers open and both arms back at their
        # exact episode-start poses.
        for arm in ("left", "right"):
            grippers[arm] = self.config.open_value
            events.append(self._event(Phase.HOME, poses, grippers, arm, self._home[arm], None))
        return events

    @staticmethod
    def _event(
        phase: Phase,
        poses: dict[str, Pose],
        grippers: dict[str, float],
        arm: str,
        target: Pose,
        bottle: str | None,
        repeat: int = 1,
    ) -> _Event:
        poses[arm] = target
        return _Event(phase, dict(poses), dict(grippers), bottle, arm, repeat)

    def _compile(self, events: list[_Event], snapshot: SceneSnapshot) -> list[PlannedAction]:
        current = dict(snapshot.arms)
        current_grippers = dict(snapshot.grippers)
        result: list[PlannedAction] = []
        for event in events:
            arm = event.active_arm
            if arm is None:
                continue
            target = event.poses[arm]
            self._check_pose(target, event.phase)
            waypoints = interpolate_pose(current[arm], target, self.config.max_translation_step)
            for waypoint in waypoints:
                action_poses = dict(current)
                action_poses[arm] = waypoint
                result.append(
                    self._planned_action(action_poses, current_grippers, event.phase, event.bottle, arm)
                )
                current[arm] = waypoint
            current_grippers = dict(event.grippers)
            for _ in range(event.repeat):
                result.append(self._planned_action(current, current_grippers, event.phase, event.bottle, arm))
        return result

    def _check_pose(self, pose: Pose, phase: Phase) -> None:
        low = np.asarray(self.config.workspace_min)
        high = np.asarray(self.config.workspace_max)
        if np.any(pose.position < low) or np.any(pose.position > high):
            raise SafetyError(f"{phase.value} target {pose.position.tolist()} is outside workspace {low}..{high}")

    @staticmethod
    def _planned_action(
        poses: Mapping[str, Pose],
        grippers: Mapping[str, float],
        phase: Phase,
        bottle: str | None,
        active_arm: str,
    ) -> PlannedAction:
        action: Action = {
            "left_ee_pose": poses["left"].as_array(),
            "right_ee_pose": poses["right"].as_array(),
            "left_ee_joint_state": np.asarray([grippers["left"]], dtype=np.float32),
            "right_ee_joint_state": np.asarray([grippers["right"]], dtype=np.float32),
        }
        return PlannedAction(
            action=action,
            phase=phase,
            bottle=bottle,
            active_arm=active_arm,
            privileged={"phase": phase.value, "bottle": bottle or "", "active_arm": active_arm},
        )
