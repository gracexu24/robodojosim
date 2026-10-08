from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import (
    align_tool_yaw_to_bbox_major_axis,
    bbox_bottom,
    bbox_top,
    bbox_world_center,
    bbox_world_corners,
    interpolate_pose,
)
from .types import Action, Pose, SceneSnapshot


class Phase(str, Enum):
    APPROACH = "approach"
    GRASP = "grasp"
    CLOSE = "close"
    LIFT = "lift"
    HOLD = "hold"
    CARRY = "carry"
    PUSH = "push"
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
    max_grasp_translation_step: float | None = None
    max_lift_translation_step: float = 0.010
    max_carry_translation_step: float = 0.015
    max_home_translation_step: float | None = None
    approach_height: float = 0.14
    lift_height: float = 0.20
    grasp_clearance: float = 0.015
    grasp_center_offset: float | None = None
    grasp_max_center_offset: float | None = None
    upright_grasp_height_fraction: float | None = None
    drop_clearance: float = 0.18
    place_clearance: float | None = None
    rim_clearance: float = 0.03
    retreat_height: float = 0.14
    bottle_fallback_half_height: float = 0.045
    dustbin_fallback_half_height: float = 0.25
    direct_left_max_x: float = 0.08
    direct_right_drop: bool = False
    open_value: float = 1.0
    closed_value: float = 0.0
    gripper_hold_steps: int = 3
    pregrasp_settle_steps: int = 0
    grasp_settle_steps: int = 0
    lift_hold_steps: int = 1
    drop_hold_steps: int = 0
    home_hold_steps: int = 1
    home_clearance_height: float = 0.15
    home_completion_height_offset: float = 0.0
    intermediate_home_completion_height_offset: float | None = None
    home_between_bottles: bool = False
    max_actions: int = 700
    bottle_limit: int | None = None
    bottle_labels: tuple[str, ...] | None = None
    stop_after_lift: bool = False
    use_overhead_approach: bool = True
    left_grasp_position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_grasp_position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    horizontal_bottle_half_height_max: float = 0.06
    left_horizontal_grasp_position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_horizontal_grasp_position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    align_grasp_to_bbox_major_axis: bool = False
    left_drop_position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_drop_position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    drop_slot_offsets: tuple[tuple[float, float], ...] = ()
    workspace_min: tuple[float, float, float] = (-0.85, -0.55, 0.30)
    workspace_max: tuple[float, float, float] = (0.75, 0.40, 1.35)
    # These are tool-center positions, not bottle positions. They are expected
    # to need a one-time calibration for the exact X5 gripper asset.
    handover_left_position: tuple[float, float, float] = (-0.025, -0.02, 1.02)
    handover_right_position: tuple[float, float, float] = (0.025, -0.02, 1.02)
    staged_handoff: bool = True
    handover_staging_bottle_position: tuple[float, float, float] = (0.025, -0.02, 0.867)
    left_grasp_quaternion: tuple[float, float, float, float] | None = None
    right_grasp_quaternion: tuple[float, float, float, float] | None = None
    left_handover_quaternion: tuple[float, float, float, float] | None = None
    right_handover_quaternion: tuple[float, float, float, float] | None = None
    vary_bottle_order: bool = False
    height_jitter: float = 0.0
    drop_xy_jitter: float = 0.0
    world_model_movements: bool = False
    push_probability: float = 0.0
    push_distance: float = 0.10
    push_center_offset: float = 0.115
    hold_steps_min: int = 1
    hold_steps_max: int = 1
    carry_waypoints_min: int = 0
    carry_waypoints_max: int = 0
    carry_xy_jitter: float = 0.0
    carry_z_jitter: float = 0.0
    regrasp_probability: float = 0.0
    live_pose_feedback: bool = False
    max_grasp_retries: int = 3
    grasp_follow_tolerance: float = 0.045
    retry_carry_translation_step: float = 0.010
    drop_validation_half_extents: tuple[float, float] = (0.29, 0.31)
    drop_validation_margin: float = 0.005
    drop_validation_height_slack: float = 0.015
    target_position_tolerance: float = 0.006
    target_orientation_tolerance_degrees: float = 5.0
    recovery_retreat_height: float = 0.12
    grasp_xy_tolerance: float = 0.012
    grasp_vertical_residual_max: float = 0.055
    grasp_orientation_tolerance_degrees: float = 12.0
    lift_retract_y: float = -0.10
    transit_corridor_y: float = -0.18
    transit_entry_margin: float = 0.065
    home_acceptance_position_tolerance: float = 0.14
    home_acceptance_orientation_degrees: float = 20.0

    def __post_init__(self) -> None:
        if self.max_translation_step <= 0:
            raise ValueError("max_translation_step must be positive")
        if self.max_grasp_translation_step is not None and self.max_grasp_translation_step <= 0:
            raise ValueError("max_grasp_translation_step must be positive when provided")
        if self.max_lift_translation_step <= 0 or self.max_carry_translation_step <= 0:
            raise ValueError("loaded translation steps must be positive")
        if self.max_home_translation_step is not None and self.max_home_translation_step <= 0:
            raise ValueError("max_home_translation_step must be positive when provided")
        if self.gripper_hold_steps < 1 or self.lift_hold_steps < 1 or self.max_actions < 1:
            raise ValueError("gripper_hold_steps, lift_hold_steps, and max_actions must be positive")
        if self.pregrasp_settle_steps < 0 or self.grasp_settle_steps < 0:
            raise ValueError("pregrasp_settle_steps and grasp_settle_steps cannot be negative")
        if self.rim_clearance < 0:
            raise ValueError("rim_clearance cannot be negative")
        if self.place_clearance is not None and self.place_clearance < 0:
            raise ValueError("place_clearance cannot be negative when provided")
        if self.drop_hold_steps < 0 or self.home_hold_steps < 0:
            raise ValueError("drop_hold_steps and home_hold_steps cannot be negative")
        if self.home_clearance_height <= 0:
            raise ValueError("home_clearance_height must be positive")
        if not 0 <= self.home_completion_height_offset < 0.15:
            raise ValueError("home_completion_height_offset must stay within RoboDojo's 0.15 m origin tolerance")
        if self.intermediate_home_completion_height_offset is not None and not (
            0 <= self.intermediate_home_completion_height_offset < 0.15
        ):
            raise ValueError(
                "intermediate_home_completion_height_offset must stay within RoboDojo's 0.15 m origin tolerance"
            )
        if self.bottle_limit is not None and self.bottle_limit < 1:
            raise ValueError("bottle_limit must be positive when provided")
        if self.grasp_max_center_offset is not None and self.grasp_max_center_offset <= 0:
            raise ValueError("grasp_max_center_offset must be positive when provided")
        if self.upright_grasp_height_fraction is not None and not 0 < self.upright_grasp_height_fraction <= 1:
            raise ValueError("upright_grasp_height_fraction must be between zero and one")
        if self.horizontal_bottle_half_height_max <= 0:
            raise ValueError("horizontal_bottle_half_height_max must be positive")
        if self.bottle_labels is not None and (
            not self.bottle_labels or len(set(self.bottle_labels)) != len(self.bottle_labels)
        ):
            raise ValueError("bottle_labels must be non-empty and unique when provided")
        if any(len(offset) != 2 for offset in self.drop_slot_offsets):
            raise ValueError("drop_slot_offsets must contain XY pairs")
        if np.any(np.asarray(self.workspace_min) >= np.asarray(self.workspace_max)):
            raise ValueError("workspace_min must be below workspace_max")
        if self.height_jitter < 0 or self.drop_xy_jitter < 0:
            raise ValueError("trajectory jitter values cannot be negative")
        if not 0.0 <= self.push_probability <= 1.0 or not 0.0 <= self.regrasp_probability <= 1.0:
            raise ValueError("movement probabilities must be between zero and one")
        if self.push_distance < 0 or self.carry_xy_jitter < 0 or self.carry_z_jitter < 0:
            raise ValueError("movement distances cannot be negative")
        if not 1 <= self.hold_steps_min <= self.hold_steps_max:
            raise ValueError("hold step range must be positive and ordered")
        if not 0 <= self.carry_waypoints_min <= self.carry_waypoints_max:
            raise ValueError("carry waypoint range must be non-negative and ordered")
        if self.max_grasp_retries < 0:
            raise ValueError("max_grasp_retries cannot be negative")
        if self.grasp_follow_tolerance <= 0 or self.retry_carry_translation_step <= 0:
            raise ValueError("feedback motion tolerances must be positive")
        if len(self.drop_validation_half_extents) != 2 or any(
            extent <= 0 for extent in self.drop_validation_half_extents
        ):
            raise ValueError("drop_validation_half_extents must contain two positive values")
        if self.drop_validation_margin < 0 or self.drop_validation_height_slack < 0:
            raise ValueError("drop validation margins cannot be negative")
        if self.transit_entry_margin < self.drop_validation_margin or any(
            self.transit_entry_margin >= extent for extent in self.drop_validation_half_extents
        ):
            raise ValueError("transit_entry_margin must fit inside the bin validation extents")
        if not 0 < self.home_acceptance_position_tolerance < 0.15:
            raise ValueError("home_acceptance_position_tolerance must stay below the task's 0.15 m tolerance")
        if not 0 < self.home_acceptance_orientation_degrees < 180:
            raise ValueError("home_acceptance_orientation_degrees must be between zero and 180")
        if self.target_position_tolerance <= 0 or not 0 < self.target_orientation_tolerance_degrees < 180:
            raise ValueError("closed-loop target tolerances must be positive")
        if self.recovery_retreat_height <= 0:
            raise ValueError("recovery_retreat_height must be positive")
        if self.grasp_xy_tolerance <= 0 or self.grasp_vertical_residual_max <= 0:
            raise ValueError("grasp acceptance distances must be positive")
        if not 0 < self.grasp_orientation_tolerance_degrees < 180:
            raise ValueError("grasp_orientation_tolerance_degrees must be between zero and 180")

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
        self._executed_total = 0
        self._feedback_complete = False
        self._feedback_queue: list[str] = []
        self._feedback_slot: dict[str, int] = {}
        self._active_bottle: str | None = None
        self._active_arm: str | None = None
        self._feedback_stage = "idle"
        self._feedback_target: Pose | None = None
        self._feedback_repeat = 0
        self._recovering = False
        self._command_grippers = {"left": self.config.open_value, "right": self.config.open_value}
        self._carry_orientation: np.ndarray | None = None
        self._grasp_to_bottom = 0.0
        self._lift_retracted = False
        self._grasp_relative_position: np.ndarray | None = None
        self._place_xy: np.ndarray | None = None
        self._retry_counts: dict[str, int] = {}
        self._debug_stage = ""
        self._tracking_stage = ""
        self._best_tracking_error = float("inf")
        self._stall_count = 0
        self._skill_plans: dict[str, dict[str, Any]] = {}
        self._active_skill = "pick_place"
        self._world_skills_complete = False
        self._carry_anchor: Pose | None = None
        self._carry_waypoint_index = 0
        self._regrasp_target: Pose | None = None
        self._regrasp_completed = False
        self._push_start: Pose | None = None
        self._push_end: Pose | None = None

    @property
    def done(self) -> bool:
        if self.config.live_pose_feedback:
            return self._feedback_complete
        return bool(self._steps) and self._cursor >= len(self._steps)

    @property
    def phase(self) -> Phase:
        if self.done:
            return Phase.DONE
        if self.config.live_pose_feedback:
            return self._feedback_phase()
        if not self._steps:
            return Phase.DONE
        return self._steps[self._cursor].phase

    @property
    def planned_action_count(self) -> int:
        return len(self._steps)

    @property
    def executed_action_count(self) -> int:
        if self.config.live_pose_feedback:
            return self._executed_total
        return self._cursor

    @property
    def retry_counts(self) -> Mapping[str, int]:
        return dict(self._retry_counts)

    def reset(self, snapshot: SceneSnapshot) -> None:
        self._home = dict(snapshot.arms)
        self._executed_total = 0
        self._feedback_complete = False
        self._active_bottle = None
        self._active_arm = None
        self._feedback_stage = "idle"
        self._feedback_target = None
        self._feedback_repeat = 0
        self._recovering = False
        self._command_grippers = dict(snapshot.grippers)
        self._carry_orientation = None
        self._grasp_to_bottom = 0.0
        self._lift_retracted = False
        self._grasp_relative_position = None
        self._place_xy = None
        self._retry_counts = {}
        self._debug_stage = ""
        self._tracking_stage = ""
        self._best_tracking_error = float("inf")
        self._stall_count = 0
        self._skill_plans = {}
        self._active_skill = "pick_place"
        self._world_skills_complete = False
        self._carry_anchor = None
        self._carry_waypoint_index = 0
        self._regrasp_target = None
        self._regrasp_completed = False
        self._push_start = None
        self._push_end = None
        if self.config.live_pose_feedback:
            self._feedback_queue = self._bottle_order(snapshot)
            self._feedback_slot = {label: index for index, label in enumerate(self._feedback_queue)}
            self._skill_plans = self._build_feedback_skill_plans(snapshot, self._feedback_queue)
            self._feedback_stage = "prepare"
            return
        events = self._build_events(snapshot)
        self._steps = self._compile(events, snapshot)
        self._cursor = 0
        if len(self._steps) > self.config.max_actions:
            raise SafetyError(
                f"plan contains {len(self._steps)} actions, exceeding max_actions={self.config.max_actions}"
            )

    def next_action(self, snapshot: SceneSnapshot | None = None) -> PlannedAction:
        if self.config.live_pose_feedback:
            if snapshot is None:
                raise RuntimeError("live_pose_feedback requires a fresh scene snapshot for every action")
            if not self._home:
                self.reset(snapshot)
            return self._next_feedback_action(snapshot)
        if not self._steps:
            if snapshot is None:
                raise RuntimeError("controller must be reset with a scene snapshot")
            self.reset(snapshot)
        if self.done:
            raise StopIteration("scripted controller is done")
        result = self._steps[self._cursor]
        self._cursor += 1
        return result

    def _next_feedback_action(self, snapshot: SceneSnapshot) -> PlannedAction:
        if self.done:
            raise StopIteration("scripted controller is done")
        if self._feedback_stage in {
            "lift",
            "hold",
            "carry",
            "regrasp_return",
            "regrasp_lower",
            "transit_retract",
            "transit_lateral",
            "transit",
            "place",
        } and self._carried_bottle_was_lost(snapshot):
            if self._bottle_is_inside_bin_xy(snapshot, self._active_bottle):
                print(f"[bottle_controller] bottle={self._active_bottle} released over bin; settling before validation")
                self._grasp_relative_position = None
                self._feedback_target = None
                self._feedback_repeat = 0
                self._feedback_stage = "release"
            else:
                self._begin_recovery(snapshot, "bottle stopped following the closed gripper")

        # A state may already be satisfied by the latest simulator result. In
        # that case advance immediately instead of emitting redundant holds.
        for _ in range(24):
            stage = self._feedback_stage
            if stage != self._tracking_stage:
                self._tracking_stage = stage
                self._best_tracking_error = float("inf")
                self._stall_count = 0
            if os.environ.get("ROBODOJOSIM_CONTROLLER_DEBUG") == "1" and stage != self._debug_stage:
                print(
                    f"[bottle_controller] step={self._executed_total} state={stage} "
                    f"bottle={self._active_bottle or ''} arm={self._active_arm or ''}"
                )
                self._debug_stage = stage
            if stage == "prepare":
                if not self._feedback_queue:
                    self._feedback_complete = True
                    raise StopIteration("scripted controller is done")
                self._active_bottle = self._feedback_queue[0]
                bottle = snapshot.bottles[self._active_bottle]
                self._active_arm = "left" if bottle.pose.position[0] <= self.config.direct_left_max_x else "right"
                self._feedback_target = None
                self._grasp_relative_position = None
                self._carry_orientation = None
                self._lift_retracted = False
                self._place_xy = None
                self._feedback_repeat = 0
                self._world_skills_complete = False
                self._carry_anchor = None
                self._carry_waypoint_index = 0
                self._regrasp_target = None
                self._regrasp_completed = False
                plan = self._skill_plans[self._active_bottle]
                self._active_skill = "push" if plan["push"] else "pick_place"
                self._push_start = None
                self._push_end = None
                if self._active_skill == "push":
                    center = bbox_world_center(bottle.pose, bottle.bbox)
                    direction = float(plan["push_direction"])
                    start = center.copy()
                    start[1] -= direction * self.config.push_distance * 0.5
                    start[2] += self.config.push_center_offset
                    end = start.copy()
                    end[1] += direction * self.config.push_distance
                    low = np.asarray(self.config.workspace_min) + 0.02
                    high = np.asarray(self.config.workspace_max) - 0.02
                    self._push_start = Pose(np.clip(start, low, high), self._orientation(self._active_arm, snapshot))
                    self._push_end = Pose(np.clip(end, low, high), self._push_start.quaternion)
                if abs(snapshot.grippers[self._active_arm] - self.config.open_value) > 0.1:
                    self._recovering = False
                    self._feedback_stage = "open"
                else:
                    self._command_grippers[self._active_arm] = self.config.open_value
                    self._feedback_stage = self._initial_feedback_stage()
                continue

            if stage == "open":
                assert self._active_arm is not None
                self._command_grippers[self._active_arm] = self.config.open_value
                if self._feedback_repeat >= self.config.gripper_hold_steps:
                    self._feedback_repeat = 0
                    self._feedback_target = None
                    self._feedback_stage = "recovery_retreat" if self._recovering else self._initial_feedback_stage()
                    continue
                self._feedback_repeat += 1
                return self._feedback_hold_action(snapshot, Phase.RELEASE)

            if stage == "recovery_retreat":
                assert self._active_arm is not None
                if self._feedback_target is None:
                    position = snapshot.arms[self._active_arm].position + np.array(
                        [0.0, 0.0, self.config.recovery_retreat_height]
                    )
                    position = np.minimum(position, np.asarray(self.config.workspace_max) - 0.02)
                    self._feedback_target = snapshot.arms[self._active_arm].at(position)
                if self._target_reached(snapshot, self._feedback_target) or self._movement_stalled(
                    snapshot, self._feedback_target
                ):
                    self._feedback_target = None
                    self._feedback_stage = "rise"
                    continue
                return self._feedback_move_action(
                    snapshot, self._feedback_target, Phase.RETREAT, self.config.max_translation_step
                )

            if stage == "push_rise":
                assert self._active_arm is not None and self._push_start is not None
                if self._feedback_target is None:
                    current = snapshot.arms[self._active_arm]
                    height = min(
                        self.config.workspace_max[2] - 0.02,
                        self._push_start.position[2] + self.config.approach_height,
                    )
                    self._feedback_target = Pose(
                        [current.position[0], current.position[1], height], self._push_start.quaternion
                    )
                if self._target_reached(snapshot, self._feedback_target) or self._movement_stalled(
                    snapshot, self._feedback_target
                ):
                    self._feedback_target = None
                    self._feedback_stage = "push_precontact"
                    continue
                return self._feedback_move_action(
                    snapshot, self._feedback_target, Phase.APPROACH, self.config.max_translation_step
                )

            if stage == "push_precontact":
                assert self._push_start is not None
                target = self._push_start.at(self._push_start.position + [0.0, 0.0, self.config.approach_height])
                if self._target_reached(snapshot, target) or self._movement_stalled(snapshot, target):
                    self._feedback_repeat = 0
                    self._feedback_stage = "push_close"
                    continue
                return self._feedback_move_action(snapshot, target, Phase.APPROACH, self.config.max_translation_step)

            if stage == "push_close":
                assert self._active_arm is not None
                self._command_grippers[self._active_arm] = self.config.closed_value
                if self._feedback_repeat >= self.config.gripper_hold_steps:
                    self._feedback_repeat = 0
                    self._feedback_stage = "push_contact"
                    continue
                self._feedback_repeat += 1
                return self._feedback_hold_action(snapshot, Phase.CLOSE)

            if stage == "push_contact":
                assert self._push_start is not None
                if self._target_reached(snapshot, self._push_start) or self._movement_stalled(
                    snapshot, self._push_start
                ):
                    self._feedback_stage = "push_execute"
                    continue
                return self._feedback_move_action(
                    snapshot, self._push_start, Phase.PUSH, self.config.max_carry_translation_step
                )

            if stage == "push_execute":
                assert self._push_end is not None
                if self._target_reached(snapshot, self._push_end) or self._movement_stalled(snapshot, self._push_end):
                    self._feedback_target = self._push_end.at(
                        self._push_end.position + [0.0, 0.0, self.config.retreat_height]
                    )
                    self._feedback_stage = "push_retreat"
                    continue
                return self._feedback_move_action(
                    snapshot, self._push_end, Phase.PUSH, self.config.max_carry_translation_step
                )

            if stage == "push_retreat":
                assert self._feedback_target is not None
                if self._target_reached(snapshot, self._feedback_target) or self._movement_stalled(
                    snapshot, self._feedback_target
                ):
                    self._feedback_target = None
                    self._feedback_repeat = 0
                    self._feedback_stage = "push_open"
                    continue
                return self._feedback_move_action(
                    snapshot, self._feedback_target, Phase.RETREAT, self.config.max_translation_step
                )

            if stage == "push_open":
                assert self._active_arm is not None
                self._command_grippers[self._active_arm] = self.config.open_value
                if self._feedback_repeat >= self.config.gripper_hold_steps:
                    self._feedback_repeat = 0
                    self._feedback_stage = "home_raise"
                    continue
                self._feedback_repeat += 1
                return self._feedback_hold_action(snapshot, Phase.RELEASE)

            if stage == "rise":
                assert self._active_arm is not None
                _, pregrasp, _ = self._live_grasp_poses(snapshot)
                if self._feedback_target is None:
                    current = snapshot.arms[self._active_arm]
                    self._feedback_target = Pose(
                        [
                            current.position[0],
                            current.position[1],
                            pregrasp.position[2],
                        ],
                        current.quaternion,
                    )
                if self._target_reached(snapshot, self._feedback_target) or self._movement_stalled(
                    snapshot, self._feedback_target
                ):
                    self._feedback_target = None
                    self._feedback_stage = "pregrasp"
                    continue
                return self._feedback_move_action(
                    snapshot, self._feedback_target, Phase.APPROACH, self.config.max_translation_step
                )

            if stage == "pregrasp":
                _, target, _ = self._live_grasp_poses(snapshot)
                if self._target_reached(snapshot, target) or self._movement_stalled(snapshot, target):
                    self._feedback_stage = "grasp"
                    continue
                return self._feedback_move_action(snapshot, target, Phase.APPROACH, self.config.max_translation_step)

            if stage == "grasp":
                target, _, bottle_bottom = self._live_grasp_poses(snapshot)
                stalled = self._movement_stalled(snapshot, target)
                if self._target_reached(snapshot, target) or (
                    stalled and self._grasp_pose_is_acceptable(snapshot, target)
                ):
                    self._carry_orientation = target.quaternion.copy()
                    self._grasp_to_bottom = max(0.0, float(target.position[2] - bottle_bottom))
                    if self._regrasp_target is None:
                        self._regrasp_target = target
                    self._feedback_repeat = 0
                    self._feedback_stage = "close"
                    continue
                step = self.config.max_grasp_translation_step or self.config.max_translation_step
                return self._feedback_move_action(snapshot, target, Phase.GRASP, step)

            if stage == "close":
                assert self._active_arm is not None and self._active_bottle is not None
                self._command_grippers[self._active_arm] = self.config.closed_value
                if self._feedback_repeat >= self.config.gripper_hold_steps:
                    bottle = snapshot.bottles[self._active_bottle]
                    self._grasp_relative_position = (
                        bbox_world_center(bottle.pose, bottle.bbox) - snapshot.arms[self._active_arm].position
                    )
                    self._feedback_repeat = 0
                    self._feedback_target = self._lift_target(snapshot)
                    self._feedback_stage = "lift"
                    continue
                self._feedback_repeat += 1
                return self._feedback_hold_action(snapshot, Phase.CLOSE)

            if stage == "lift":
                assert self._feedback_target is not None
                reached = self._target_reached(snapshot, self._feedback_target)
                stalled = self._movement_stalled(snapshot, self._feedback_target)
                if reached or (stalled and self._bottle_clears_rim(snapshot)):
                    self._feedback_target = None
                    if self.config.stop_after_lift:
                        self._feedback_complete = True
                        raise StopIteration("scripted controller completed requested lift")
                    if self.config.world_model_movements and not self._world_skills_complete:
                        self._carry_anchor = snapshot.arms[self._active_arm]
                        self._feedback_repeat = 0
                        self._feedback_stage = "hold"
                    else:
                        self._active_skill = "pick_place"
                        self._feedback_stage = "transit_retract"
                    continue
                if stalled and not self._lift_retracted:
                    current = snapshot.arms[self._active_arm]
                    position = self._feedback_target.position.copy()
                    position[0] = current.position[0]
                    position[1] = min(current.position[1], self.config.lift_retract_y)
                    self._feedback_target = Pose(position, self._feedback_target.quaternion)
                    self._lift_retracted = True
                    self._best_tracking_error = float("inf")
                    self._stall_count = 0
                    if os.environ.get("ROBODOJOSIM_CONTROLLER_DEBUG") == "1":
                        print(
                            f"[bottle_controller] step={self._executed_total} state=lift "
                            f"recompute=retract target={position.round(5).tolist()}"
                        )
                    continue
                if stalled:
                    self._begin_recovery(snapshot, "lift stalled before the bottle cleared the bin rim")
                    continue
                return self._feedback_move_action(
                    snapshot, self._feedback_target, Phase.LIFT, self.config.max_lift_translation_step
                )

            if stage == "hold":
                assert self._active_bottle is not None
                hold_steps = int(self._skill_plans[self._active_bottle]["hold_steps"])
                if self._feedback_repeat >= hold_steps:
                    self._feedback_repeat = 0
                    self._carry_waypoint_index = 0
                    self._feedback_stage = "carry"
                    continue
                self._feedback_repeat += 1
                return self._feedback_hold_action(snapshot, Phase.HOLD)

            if stage == "carry":
                assert self._active_bottle is not None and self._active_arm is not None
                assert self._carry_anchor is not None and self._carry_orientation is not None
                offsets = self._skill_plans[self._active_bottle]["carry_offsets"]
                if self._carry_waypoint_index >= len(offsets):
                    self._feedback_target = None
                    if self._skill_plans[self._active_bottle]["regrasp"] and not self._regrasp_completed:
                        self._active_skill = "regrasp"
                        self._feedback_stage = "regrasp_return"
                    else:
                        self._world_skills_complete = True
                        self._active_skill = "pick_place"
                        self._feedback_stage = "transit_retract"
                    continue
                if self._feedback_target is None:
                    offset = np.asarray(offsets[self._carry_waypoint_index], dtype=np.float64)
                    position = np.clip(
                        self._carry_anchor.position + offset,
                        np.asarray(self.config.workspace_min) + 0.02,
                        np.asarray(self.config.workspace_max) - 0.02,
                    )
                    self._feedback_target = Pose(position, self._carry_orientation)
                if self._target_reached(snapshot, self._feedback_target) or self._movement_stalled(
                    snapshot, self._feedback_target
                ):
                    self._carry_waypoint_index += 1
                    self._feedback_target = None
                    continue
                return self._feedback_move_action(snapshot, self._feedback_target, Phase.CARRY, self._carry_step())

            if stage == "regrasp_return":
                assert self._regrasp_target is not None and self._active_arm is not None
                if self._feedback_target is None:
                    current = snapshot.arms[self._active_arm]
                    self._feedback_target = Pose(
                        [self._regrasp_target.position[0], self._regrasp_target.position[1], current.position[2]],
                        self._carry_orientation,
                    )
                if self._target_reached(snapshot, self._feedback_target) or self._movement_stalled(
                    snapshot, self._feedback_target
                ):
                    self._feedback_target = None
                    self._feedback_stage = "regrasp_lower"
                    continue
                return self._feedback_move_action(snapshot, self._feedback_target, Phase.CARRY, self._carry_step())

            if stage == "regrasp_lower":
                assert self._regrasp_target is not None
                if self._target_reached(snapshot, self._regrasp_target) or self._movement_stalled(
                    snapshot, self._regrasp_target
                ):
                    self._feedback_repeat = 0
                    self._feedback_stage = "regrasp_release"
                    continue
                return self._feedback_move_action(
                    snapshot, self._regrasp_target, Phase.CARRY, self.config.max_lift_translation_step
                )

            if stage == "regrasp_release":
                assert self._active_arm is not None
                self._command_grippers[self._active_arm] = self.config.open_value
                if self._feedback_repeat >= self.config.gripper_hold_steps:
                    self._feedback_repeat = 0
                    self._grasp_relative_position = None
                    current = snapshot.arms[self._active_arm]
                    position = np.minimum(
                        current.position + [0.0, 0.0, self.config.recovery_retreat_height],
                        np.asarray(self.config.workspace_max) - 0.02,
                    )
                    self._feedback_target = current.at(position)
                    self._feedback_stage = "regrasp_retreat"
                    continue
                self._feedback_repeat += 1
                return self._feedback_hold_action(snapshot, Phase.RELEASE)

            if stage == "regrasp_retreat":
                assert self._feedback_target is not None
                if self._target_reached(snapshot, self._feedback_target) or self._movement_stalled(
                    snapshot, self._feedback_target
                ):
                    self._feedback_target = None
                    self._regrasp_completed = True
                    self._world_skills_complete = True
                    self._active_skill = "regrasp"
                    self._feedback_stage = "rise"
                    continue
                return self._feedback_move_action(
                    snapshot, self._feedback_target, Phase.RETREAT, self.config.max_translation_step
                )

            if stage == "transit_retract":
                assert self._active_arm is not None and self._carry_orientation is not None
                if self._feedback_target is None:
                    current = snapshot.arms[self._active_arm]
                    position = current.position.copy()
                    position[1] = self.config.transit_corridor_y
                    self._feedback_target = Pose(position, self._carry_orientation)
                reached = self._target_reached(snapshot, self._feedback_target)
                stalled = self._movement_stalled(snapshot, self._feedback_target)
                if reached or stalled:
                    self._feedback_target = None
                    self._feedback_stage = "transit_lateral"
                    continue
                return self._feedback_move_action(snapshot, self._feedback_target, Phase.TRANSIT, self._carry_step())

            if stage == "transit_lateral":
                assert self._active_arm is not None
                if self._bottle_is_inside_bin_xy(
                    snapshot, self._active_bottle, margin=self.config.transit_entry_margin
                ):
                    self._place_xy = snapshot.arms[self._active_arm].position[:2].copy()
                    self._feedback_target = None
                    self._feedback_stage = "place" if self.config.place_clearance is not None else "release"
                    continue
                if self._feedback_target is None:
                    desired = self._drop_target(snapshot, place=False)
                    position = desired.position.copy()
                    position[1] = self.config.transit_corridor_y
                    self._feedback_target = Pose(position, desired.quaternion)
                reached = self._target_reached(snapshot, self._feedback_target)
                stalled = self._movement_stalled(snapshot, self._feedback_target)
                if reached:
                    self._feedback_target = None
                    self._feedback_stage = "transit"
                    continue
                if stalled:
                    if self._bottle_is_inside_bin_xy(snapshot, self._active_bottle):
                        self._place_xy = snapshot.arms[self._active_arm].position[:2].copy()
                        self._feedback_target = None
                        self._feedback_stage = "place" if self.config.place_clearance is not None else "release"
                        continue
                    self._begin_recovery(snapshot, "lateral transit stalled before the bottle entered the bin")
                    continue
                return self._feedback_move_action(snapshot, self._feedback_target, Phase.TRANSIT, self._carry_step())

            if stage == "transit":
                assert self._active_arm is not None
                if self._feedback_target is None:
                    self._feedback_target = self._drop_target(snapshot, place=False)
                reached = self._target_reached(snapshot, self._feedback_target)
                stalled = self._movement_stalled(snapshot, self._feedback_target)
                if reached or (stalled and self._bottle_is_inside_bin_xy(snapshot, self._active_bottle)):
                    self._place_xy = snapshot.arms[self._active_arm].position[:2].copy()
                    self._feedback_target = None
                    self._feedback_stage = "place" if self.config.place_clearance is not None else "release"
                    continue
                if stalled:
                    self._begin_recovery(snapshot, "forward transit stalled before the bottle entered the bin")
                    continue
                return self._feedback_move_action(snapshot, self._feedback_target, Phase.TRANSIT, self._carry_step())

            if stage == "place":
                if self._feedback_target is None:
                    self._feedback_target = self._drop_target(snapshot, place=True)
                reached = self._target_reached(snapshot, self._feedback_target)
                stalled = self._movement_stalled(snapshot, self._feedback_target)
                if reached or (stalled and self._bottle_is_inside_bin(snapshot, self._active_bottle)):
                    self._feedback_target = None
                    self._feedback_repeat = 0
                    self._feedback_stage = "drop_hold"
                    continue
                if stalled:
                    self._begin_recovery(snapshot, "placement stalled before the bottle was inside the bin")
                    continue
                return self._feedback_move_action(snapshot, self._feedback_target, Phase.TRANSIT, self._carry_step())

            if stage == "drop_hold":
                if self._feedback_repeat >= self.config.drop_hold_steps:
                    self._feedback_repeat = 0
                    self._feedback_stage = "release"
                    continue
                self._feedback_repeat += 1
                return self._feedback_hold_action(snapshot, Phase.HOLD)

            if stage == "release":
                assert self._active_arm is not None
                self._command_grippers[self._active_arm] = self.config.open_value
                if self._feedback_repeat >= self.config.gripper_hold_steps:
                    self._feedback_repeat = 0
                    position = snapshot.arms[self._active_arm].position + np.array(
                        [0.0, 0.0, self.config.retreat_height]
                    )
                    self._feedback_target = snapshot.arms[self._active_arm].at(position)
                    self._feedback_stage = "retreat"
                    continue
                self._feedback_repeat += 1
                return self._feedback_hold_action(snapshot, Phase.RELEASE)

            if stage == "retreat":
                assert self._feedback_target is not None
                if self._target_reached(snapshot, self._feedback_target):
                    self._feedback_target = None
                    self._feedback_stage = "home_raise"
                    continue
                return self._feedback_move_action(
                    snapshot, self._feedback_target, Phase.RETREAT, self.config.max_translation_step
                )

            if stage == "home_raise":
                assert self._active_arm is not None
                if self._feedback_target is None:
                    current = snapshot.arms[self._active_arm]
                    clearance = min(
                        self.config.workspace_max[2] - 0.02,
                        max(current.position[2], self._home[self._active_arm].position[2])
                        + self.config.home_clearance_height,
                    )
                    self._feedback_target = current.at([current.position[0], current.position[1], clearance])
                if self._target_reached(snapshot, self._feedback_target) or self._movement_stalled(
                    snapshot, self._feedback_target
                ):
                    self._feedback_target = None
                    self._feedback_stage = "home_overhead"
                    continue
                return self._feedback_move_action(snapshot, self._feedback_target, Phase.HOME, self._home_step())

            if stage == "home_overhead":
                assert self._active_arm is not None
                if self._feedback_target is None:
                    current = snapshot.arms[self._active_arm]
                    home = self._home[self._active_arm]
                    self._feedback_target = Pose(
                        [home.position[0], home.position[1], current.position[2]], current.quaternion
                    )
                if self._target_reached(snapshot, self._feedback_target) or self._movement_stalled(
                    snapshot, self._feedback_target
                ):
                    self._feedback_target = None
                    self._feedback_stage = "home_descend"
                    continue
                return self._feedback_move_action(snapshot, self._feedback_target, Phase.HOME, self._home_step())

            if stage == "home_descend":
                assert self._active_arm is not None
                home = self._home[self._active_arm]
                target = home.at(home.position + [0.0, 0.0, self.config.home_completion_height_offset])
                reached = self._target_reached(snapshot, target)
                stalled = self._movement_stalled(snapshot, target)
                if reached or (stalled and self._home_pose_is_acceptable(snapshot)):
                    if self._active_skill != "push" and not self._bottle_is_inside_bin(snapshot, self._active_bottle):
                        self._begin_recovery(snapshot, "released bottle is not fully inside the dustbin")
                        continue
                    self._feedback_queue.pop(0)
                    self._active_bottle = None
                    self._active_arm = None
                    self._feedback_target = None
                    self._grasp_relative_position = None
                    self._feedback_stage = "prepare"
                    continue
                if stalled:
                    raise SafetyError(f"{self._active_arm} arm stalled outside the task's home tolerance")
                return self._feedback_move_action(snapshot, target, Phase.HOME, self._home_step())

            raise RuntimeError(f"unknown feedback stage {stage!r}")

        raise RuntimeError("closed-loop state machine made too many transitions without emitting an action")

    def _begin_recovery(self, snapshot: SceneSnapshot, reason: str) -> None:
        if self._active_bottle is None:
            raise SafetyError(f"feedback recovery has no active bottle: {reason}")
        retry = self._retry_counts.get(self._active_bottle, 0) + 1
        self._retry_counts[self._active_bottle] = retry
        if retry > self.config.max_grasp_retries:
            raise SafetyError(f"{self._active_bottle} failed after {self.config.max_grasp_retries} retries: {reason}")
        print(f"[bottle_controller] retry={retry} bottle={self._active_bottle} reason={reason}")
        self._feedback_target = None
        self._feedback_repeat = 0
        self._recovering = True
        self._grasp_relative_position = None
        self._place_xy = None
        self._feedback_stage = "open"

    def _carried_bottle_was_lost(self, snapshot: SceneSnapshot) -> bool:
        if self._active_bottle is None or self._active_arm is None or self._grasp_relative_position is None:
            return False
        bottle = snapshot.bottles[self._active_bottle]
        current_relative = bbox_world_center(bottle.pose, bottle.bbox) - snapshot.arms[self._active_arm].position
        return (
            float(np.linalg.norm(current_relative - self._grasp_relative_position)) > self.config.grasp_follow_tolerance
        )

    def _live_grasp_poses(self, snapshot: SceneSnapshot) -> tuple[Pose, Pose, float]:
        assert self._active_bottle is not None and self._active_arm is not None
        bottle = snapshot.bottles[self._active_bottle]
        bottle_center = bbox_world_center(bottle.pose, bottle.bbox)
        bottle_top = bbox_top(bottle.pose, bottle.bbox, self.config.bottle_fallback_half_height)
        bottle_bottom = bbox_bottom(bottle.pose, bottle.bbox, self.config.bottle_fallback_half_height)
        grasp_position = bottle_center.copy()
        if self.config.grasp_center_offset is None:
            offset = bottle_top + self.config.grasp_clearance - bottle_center[2]
            if self.config.grasp_max_center_offset is not None:
                offset = min(offset, self.config.grasp_max_center_offset)
            world_half_height = (bottle_top - bottle_bottom) / 2.0
            if (
                self.config.upright_grasp_height_fraction is not None
                and world_half_height > self.config.horizontal_bottle_half_height_max
            ):
                offset = min(offset, world_half_height * self.config.upright_grasp_height_fraction)
            grasp_position[2] = bottle_center[2] + offset
        else:
            grasp_position[2] = bottle_center[2] + self.config.grasp_center_offset
        grasp_position += np.asarray(
            getattr(self.config, f"{self._active_arm}_grasp_position_offset"), dtype=np.float64
        )
        if bottle_top - bottle_center[2] <= self.config.horizontal_bottle_half_height_max:
            grasp_position += np.asarray(
                getattr(self.config, f"{self._active_arm}_horizontal_grasp_position_offset"),
                dtype=np.float64,
            )
        orientation = self._orientation(self._active_arm, snapshot)
        if self.config.align_grasp_to_bbox_major_axis:
            orientation = align_tool_yaw_to_bbox_major_axis(orientation, bottle.pose, bottle.bbox)
        grasp = Pose(grasp_position, orientation)
        pregrasp = grasp.at(grasp.position + [0.0, 0.0, self.config.approach_height])
        return grasp, pregrasp, bottle_bottom

    def _lift_target(self, snapshot: SceneSnapshot) -> Pose:
        assert self._active_arm is not None and self._carry_orientation is not None
        current = snapshot.arms[self._active_arm]
        dustbin_top = bbox_top(snapshot.dustbin.pose, snapshot.dustbin.bbox, self.config.dustbin_fallback_half_height)
        target_z = max(
            current.position[2] + self.config.lift_height,
            dustbin_top + self.config.rim_clearance + self._grasp_to_bottom,
        )
        return Pose([current.position[0], current.position[1], target_z], self._carry_orientation)

    def _drop_target(self, snapshot: SceneSnapshot, *, place: bool) -> Pose:
        assert self._active_bottle is not None and self._active_arm is not None
        assert self._carry_orientation is not None
        slot = self._feedback_slot[self._active_bottle]
        dustbin_top = bbox_top(snapshot.dustbin.pose, snapshot.dustbin.bbox, self.config.dustbin_fallback_half_height)
        dustbin_bottom = bbox_bottom(
            snapshot.dustbin.pose, snapshot.dustbin.bbox, self.config.dustbin_fallback_half_height
        )
        position = snapshot.dustbin.pose.position.copy()
        if self.config.drop_slot_offsets:
            position[:2] += np.asarray(
                self.config.drop_slot_offsets[slot % len(self.config.drop_slot_offsets)], dtype=np.float64
            )
        position += np.asarray(getattr(self.config, f"{self._active_arm}_drop_position_offset"), dtype=np.float64)
        if place:
            if self._place_xy is not None:
                position[:2] = self._place_xy
            position[2] = dustbin_bottom + self._grasp_to_bottom + float(self.config.place_clearance)
        else:
            position[2] = max(
                dustbin_top + self.config.drop_clearance,
                dustbin_top + self.config.rim_clearance + self._grasp_to_bottom,
            )
        return Pose(position, self._carry_orientation)

    def _bottle_clears_rim(self, snapshot: SceneSnapshot) -> bool:
        assert self._active_bottle is not None
        bottle = snapshot.bottles[self._active_bottle]
        bottle_bottom = bbox_bottom(bottle.pose, bottle.bbox, self.config.bottle_fallback_half_height)
        dustbin_top = bbox_top(snapshot.dustbin.pose, snapshot.dustbin.bbox, self.config.dustbin_fallback_half_height)
        return bottle_bottom >= dustbin_top + self.config.rim_clearance

    def _target_reached(self, snapshot: SceneSnapshot, target: Pose) -> bool:
        assert self._active_arm is not None
        current = snapshot.arms[self._active_arm]
        position_ok = np.linalg.norm(current.position - target.position) <= self.config.target_position_tolerance
        cosine = np.cos(np.deg2rad(self.config.target_orientation_tolerance_degrees) / 2.0)
        orientation_ok = abs(float(np.dot(current.quaternion, target.quaternion))) >= cosine
        return bool(position_ok and orientation_ok)

    def _movement_stalled(self, snapshot: SceneSnapshot, target: Pose) -> bool:
        """Detect an IK staging target that has stopped making useful progress.

        This is used only for collision-safe approach states. Task-critical
        grasp, lift, transit, and place states still require their target or
        enter explicit recovery instead of silently advancing.
        """

        assert self._active_arm is not None
        current = snapshot.arms[self._active_arm]
        position_error = float(np.linalg.norm(current.position - target.position))
        angle = np.rad2deg(
            2.0 * np.arccos(np.clip(abs(float(np.dot(current.quaternion, target.quaternion))), 0.0, 1.0))
        )
        combined_error = position_error + 0.002 * float(angle)
        if combined_error < self._best_tracking_error - 0.0005:
            self._best_tracking_error = combined_error
            self._stall_count = 0
        else:
            self._stall_count += 1
        return self._stall_count >= 12

    def _grasp_pose_is_acceptable(self, snapshot: SceneSnapshot, target: Pose) -> bool:
        assert self._active_arm is not None
        current = snapshot.arms[self._active_arm]
        xy_error = float(np.linalg.norm(current.position[:2] - target.position[:2]))
        vertical_residual = float(current.position[2] - target.position[2])
        angle = np.rad2deg(
            2.0 * np.arccos(np.clip(abs(float(np.dot(current.quaternion, target.quaternion))), 0.0, 1.0))
        )
        return bool(
            xy_error <= self.config.grasp_xy_tolerance
            and 0.0 <= vertical_residual <= self.config.grasp_vertical_residual_max
            and angle <= self.config.grasp_orientation_tolerance_degrees
        )

    def _feedback_move_action(
        self, snapshot: SceneSnapshot, target: Pose, phase: Phase, max_step: float
    ) -> PlannedAction:
        assert self._active_arm is not None
        self._check_pose(target, phase)
        current = snapshot.arms[self._active_arm]
        if (
            os.environ.get("ROBODOJOSIM_CONTROLLER_DEBUG") == "1"
            and self._executed_total > 0
            and self._executed_total % 25 == 0
        ):
            orientation_error = np.rad2deg(
                2.0 * np.arccos(np.clip(abs(float(np.dot(current.quaternion, target.quaternion))), 0.0, 1.0))
            )
            print(
                f"[bottle_controller] step={self._executed_total} state={self._feedback_stage} "
                f"position_error={np.linalg.norm(current.position - target.position):.5f} "
                f"orientation_error_deg={orientation_error:.2f} "
                f"current={current.position.round(5).tolist()} target={target.position.round(5).tolist()}"
            )
        waypoint = interpolate_pose(current, target, max_step)[0]
        poses = dict(snapshot.arms)
        poses[self._active_arm] = waypoint
        return self._emit_feedback_action(poses, phase)

    def _feedback_hold_action(self, snapshot: SceneSnapshot, phase: Phase) -> PlannedAction:
        return self._emit_feedback_action(snapshot.arms, phase)

    def _emit_feedback_action(self, poses: Mapping[str, Pose], phase: Phase) -> PlannedAction:
        assert self._active_arm is not None
        base = self._planned_action(poses, self._command_grippers, phase, self._active_bottle, self._active_arm)
        privileged = dict(base.privileged)
        privileged["skill"] = self._feedback_skill()
        result = PlannedAction(
            action=base.action,
            phase=base.phase,
            bottle=base.bottle,
            active_arm=base.active_arm,
            privileged=privileged,
        )
        self._executed_total += 1
        if self._executed_total > self.config.max_actions:
            raise SafetyError(f"feedback controller exceeded max_actions={self.config.max_actions}")
        return result

    def _carry_step(self) -> float:
        if self._active_bottle is not None and self._retry_counts.get(self._active_bottle, 0):
            return self.config.retry_carry_translation_step
        return self.config.max_carry_translation_step

    def _home_step(self) -> float:
        return self.config.max_home_translation_step or self.config.max_translation_step

    def _feedback_phase(self) -> Phase:
        return {
            "prepare": Phase.APPROACH,
            "open": Phase.RELEASE,
            "recovery_retreat": Phase.RETREAT,
            "push_rise": Phase.APPROACH,
            "push_precontact": Phase.APPROACH,
            "push_close": Phase.CLOSE,
            "push_contact": Phase.PUSH,
            "push_execute": Phase.PUSH,
            "push_retreat": Phase.RETREAT,
            "push_open": Phase.RELEASE,
            "rise": Phase.APPROACH,
            "pregrasp": Phase.APPROACH,
            "grasp": Phase.GRASP,
            "close": Phase.CLOSE,
            "lift": Phase.LIFT,
            "hold": Phase.HOLD,
            "carry": Phase.CARRY,
            "regrasp_return": Phase.CARRY,
            "regrasp_lower": Phase.CARRY,
            "regrasp_release": Phase.RELEASE,
            "regrasp_retreat": Phase.RETREAT,
            "transit_retract": Phase.TRANSIT,
            "transit_lateral": Phase.TRANSIT,
            "transit": Phase.TRANSIT,
            "place": Phase.TRANSIT,
            "drop_hold": Phase.HOLD,
            "release": Phase.RELEASE,
            "retreat": Phase.RETREAT,
            "home_raise": Phase.HOME,
            "home_overhead": Phase.HOME,
            "home_descend": Phase.HOME,
        }.get(self._feedback_stage, Phase.DONE)

    def _feedback_skill(self) -> str:
        if self._feedback_stage.startswith("push_") or self._active_skill == "push":
            return "push"
        if self._feedback_stage == "hold":
            return "hold"
        if self._feedback_stage == "carry":
            return "carry"
        if self._feedback_stage.startswith("regrasp_") or self._active_skill == "regrasp":
            return "regrasp"
        return "pick_place"

    def _initial_feedback_stage(self) -> str:
        return "push_rise" if self._active_skill == "push" else "rise"

    def _bottle_is_inside_bin(self, snapshot: SceneSnapshot, label: str | None) -> bool:
        if label is None:
            return False
        bottle = snapshot.bottles[label]
        if bottle.bbox is None:
            corners = bottle.pose.position.reshape(1, 3)
        else:
            corners = bbox_world_corners(bottle.pose, bottle.bbox)
        center = snapshot.dustbin.pose.position
        half_xy = np.asarray(self.config.drop_validation_half_extents) - self.config.drop_validation_margin
        relative_xy = corners[:, :2] - center[:2]
        if np.any(np.abs(relative_xy) > half_xy):
            return False
        bottom = bbox_bottom(snapshot.dustbin.pose, snapshot.dustbin.bbox, self.config.dustbin_fallback_half_height)
        top = bbox_top(snapshot.dustbin.pose, snapshot.dustbin.bbox, self.config.dustbin_fallback_half_height)
        slack = self.config.drop_validation_height_slack
        return bool(np.min(corners[:, 2]) >= bottom - slack and np.max(corners[:, 2]) <= top + slack)

    def _bottle_is_inside_bin_xy(
        self, snapshot: SceneSnapshot, label: str | None, *, margin: float | None = None
    ) -> bool:
        if label is None:
            return False
        bottle = snapshot.bottles[label]
        if bottle.bbox is None:
            corners = bottle.pose.position.reshape(1, 3)
        else:
            corners = bbox_world_corners(bottle.pose, bottle.bbox)
        center = snapshot.dustbin.pose.position
        inset = self.config.drop_validation_margin if margin is None else margin
        half_xy = np.asarray(self.config.drop_validation_half_extents) - inset
        return bool(np.all(np.abs(corners[:, :2] - center[:2]) <= half_xy))

    def _home_pose_is_acceptable(self, snapshot: SceneSnapshot) -> bool:
        assert self._active_arm is not None
        current = snapshot.arms[self._active_arm]
        home = self._home[self._active_arm]
        position_ok = np.linalg.norm(current.position - home.position) <= self.config.home_acceptance_position_tolerance
        cosine = np.cos(np.deg2rad(self.config.home_acceptance_orientation_degrees) / 2.0)
        orientation_ok = abs(float(np.dot(current.quaternion, home.quaternion))) >= cosine
        return bool(position_ok and orientation_ok)

    def _orientation(self, arm: str, snapshot: SceneSnapshot, handover: bool = False) -> np.ndarray:
        name = f"{arm}_{'handover' if handover else 'grasp'}_quaternion"
        configured = getattr(self.config, name)
        return snapshot.arms[arm].quaternion if configured is None else np.asarray(configured, dtype=np.float64)

    def _bottle_order(self, snapshot: SceneSnapshot) -> list[str]:
        layout_words = [
            round((coordinate + 2.0) * 10_000)
            for name in sorted(snapshot.bottles)
            for coordinate in snapshot.bottles[name].pose.position[:2]
        ]
        rng = np.random.default_rng(np.random.SeedSequence([self.trajectory_variant, *layout_words]))
        bottle_order = sorted(snapshot.bottles, key=lambda name: snapshot.bottles[name].pose.position[0])
        if self.config.vary_bottle_order:
            rng.shuffle(bottle_order)
        if self.config.bottle_labels is not None:
            missing = set(self.config.bottle_labels) - set(snapshot.bottles)
            if missing:
                raise SafetyError(f"configured bottles are absent from the scene: {sorted(missing)}")
            bottle_order = list(self.config.bottle_labels)
        if self.config.bottle_limit is not None:
            bottle_order = bottle_order[: self.config.bottle_limit]
        return bottle_order

    def _build_feedback_skill_plans(
        self, snapshot: SceneSnapshot, bottle_order: list[str]
    ) -> dict[str, dict[str, Any]]:
        layout_words = [
            round((coordinate + 2.0) * 10_000)
            for name in sorted(snapshot.bottles)
            for coordinate in snapshot.bottles[name].pose.position[:2]
        ]
        rng = np.random.default_rng(np.random.SeedSequence([self.trajectory_variant, 0xB0771E, *layout_words]))
        plans: dict[str, dict[str, Any]] = {}
        for index, label in enumerate(bottle_order):
            push = bool(
                self.config.world_model_movements and index == 0 and rng.random() < self.config.push_probability
            )
            hold_steps = (
                int(rng.integers(self.config.hold_steps_min, self.config.hold_steps_max + 1))
                if self.config.world_model_movements and not push
                else 0
            )
            waypoint_count = (
                int(rng.integers(self.config.carry_waypoints_min, self.config.carry_waypoints_max + 1))
                if self.config.world_model_movements and not push
                else 0
            )
            offsets = [
                (
                    float(rng.uniform(-self.config.carry_xy_jitter, self.config.carry_xy_jitter)),
                    float(rng.uniform(-self.config.carry_xy_jitter, self.config.carry_xy_jitter)),
                    float(rng.uniform(-self.config.carry_z_jitter, self.config.carry_z_jitter)),
                )
                for _ in range(waypoint_count)
            ]
            if offsets:
                offsets.append((0.0, 0.0, 0.0))
            plans[label] = {
                "push": push,
                "push_direction": 1.0 if rng.random() < 0.5 else -1.0,
                "hold_steps": hold_steps,
                "carry_offsets": tuple(offsets),
                "regrasp": bool(
                    self.config.world_model_movements and not push and rng.random() < self.config.regrasp_probability
                ),
            }
        return plans

    def _build_events(
        self,
        snapshot: SceneSnapshot,
        *,
        bottle_labels: tuple[str, ...] | None = None,
        slot_indices: tuple[int, ...] | None = None,
        append_final_home: bool = True,
    ) -> list[_Event]:
        poses = dict(snapshot.arms)
        grippers = dict(snapshot.grippers)
        events: list[_Event] = []
        layout_words = [
            round((coordinate + 2.0) * 10_000)
            for name in sorted(snapshot.bottles)
            for coordinate in snapshot.bottles[name].pose.position[:2]
        ]
        rng = np.random.default_rng(np.random.SeedSequence([self.trajectory_variant, *layout_words]))
        dustbin_top = bbox_top(snapshot.dustbin.pose, snapshot.dustbin.bbox, self.config.dustbin_fallback_half_height)
        dustbin_bottom = bbox_bottom(
            snapshot.dustbin.pose,
            snapshot.dustbin.bbox,
            self.config.dustbin_fallback_half_height,
        )

        # Left-side bottles first keeps the bin-side workspace uncluttered.
        bottle_order = list(bottle_labels) if bottle_labels is not None else self._bottle_order(snapshot)
        if slot_indices is None:
            slot_indices = tuple(range(len(bottle_order)))
        if len(slot_indices) != len(bottle_order):
            raise ValueError("slot_indices must match bottle_labels")
        for bottle_index, label in enumerate(bottle_order):
            height_delta = float(rng.uniform(-self.config.height_jitter, self.config.height_jitter))
            drop_xy = rng.uniform(-self.config.drop_xy_jitter, self.config.drop_xy_jitter, size=2)
            drop_position = np.array(
                [
                    snapshot.dustbin.pose.position[0] + drop_xy[0],
                    snapshot.dustbin.pose.position[1] + drop_xy[1],
                    dustbin_top + self.config.drop_clearance + height_delta,
                ]
            )
            if self.config.drop_slot_offsets:
                drop_position[:2] += np.asarray(
                    self.config.drop_slot_offsets[slot_indices[bottle_index] % len(self.config.drop_slot_offsets)],
                    dtype=np.float64,
                )
            bottle = snapshot.bottles[label]
            bottle_center = bbox_world_center(bottle.pose, bottle.bbox)
            pick_arm = "left" if bottle.pose.position[0] <= self.config.direct_left_max_x else "right"
            if grippers[pick_arm] != self.config.open_value:
                grippers[pick_arm] = self.config.open_value
                events.append(
                    self._event(
                        Phase.RELEASE,
                        poses,
                        grippers,
                        pick_arm,
                        poses[pick_arm],
                        label,
                        self.config.gripper_hold_steps,
                    )
                )
            drop_position += np.asarray(getattr(self.config, f"{pick_arm}_drop_position_offset"), dtype=np.float64)
            orientation = self._orientation(pick_arm, snapshot)
            if self.config.align_grasp_to_bbox_major_axis:
                orientation = align_tool_yaw_to_bbox_major_axis(
                    orientation,
                    bottle.pose,
                    bottle.bbox,
                )
            if self.config.world_model_movements and bottle_index == 0 and rng.random() < self.config.push_probability:
                self._append_push(events, poses, grippers, bottle_center, orientation, pick_arm, label, rng)
                if self.config.home_between_bottles:
                    self._append_home(
                        events,
                        poses,
                        grippers,
                        pick_arm,
                        completion_height_offset=self.config.intermediate_home_completion_height_offset,
                    )
                continue
            top = bbox_top(bottle.pose, bottle.bbox, self.config.bottle_fallback_half_height)
            bottle_bottom = bbox_bottom(bottle.pose, bottle.bbox, self.config.bottle_fallback_half_height)
            grasp_position = bottle_center.copy()
            if self.config.grasp_center_offset is None:
                center_offset = top + self.config.grasp_clearance - bottle_center[2]
                if self.config.grasp_max_center_offset is not None:
                    center_offset = min(center_offset, self.config.grasp_max_center_offset)
                world_half_height = (top - bottle_bottom) / 2.0
                if (
                    self.config.upright_grasp_height_fraction is not None
                    and world_half_height > self.config.horizontal_bottle_half_height_max
                ):
                    center_offset = min(
                        center_offset,
                        world_half_height * self.config.upright_grasp_height_fraction,
                    )
                grasp_position[2] = bottle_center[2] + center_offset
            else:
                # RoboDojo's bottle assets include upright and sideways poses.
                # A tool-center offset from the live bounding-box center is
                # invariant to that orientation, unlike a local-z "top".
                grasp_position[2] = bottle_center[2] + self.config.grasp_center_offset
            grasp_position += np.asarray(getattr(self.config, f"{pick_arm}_grasp_position_offset"), dtype=np.float64)
            if top - bottle_center[2] <= self.config.horizontal_bottle_half_height_max:
                grasp_position += np.asarray(
                    getattr(self.config, f"{pick_arm}_horizontal_grasp_position_offset"),
                    dtype=np.float64,
                )
            # A tall reward-valid bin must be approached from above.  Lifting
            # by a fixed distance can leave the lower end of an upright or
            # sideways bottle below the rim, so a lateral transit strikes the
            # outside wall.  Preserve the grasp-to-object transform and raise
            # the complete oriented bound above the rim before translating.
            grasp_to_bottom = max(0.0, float(grasp_position[2] - bottle_bottom))
            rim_safe_tool_z = dustbin_top + self.config.rim_clearance + grasp_to_bottom
            drop_position[2] = max(drop_position[2], rim_safe_tool_z)
            pregrasp = Pose(
                grasp_position + np.array([0.0, 0.0, self.config.approach_height + height_delta]), orientation
            )
            grasp = Pose(grasp_position, orientation)
            lift_position = grasp_position + np.array([0.0, 0.0, self.config.lift_height + height_delta])
            lift_position[2] = max(lift_position[2], rim_safe_tool_z)
            lift = Pose(lift_position, orientation)

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
                    self._event(
                        Phase.APPROACH,
                        poses,
                        grippers,
                        pick_arm,
                        pregrasp,
                        label,
                        self.config.pregrasp_settle_steps,
                    ),
                    self._event(
                        Phase.GRASP,
                        poses,
                        grippers,
                        pick_arm,
                        grasp,
                        label,
                        self.config.grasp_settle_steps,
                    ),
                ]
            )
            grippers[pick_arm] = self.config.closed_value
            events.append(
                self._event(Phase.CLOSE, poses, grippers, pick_arm, grasp, label, self.config.gripper_hold_steps)
            )
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
            if self.config.world_model_movements:
                hold_steps = int(rng.integers(self.config.hold_steps_min, self.config.hold_steps_max + 1))
                events.append(self._event(Phase.HOLD, poses, grippers, pick_arm, lift, label, hold_steps))
                waypoint_count = int(rng.integers(self.config.carry_waypoints_min, self.config.carry_waypoints_max + 1))
                for _ in range(waypoint_count):
                    offset = np.array(
                        [
                            rng.uniform(-self.config.carry_xy_jitter, self.config.carry_xy_jitter),
                            rng.uniform(-self.config.carry_xy_jitter, self.config.carry_xy_jitter),
                            rng.uniform(-self.config.carry_z_jitter, self.config.carry_z_jitter),
                        ]
                    )
                    position = np.clip(
                        lift.position + offset,
                        np.asarray(self.config.workspace_min) + 0.02,
                        np.asarray(self.config.workspace_max) - 0.02,
                    )
                    events.append(
                        self._event(Phase.CARRY, poses, grippers, pick_arm, Pose(position, orientation), label)
                    )
                if waypoint_count:
                    events.append(self._event(Phase.CARRY, poses, grippers, pick_arm, lift, label))
            if self.config.stop_after_lift:
                return events

            carrying_arm = pick_arm
            if pick_arm == "right" and not self.config.direct_right_drop:
                right_handover = Pose(
                    self.config.handover_right_position, self._orientation("right", snapshot, handover=True)
                )
                events.append(self._event(Phase.HANDOVER, poses, grippers, "right", right_handover, label))
                if self.config.staged_handoff:
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
                    left_orientation = self._orientation("left", snapshot)
                    staged_grasp_position = np.asarray(
                        self.config.handover_staging_bottle_position, dtype=np.float64
                    ).copy()
                    staged_grasp_position[2] += (
                        self.config.bottle_fallback_half_height + self.config.grasp_clearance
                        if self.config.grasp_center_offset is None
                        else self.config.grasp_center_offset
                    )
                    staged_grasp_position += np.asarray(self.config.left_grasp_position_offset, dtype=np.float64)
                    staged_grasp = Pose(staged_grasp_position, left_orientation)
                    staged_pregrasp = staged_grasp.at(
                        staged_grasp.position + np.array([0.0, 0.0, self.config.approach_height])
                    )
                    staged_overhead = Pose(
                        np.array(
                            [
                                poses["left"].position[0],
                                poses["left"].position[1],
                                staged_pregrasp.position[2],
                            ]
                        ),
                        left_orientation,
                    )
                    events.append(self._event(Phase.APPROACH, poses, grippers, "left", staged_overhead, label))
                    events.append(self._event(Phase.APPROACH, poses, grippers, "left", staged_pregrasp, label))
                    events.append(self._event(Phase.GRASP, poses, grippers, "left", staged_grasp, label))
                    grippers["left"] = self.config.closed_value
                    events.append(
                        self._event(
                            Phase.CLOSE,
                            poses,
                            grippers,
                            "left",
                            staged_grasp,
                            label,
                            self.config.gripper_hold_steps,
                        )
                    )
                    left_lift = staged_grasp.at(staged_grasp.position + np.array([0.0, 0.0, self.config.lift_height]))
                    events.append(
                        self._event(
                            Phase.LIFT,
                            poses,
                            grippers,
                            "left",
                            left_lift,
                            label,
                            self.config.lift_hold_steps,
                        )
                    )
                else:
                    left_handover = Pose(
                        self.config.handover_left_position, self._orientation("left", snapshot, handover=True)
                    )
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

            # Keep the yaw used to align the gripper with the bottle's major
            # axis. Reverting to the generic arm orientation during transit
            # twists a securely held bottle and makes its landing footprint
            # unpredictable. Handover paths still use the receiving arm's
            # calibrated orientation.
            carry_orientation = orientation if carrying_arm == pick_arm else self._orientation(carrying_arm, snapshot)
            carry_pose = Pose(drop_position, carry_orientation)
            events.append(self._event(Phase.TRANSIT, poses, grippers, carrying_arm, carry_pose, label))
            if self.config.place_clearance is not None:
                place_position = drop_position.copy()
                place_position[2] = dustbin_bottom + grasp_to_bottom + self.config.place_clearance
                carry_pose = Pose(place_position, carry_orientation)
                events.append(self._event(Phase.TRANSIT, poses, grippers, carrying_arm, carry_pose, label))
            if self.config.drop_hold_steps:
                events.append(
                    self._event(
                        Phase.HOLD,
                        poses,
                        grippers,
                        carrying_arm,
                        carry_pose,
                        label,
                        self.config.drop_hold_steps,
                    )
                )
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
            if self.config.home_between_bottles:
                for arm in dict.fromkeys((pick_arm, carrying_arm)):
                    self._append_home(
                        events,
                        poses,
                        grippers,
                        arm,
                        completion_height_offset=self.config.intermediate_home_completion_height_offset,
                    )

        # Full reward requires both grippers open and both arms within 15 cm
        # and 20 degrees of their episode-start poses. Return through a high
        # waypoint: a direct diagonal move from the bin can sweep the long X5
        # fingers through an object that was just released successfully.
        if append_final_home:
            for arm in ("left", "right"):
                self._append_home(events, poses, grippers, arm)
        return events

    def _append_home(
        self,
        events: list[_Event],
        poses: dict[str, Pose],
        grippers: dict[str, float],
        arm: str,
        completion_height_offset: float | None = None,
    ) -> None:
        grippers[arm] = self.config.open_value
        current_pose = poses[arm]
        home_pose = self._home[arm]
        height_offset = (
            self.config.home_completion_height_offset if completion_height_offset is None else completion_height_offset
        )
        home_target = home_pose.at(home_pose.position + np.array([0.0, 0.0, height_offset]))
        if (
            np.linalg.norm(current_pose.position - home_target.position) < 1e-6
            and abs(float(np.dot(current_pose.quaternion, home_target.quaternion))) > 1.0 - 1e-9
        ):
            events.append(self._event(Phase.HOME, poses, grippers, arm, home_target, None))
            return
        clearance = min(
            self.config.workspace_max[2] - 0.02,
            max(current_pose.position[2], home_pose.position[2]) + self.config.home_clearance_height,
        )
        raised = current_pose.at(np.array([current_pose.position[0], current_pose.position[1], clearance]))
        home_overhead = Pose(
            np.array([home_pose.position[0], home_pose.position[1], clearance]),
            current_pose.quaternion,
        )
        events.append(self._event(Phase.HOME, poses, grippers, arm, raised, None))
        events.append(self._event(Phase.HOME, poses, grippers, arm, home_overhead, None))
        events.append(
            self._event(
                Phase.HOME,
                poses,
                grippers,
                arm,
                home_target,
                None,
                self.config.home_hold_steps,
            )
        )

    def _append_push(
        self,
        events: list[_Event],
        poses: dict[str, Pose],
        grippers: dict[str, float],
        bottle_position: np.ndarray,
        orientation: np.ndarray,
        arm: str,
        label: str,
        rng: np.random.Generator,
    ) -> None:
        direction = 1.0 if rng.random() < 0.5 else -1.0
        start = bottle_position.copy()
        start[1] -= direction * self.config.push_distance * 0.5
        start[2] += self.config.push_center_offset
        end = start.copy()
        end[1] += direction * self.config.push_distance
        overhead_start = start + np.array([0.0, 0.0, self.config.approach_height])
        current_overhead = np.array([poses[arm].position[0], poses[arm].position[1], overhead_start[2]])
        events.append(self._event(Phase.APPROACH, poses, grippers, arm, Pose(current_overhead, orientation), label))
        events.append(self._event(Phase.APPROACH, poses, grippers, arm, Pose(overhead_start, orientation), label))
        grippers[arm] = self.config.closed_value
        events.append(
            self._event(
                Phase.CLOSE,
                poses,
                grippers,
                arm,
                Pose(overhead_start, orientation),
                label,
                self.config.gripper_hold_steps,
            )
        )
        events.append(self._event(Phase.PUSH, poses, grippers, arm, Pose(start, orientation), label))
        events.append(self._event(Phase.PUSH, poses, grippers, arm, Pose(end, orientation), label))
        retreat = end + np.array([0.0, 0.0, self.config.retreat_height])
        events.append(self._event(Phase.RETREAT, poses, grippers, arm, Pose(retreat, orientation), label))
        grippers[arm] = self.config.open_value
        events.append(
            self._event(
                Phase.RELEASE,
                poses,
                grippers,
                arm,
                Pose(retreat, orientation),
                label,
                self.config.gripper_hold_steps,
            )
        )

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
            loaded_phases = {Phase.HOLD, Phase.CARRY, Phase.TRANSIT, Phase.HANDOVER}
            if event.phase is Phase.HOME and self.config.max_home_translation_step is not None:
                max_step = self.config.max_home_translation_step
            elif event.phase is Phase.GRASP and self.config.max_grasp_translation_step is not None:
                max_step = self.config.max_grasp_translation_step
            elif event.phase is Phase.LIFT and current_grippers[arm] == self.config.closed_value:
                max_step = self.config.max_lift_translation_step
            elif event.phase in loaded_phases and current_grippers[arm] == self.config.closed_value:
                if self.config.live_pose_feedback and self._retry_counts.get(event.bottle or "", 0):
                    max_step = self.config.retry_carry_translation_step
                else:
                    max_step = self.config.max_carry_translation_step
            else:
                max_step = self.config.max_translation_step
            waypoints = interpolate_pose(current[arm], target, max_step)
            for waypoint in waypoints:
                action_poses = dict(current)
                action_poses[arm] = waypoint
                result.append(self._planned_action(action_poses, current_grippers, event.phase, event.bottle, arm))
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
