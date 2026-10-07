from pathlib import Path

import numpy as np
import pytest

from robodojosim.controller import BottleController, ControllerConfig, Phase, SafetyError
from robodojosim.geometry import bbox_top, quaternion_rotation_matrix
from robodojosim.mock_env import MockBottleEnv
from robodojosim.types import ObjectState, Pose


def test_production_profiles_use_physically_calibrated_left_grasp_center():
    config_root = Path(__file__).parents[1] / "configs"
    for name in ("bottle_task.json", "policy_data.json", "world_model_data.json"):
        config = ControllerConfig.from_json(config_root / name)
        assert config.left_grasp_position_offset == [0.0, 0.0, 0.0]


def test_pose_normalizes_quaternion():
    pose = Pose([1, 2, 3], [2, 0, 0, 0])
    np.testing.assert_allclose(pose.as_array(), [1, 2, 3, 1, 0, 0, 0])


def test_controller_completes_mock_task_with_handover():
    env = MockBottleEnv(seed=7)
    controller = BottleController()
    controller.reset(env.snapshot())
    phases = []
    previous = dict(env.arms)
    while not controller.done:
        planned = controller.next_action(env.snapshot())
        phases.append(planned.phase)
        assert set(planned.action) == {
            "left_ee_pose",
            "right_ee_pose",
            "left_ee_joint_state",
            "right_ee_joint_state",
        }
        for arm in ("left", "right"):
            current = Pose.from_array(planned.action[f"{arm}_ee_pose"])
            assert np.linalg.norm(current.position - previous[arm].position) <= 0.03501
            previous[arm] = current
        env.step(planned.action)
    assert Phase.HANDOVER in phases
    assert env.success


def test_trajectory_variants_are_deterministic_and_change_the_plan():
    config = ControllerConfig(vary_bottle_order=True, height_jitter=0.01, drop_xy_jitter=0.02)
    snapshot = MockBottleEnv(seed=3).snapshot()

    def actions(variant):
        controller = BottleController(config, trajectory_variant=variant)
        controller.reset(snapshot)
        return [controller.next_action().action["left_ee_pose"] for _ in range(controller.planned_action_count)]

    first = actions(4)
    repeated = actions(4)
    different = actions(5)
    assert len(first) == len(repeated)
    assert all(np.array_equal(left, right) for left, right in zip(first, repeated))
    assert len(first) != len(different) or any(not np.array_equal(left, right) for left, right in zip(first, different))


def test_workspace_guard_rejects_unsafe_target():
    env = MockBottleEnv()
    snapshot = env.snapshot()
    bottles = dict(snapshot.bottles)
    first = bottles["bottle0"]
    bottles["bottle0"] = type(first)(Pose([-2, 0, 0.8], [1, 0, 0, 0]), first.bbox)
    unsafe = type(snapshot)(snapshot.arms, snapshot.grippers, bottles, snapshot.dustbin)
    with pytest.raises(SafetyError):
        BottleController(ControllerConfig()).reset(unsafe)


def test_center_offset_calibration_plan_stops_after_one_lift():
    env = MockBottleEnv(seed=2)
    config = ControllerConfig(
        grasp_center_offset=0.06,
        bottle_limit=1,
        stop_after_lift=True,
        lift_hold_steps=8,
    )
    controller = BottleController(config)
    controller.reset(env.snapshot())
    planned = [controller.next_action() for _ in range(controller.planned_action_count)]
    assert planned[-1].phase is Phase.LIFT
    assert {step.bottle for step in planned} == {"bottle0"}
    assert sum(step.phase is Phase.LIFT for step in planned) >= 8


def test_overhead_approach_rises_before_translating_to_bottle():
    env = MockBottleEnv(seed=1)
    snapshot = env.snapshot()
    config = ControllerConfig(grasp_center_offset=0.1, bottle_limit=1, stop_after_lift=True)
    controller = BottleController(config)
    controller.reset(snapshot)
    approach = []
    while controller.phase is Phase.APPROACH:
        approach.append(controller.next_action())
    first_bottle = min(snapshot.bottles, key=lambda name: snapshot.bottles[name].pose.position[0])
    arm = "left"
    home = snapshot.arms[arm].position
    target = snapshot.bottles[first_bottle].pose.position
    # The arm first climbs near its home XY; only later does it translate over
    # the target bottle at the full pre-grasp height.
    first = approach[0].action["left_ee_pose"][:3]
    last = approach[-1].action["left_ee_pose"][:3]
    assert np.linalg.norm(first[:2] - home[:2]) < np.linalg.norm(last[:2] - home[:2])
    np.testing.assert_allclose(last[:2], target[:2], atol=1e-6)
    assert last[2] > target[2] + config.grasp_center_offset


def test_arm_specific_grasp_position_offset_is_applied():
    env = MockBottleEnv(seed=1)
    snapshot = env.snapshot()
    offset = np.array([0.006, 0.025, 0.0])
    controller = BottleController(
        ControllerConfig(
            grasp_center_offset=0.055,
            left_grasp_position_offset=tuple(offset),
            bottle_limit=1,
            stop_after_lift=True,
        )
    )
    controller.reset(snapshot)
    grasp_steps = []
    while not controller.done:
        step = controller.next_action()
        if step.phase is Phase.GRASP:
            grasp_steps.append(step)
    bottle = min(snapshot.bottles, key=lambda name: snapshot.bottles[name].pose.position[0])
    target = grasp_steps[-1].action["left_ee_pose"][:3]
    expected = snapshot.bottles[bottle].pose.position + offset
    expected[2] += 0.055
    np.testing.assert_allclose(target, expected)


def test_grasp_targets_world_center_of_asymmetric_mesh_bounds():
    snapshot = MockBottleEnv(seed=1).snapshot()
    bottles = dict(snapshot.bottles)
    original = bottles["bottle3"]
    # The mesh origin is at one end: its local AABB center is +8 cm on x.
    bottles["bottle3"] = ObjectState(
        original.pose,
        np.array([0.02, -0.03, -0.04, 0.14, 0.03, 0.04]),
    )
    snapshot = type(snapshot)(snapshot.arms, snapshot.grippers, bottles, snapshot.dustbin)
    controller = BottleController(
        ControllerConfig(
            grasp_center_offset=0.055,
            bottle_labels=("bottle3",),
            stop_after_lift=True,
        )
    )
    controller.reset(snapshot)
    grasp = None
    while not controller.done:
        step = controller.next_action()
        if step.phase is Phase.GRASP:
            grasp = step.action["right_ee_pose"][:3]
    expected = original.pose.position + np.array([0.08, 0.0, 0.055])
    np.testing.assert_allclose(grasp, expected)


def test_grasp_yaw_aligns_tool_fingers_with_projected_bbox_major_axis():
    snapshot = MockBottleEnv(seed=1).snapshot()
    bottles = dict(snapshot.bottles)
    original = bottles["bottle3"]
    bottles["bottle3"] = ObjectState(
        original.pose,
        np.array([-0.03, -0.02, -0.04, 0.03, 0.18, 0.04]),
    )
    snapshot = type(snapshot)(snapshot.arms, snapshot.grippers, bottles, snapshot.dustbin)
    controller = BottleController(
        ControllerConfig(
            grasp_center_offset=0.055,
            bottle_labels=("bottle3",),
            stop_after_lift=True,
            right_grasp_quaternion=(-0.353523, 0.61239, -0.353524, -0.61239),
            align_grasp_to_bbox_major_axis=True,
        )
    )
    controller.reset(snapshot)
    grasp = None
    while not controller.done:
        step = controller.next_action()
        if step.phase is Phase.GRASP:
            grasp = step.action["right_ee_pose"][3:]
    tool_length = quaternion_rotation_matrix(grasp)[:2, 2]
    tool_length /= np.linalg.norm(tool_length)
    assert abs(float(np.dot(tool_length, [0.0, 1.0]))) > 0.999


def test_bbox_top_transforms_oriented_bounds_to_world_space():
    # Rotate a 24 cm local-z bottle 90 degrees around world y. Its world
    # half-height is then the 4 cm local-x extent, not its 12 cm length.
    pose = Pose([0.2, -0.1, 0.8], [np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])
    bbox = np.array([-0.04, -0.03, -0.12, 0.04, 0.03, 0.12])
    assert bbox_top(pose, bbox, 0.0) == pytest.approx(0.84)


def test_loaded_lift_uses_smaller_motion_steps():
    env = MockBottleEnv(seed=1)
    controller = BottleController(
        ControllerConfig(
            max_translation_step=0.04,
            max_lift_translation_step=0.01,
            max_carry_translation_step=0.01,
            grasp_center_offset=0.055,
            bottle_limit=1,
            stop_after_lift=True,
        )
    )
    controller.reset(env.snapshot())
    lift_positions = []
    while not controller.done:
        step = controller.next_action()
        if step.phase is Phase.LIFT:
            lift_positions.append(step.action["left_ee_pose"][:3])
    deltas = np.linalg.norm(np.diff(lift_positions, axis=0), axis=1)
    assert np.max(deltas) <= 0.010001


def test_world_model_plan_contains_push_hold_and_randomized_carry():
    config = ControllerConfig(
        world_model_movements=True,
        push_probability=1.0,
        bottle_limit=2,
        hold_steps_min=3,
        hold_steps_max=5,
        carry_waypoints_min=2,
        carry_waypoints_max=3,
        carry_xy_jitter=0.05,
        carry_z_jitter=0.03,
        max_actions=700,
    )
    controller = BottleController(config, trajectory_variant=9)
    controller.reset(MockBottleEnv(seed=4).snapshot())
    phases = []
    while not controller.done:
        phases.append(controller.next_action().phase)
    assert Phase.PUSH in phases
    assert Phase.HOLD in phases
    assert Phase.CARRY in phases


def test_bottle_labels_can_isolate_right_arm_calibration():
    snapshot = MockBottleEnv(seed=0).snapshot()
    controller = BottleController(ControllerConfig(bottle_labels=("bottle3",), bottle_limit=1, stop_after_lift=True))
    controller.reset(snapshot)
    active_arms = set()
    bottles = set()
    while not controller.done:
        step = controller.next_action()
        active_arms.add(step.active_arm)
        bottles.add(step.bottle)
    assert active_arms == {"right"}
    assert bottles == {"bottle3"}


def test_direct_right_drop_avoids_handoff_for_central_bin_variant():
    snapshot = MockBottleEnv(seed=0).snapshot()
    controller = BottleController(
        ControllerConfig(
            bottle_labels=("bottle3",),
            bottle_limit=1,
            direct_right_drop=True,
        )
    )
    controller.reset(snapshot)
    phases = [controller.next_action().phase for _ in range(controller.planned_action_count)]
    assert Phase.HANDOVER not in phases
    assert Phase.TRANSIT in phases


def test_arm_specific_drop_position_offsets_are_applied():
    snapshot = MockBottleEnv(seed=0).snapshot()
    controller = BottleController(
        ControllerConfig(
            bottle_labels=("bottle0", "bottle3"),
            direct_right_drop=True,
            left_drop_position_offset=(-0.15, 0.01, 0.02),
            right_drop_position_offset=(0.12, -0.01, 0.03),
        )
    )
    controller.reset(snapshot)
    transit_targets = {}
    while not controller.done:
        step = controller.next_action()
        if step.phase is Phase.TRANSIT:
            transit_targets[step.bottle] = step.action[f"{step.active_arm}_ee_pose"][:3]
    dustbin = snapshot.dustbin.pose.position
    np.testing.assert_allclose(transit_targets["bottle0"][:2], dustbin[:2] + [-0.15, 0.01])
    np.testing.assert_allclose(transit_targets["bottle3"][:2], dustbin[:2] + [0.12, -0.01])
    assert transit_targets["bottle0"][2] > dustbin[2]
    assert transit_targets["bottle3"][2] > dustbin[2]


def test_drop_hold_settles_with_closed_gripper_before_release():
    controller = BottleController(ControllerConfig(bottle_limit=1, direct_right_drop=True, drop_hold_steps=4))
    controller.reset(MockBottleEnv(seed=0).snapshot())
    planned = [controller.next_action() for _ in range(controller.planned_action_count)]
    release_index = next(i for i, step in enumerate(planned) if step.phase is Phase.RELEASE)
    hold = planned[release_index - 4 : release_index]
    assert [step.phase for step in hold] == [Phase.HOLD] * 4
    assert all(float(step.action["left_ee_joint_state"][0]) == 0.0 for step in hold)
