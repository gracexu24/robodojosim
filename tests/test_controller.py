import numpy as np
import pytest

from robodojosim.controller import BottleController, ControllerConfig, Phase, SafetyError
from robodojosim.mock_env import MockBottleEnv
from robodojosim.types import Pose


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
    assert len(first) != len(different) or any(
        not np.array_equal(left, right) for left, right in zip(first, different)
    )


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
