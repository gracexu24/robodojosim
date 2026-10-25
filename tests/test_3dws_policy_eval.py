import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_policy_in_3dws.py"
SPEC = importlib.util.spec_from_file_location("evaluate_policy_in_3dws", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_action_state_round_trip_keeps_pose_and_gripper():
    action = np.array(
        [-0.3, -0.2, 0.9, 0.2, 1.4, -2.8, 1.0, 0.3, -0.1, 1.0, -0.4, 1.2, 2.9, 0.0]
    )
    state = MODULE.action14_to_state16(action)
    recovered = MODULE.state16_to_action14(state, action)
    np.testing.assert_allclose(recovered, action, atol=1e-6)


def test_wrist_camera_rig_follows_cartesian_target():
    action = np.zeros(14)
    action[[2, 9]] = 1.0
    state = MODULE.action14_to_state16(action)
    intrinsics = np.repeat(np.eye(3)[None], 3, axis=0)
    extrinsics = np.repeat(np.eye(4)[None], 3, axis=0)
    rig = MODULE.camera_rig(state, intrinsics, extrinsics)
    moved = state.copy()
    moved[0] += 0.1
    moved[8] -= 0.2
    cameras = MODULE.cameras_for_state(rig, moved)
    left_cam2world = np.linalg.inv(cameras[1]["viewmat"])
    right_cam2world = np.linalg.inv(cameras[2]["viewmat"])
    np.testing.assert_allclose(left_cam2world[0, 3], 0.1, atol=1e-6)
    np.testing.assert_allclose(right_cam2world[0, 3], -0.2, atol=1e-6)
