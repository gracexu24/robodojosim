import unittest
from types import SimpleNamespace

import numpy as np

from scripts.serve_3dws_bottle_viser import (
    apply_keyboard_action,
    action_to_poses,
    bbox_segments,
    blocked_keyboard_axes,
    camera_to_renderer,
    clamp_action,
    evenly_sample,
    gaussian_covariances,
    update_action_pose,
)


class ViewerUtilityTests(unittest.TestCase):
    def test_action_pose_round_trip(self):
        action = np.array(
            [-0.2, -0.3, 1.0, 0.1, 0.2, -0.3, 0.75, 0.3, -0.2, 1.1, -0.2, 0.4, 1.0, 0.25]
        )
        poses = action_to_poses(action)
        rebuilt = action.copy()
        for arm, (position, wxyz, gripper) in enumerate(poses):
            rebuilt = update_action_pose(rebuilt, arm, position, wxyz, gripper)
        self.assertTrue(np.allclose(rebuilt, action, atol=1e-7))

    def test_clamp_action_respects_each_dimension(self):
        lower = np.arange(14, dtype=float)
        upper = lower + 1.0
        action = lower + 0.5
        action[0] = -100
        action[-1] = 100
        clamped = clamp_action(action, lower, upper)
        self.assertEqual(clamped[0], lower[0])
        self.assertEqual(clamped[-1], upper[-1])
        self.assertTrue(np.allclose(clamped[1:-1], action[1:-1]))

    def test_even_sampling_is_bounded_and_includes_ends(self):
        self.assertTrue(np.array_equal(evenly_sample(4, 10), np.arange(4)))
        sampled = evenly_sample(100, 7)
        self.assertEqual(len(sampled), 7)
        self.assertEqual(sampled[0], 0)
        self.assertEqual(sampled[-1], 99)

    def test_bbox_has_twelve_three_dimensional_segments(self):
        segments = bbox_segments(np.array([0, 1, 2, 3, 4, 5]))
        self.assertEqual(segments.shape, (12, 2, 3))
        self.assertEqual(set(np.unique(segments[:, :, 0])), {0.0, 1.0})

    def test_keyboard_moves_both_arms_and_clamps(self):
        action = np.zeros(14)
        lower = np.full(14, -0.01)
        upper = np.full(14, 0.01)
        moved = apply_keyboard_action(
            action, {"KeyD", "KeyQ", "KeyI", "KeyL"}, 0.02, lower, upper
        )
        self.assertEqual(moved[0], 0.01)
        self.assertEqual(moved[1], 0.01)
        self.assertEqual(moved[7], 0.01)
        self.assertEqual(moved[9], 0.01)
        self.assertTrue(np.all(moved[[2, 3, 4, 5, 6, 8, 10, 11, 12, 13]] == 0.0))

    def test_opposed_keyboard_keys_cancel(self):
        action = np.zeros(14)
        moved = apply_keyboard_action(
            action, {"KeyA", "KeyD", "KeyU", "KeyO"}, 0.005,
            np.full(14, -1.0), np.full(14, 1.0)
        )
        self.assertTrue(np.allclose(moved, action))

    def test_workspace_limit_warning_names_blocked_axis(self):
        action = np.zeros(14)
        lower = np.full(14, -1.0)
        upper = np.full(14, 1.0)
        action[0] = lower[0]
        action[9] = upper[9]
        blocked = blocked_keyboard_axes(action, {"KeyA", "KeyI", "KeyD"}, lower, upper)
        self.assertEqual(blocked, ["left x minimum (A)", "right z maximum (I)"])

    def test_gaussian_covariance_respects_scale_and_rotation(self):
        scale = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        identity = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        covariance = gaussian_covariances(scale, identity)
        self.assertTrue(np.allclose(covariance[0], np.diag([1.0, 4.0, 9.0])))

    def test_viser_camera_conversion_is_opencv_world_to_camera(self):
        camera = SimpleNamespace(
            image_width=800,
            image_height=400,
            aspect=2.0,
            fov=np.pi / 2,
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position=np.array([1.0, 2.0, 3.0]),
        )
        converted = camera_to_renderer(camera, max_width=640)
        self.assertEqual((converted["W"], converted["H"]), (640, 320))
        self.assertTrue(np.allclose(converted["viewmat"][:3, 3], [-1, -2, -3]))
        self.assertAlmostEqual(float(converted["K"][0, 0]), 160.0, places=5)


if __name__ == "__main__":
    unittest.main()
