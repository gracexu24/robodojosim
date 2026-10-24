import unittest

import numpy as np

from scripts.serve_3dws_bottle_viser import (
    action_to_poses,
    bbox_segments,
    clamp_action,
    evenly_sample,
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


if __name__ == "__main__":
    unittest.main()
