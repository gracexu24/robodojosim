import numpy as np
import pytest

from robodojosim.inference import action_mapping, observation_images, observation_state, sanitize_action
from robodojosim.mock_env import MockBottleEnv


def test_policy_observation_mapping_matches_export_contract():
    observation = MockBottleEnv(3).observation()
    observation["vision"]["cam_left_wrist"] = observation["vision"]["cam_head"]
    observation["vision"]["cam_right_wrist"] = observation["vision"]["cam_head"]
    state = observation_state(observation)
    images = observation_images(observation)
    assert state.shape == (16,)
    assert state.dtype == np.float32
    assert set(images) == {
        "observation.images.cam_head",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    }
    assert all(image.shape == (16, 16, 3) for image in images.values())


def test_policy_action_is_normalized_clamped_and_converted():
    action = np.zeros(16, dtype=np.float32)
    action[0:3] = [-2.0, 2.0, 2.0]
    action[3] = 2.0
    action[7] = -1.0
    action[8:11] = [2.0, -2.0, -1.0]
    action[11] = 3.0
    action[15] = 2.0
    safe = sanitize_action(action)
    np.testing.assert_allclose(safe[0:3], [-0.85, 0.40, 1.35])
    np.testing.assert_allclose(safe[8:11], [0.75, -0.55, 0.30])
    assert np.isclose(np.linalg.norm(safe[3:7]), 1.0)
    assert np.isclose(np.linalg.norm(safe[11:15]), 1.0)
    assert safe[7] == 0.0 and safe[15] == 1.0
    mapped = action_mapping(action)
    assert mapped["left_ee_pose"].shape == (7,)
    assert mapped["right_ee_joint_state"].shape == (1,)


def test_policy_action_rejects_non_finite_and_zero_quaternion():
    action = np.ones(16, dtype=np.float32)
    action[2] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        sanitize_action(action)
    action[2] = 0.5
    action[3:7] = 0
    with pytest.raises(ValueError, match="zero-length quaternion"):
        sanitize_action(action)
