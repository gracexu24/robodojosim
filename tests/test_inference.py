import math
from types import SimpleNamespace

import numpy as np
import pytest

from robodojosim.inference import (
    LeRobotPolicyRuntime,
    PolicySafetyConfig,
    PolicySafetyFilter,
    action_mapping,
    observation_images,
    observation_state,
    sanitize_action,
)
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


def _state_and_action():
    state = np.zeros(16, dtype=np.float32)
    state[0:3] = [-0.3, 0.0, 0.7]
    state[3] = 1.0
    state[7] = 1.0
    state[8:11] = [0.3, 0.0, 0.7]
    state[11] = 1.0
    state[15] = 1.0
    return state, state.copy()


def test_policy_safety_filter_rate_limits_pose_and_debounces_close():
    state, action = _state_and_action()
    action[0:3] += [0.3, 0.0, 0.0]
    action[3:7] = [0.0, 1.0, 0.0, 0.0]
    action[7] = 0.0
    safety = PolicySafetyFilter()

    first = safety.apply(action, state)
    assert np.isclose(np.linalg.norm(first[0:3] - state[0:3]), 0.015)
    angle = 2.0 * math.acos(float(np.clip(np.dot(first[3:7], state[3:7]), -1.0, 1.0)))
    assert np.isclose(angle, math.radians(5.0))
    assert first[7] == 1.0

    second = safety.apply(action, state)
    assert second[7] == 0.0
    summary = safety.summary()
    assert summary["steps"] == 2
    assert summary["position_limited"][0] == 2
    assert summary["rotation_limited"][0] == 2
    assert summary["close_commands"][0] == 1


def test_policy_safety_filter_latches_closed_until_stable_release():
    config = PolicySafetyConfig(open_debounce_steps=3, close_debounce_steps=1, minimum_closed_steps=3)
    safety = PolicySafetyFilter(config)
    state, action = _state_and_action()
    action[7] = 0.0
    assert safety.apply(action, state)[7] == 0.0

    action[7] = 1.0
    assert safety.apply(action, state)[7] == 0.0
    assert safety.apply(action, state)[7] == 0.0
    assert safety.apply(action, state)[7] == 1.0
    summary = safety.summary()
    assert summary["open_commands"][0] == 1
    assert summary["suppressed_open_steps"][0] == 2


def test_policy_safety_filter_aligns_equivalent_quaternion_signs():
    state, action = _state_and_action()
    action[3:7] = [-1.0, 0.0, 0.0, 0.0]
    filtered = PolicySafetyFilter().apply(action, state)
    np.testing.assert_allclose(filtered[3:7], state[3:7])


def test_runtime_enables_act_temporal_ensemble():
    runtime = LeRobotPolicyRuntime.__new__(LeRobotPolicyRuntime)
    runtime.temporal_ensemble_coeff = 0.01
    config = SimpleNamespace(type="act", n_action_steps=10, temporal_ensemble_coeff=None)
    runtime._configure_policy_execution(config)
    assert config.n_action_steps == 1
    assert config.temporal_ensemble_coeff == 0.01
    assert runtime._policy_execution == "act_temporal_ensemble"


def test_runtime_preserves_diffusion_action_queue():
    runtime = LeRobotPolicyRuntime.__new__(LeRobotPolicyRuntime)
    runtime.temporal_ensemble_coeff = 0.01
    config = SimpleNamespace(type="diffusion", n_action_steps=8)
    runtime._configure_policy_execution(config)
    assert config.n_action_steps == 8
    assert runtime._policy_execution == "diffusion_action_queue"


def test_runtime_rejects_act_without_temporal_ensemble_support():
    runtime = LeRobotPolicyRuntime.__new__(LeRobotPolicyRuntime)
    runtime.temporal_ensemble_coeff = 0.01
    with pytest.raises(TypeError, match="does not support temporal ensembling"):
        runtime._configure_policy_execution(SimpleNamespace(type="act", n_action_steps=10))


def test_runtime_rejects_unknown_policy_type():
    runtime = LeRobotPolicyRuntime.__new__(LeRobotPolicyRuntime)
    runtime.temporal_ensemble_coeff = 0.01
    with pytest.raises(TypeError, match="unsupported checkpoint policy type"):
        runtime._configure_policy_execution(SimpleNamespace(type="other", n_action_steps=1))
