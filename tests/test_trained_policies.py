import json
from pathlib import Path


def test_trained_policy_registry_marks_act_and_diffusion_checkpoints():
    path = Path(__file__).parents[1] / "configs" / "trained_policies.json"
    registry = json.loads(path.read_text())
    policies = {entry["family"]: entry for entry in registry["policies"]}
    assert registry["task"] == "two_bottle_brown_yellow"
    assert set(policies) == {"act", "diffusion"}
    assert all(entry["status"] == "trained" for entry in policies.values())
    assert all(entry["checkpoint"].endswith("/checkpoints/last/pretrained_model") for entry in policies.values())
