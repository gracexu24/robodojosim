import json

from robodojosim.evaluation import EvaluationSpec, build_evaluation_command, summarize_evaluation


def test_evaluation_command_uses_separate_policy_and_simulator_environments(tmp_path):
    spec = EvaluationSpec(
        xpolicylab=tmp_path / "XPolicyLab",
        checkpoint=tmp_path / "checkpoint",
        output=tmp_path / "results",
        policy_gpu=1,
        environment_gpu=0,
    )
    command = build_evaluation_command(spec)
    assert command[-4:] == ["1", "0", "RoboDojoLeRobotTrain", "RoboDojo"]
    assert "bottle_lerobot" in command[1]


def test_evaluation_summary_writes_github_readable_artifacts(tmp_path):
    records = [
        {"episode_id": 0, "layout_seed": 0, "success": True, "steps": 500, "termination_reason": "task_success"},
        {"episode_id": 1, "layout_seed": 1, "success": False, "steps": 900, "termination_reason": "policy_timeout"},
    ]
    for record in records:
        (tmp_path / f"episode_{record['episode_id']:06d}.json").write_text(json.dumps(record))
    report = summarize_evaluation(tmp_path)
    assert report["success_rate"] == 0.5
    assert report["mean_steps"] == 700
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "episodes.csv").is_file()
    assert "Success rate: 50.0%" in (tmp_path / "summary.md").read_text()
