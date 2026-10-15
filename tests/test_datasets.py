import json

import robodojosim.campaign as campaign_module
from robodojosim.campaign import campaign_target, run_campaign
from robodojosim.controller import BottleController
from robodojosim.datasets import summarize_dataset, write_splits
from robodojosim.mock_env import MockBottleEnv
from robodojosim.recording import EpisodeRecorder


def _one_frame_episode(directory, episode_id, layout_seed, success=True, profile="policy"):
    env = MockBottleEnv(layout_seed)
    controller = BottleController()
    controller.reset(env.snapshot())
    planned = controller.next_action()
    recorder = EpisodeRecorder(
        directory,
        episode_id,
        seed=layout_seed,
        instruction="test",
        metadata={
            "layout_seed": layout_seed,
            "collection_profile": profile,
            "trajectory_variant": episode_id // 100_000,
        },
    )
    recorder.append(env.observation(), planned.action)
    recorder.close(success=success)


def test_dataset_stats_and_layout_grouped_splits(tmp_path):
    for episode_id, layout_seed in enumerate([0, 0, 1, 1, 2, 2]):
        _one_frame_episode(tmp_path, episode_id, layout_seed, success=episode_id != 5)
    stats = summarize_dataset(tmp_path, profile="policy")
    assert stats.complete_episodes == 6
    assert stats.complete_frames == 6
    assert stats.complete_seconds == 0.24
    assert stats.successful_episodes == 5
    assert stats.failed_episodes == 1
    assert stats.successful_frames == 5
    assert stats.successful_seconds == 0.2

    split_path = write_splits(tmp_path, profile="policy", seed=4, train_ratio=0.5, validation_ratio=0.25)
    payload = json.loads(split_path.read_text())
    seen = {}
    for split_name, split in payload["splits"].items():
        for layout in split["layout_seeds"]:
            assert layout not in seen
            seen[layout] = split_name
    assert set(seen) == {0, 1, 2}


def test_world_model_stats_and_splits_include_complete_failed_interactions(tmp_path):
    _one_frame_episode(tmp_path, 0, 0, success=False, profile="world_model")
    _one_frame_episode(tmp_path, 1, 1, success=True, profile="world_model")
    stats = summarize_dataset(tmp_path, profile="world_model")
    assert stats.complete_episodes == 2
    assert stats.complete_seconds == 0.08
    assert stats.successful_episodes == 1
    split_path = write_splits(tmp_path, profile="world_model", train_ratio=0.5, validation_ratio=0.0)
    payload = json.loads(split_path.read_text())
    episodes = {name for split in payload["splits"].values() for name in split["episodes"]}
    assert episodes == {"episode_000000.hdf5", "episode_000001.hdf5"}


def test_campaign_targets_and_dry_run(tmp_path, capsys):
    stats = summarize_dataset(tmp_path, profile="world_model")
    world = campaign_target("world_model", target_hours=6)
    policy = campaign_target("policy", target_episodes=50)
    assert not world.reached(stats)
    assert policy.remaining_batch_size(stats, 25) == 25
    balanced_success = campaign_target("world_model_success", target_hours=1 / 3)
    assert balanced_success.successful_seconds_only is True
    assert balanced_success.target_seconds == 1200

    run_campaign(
        profile="policy",
        xpolicylab=tmp_path / "XPolicyLab",
        policy_env="RoboDojo",
        eval_env="RoboDojo",
        output=tmp_path / "campaign",
        dry_run=True,
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["evaluation_count"] == 25
    assert payload["episode_offset"] == 0


def test_campaign_cycles_installed_layout_seeds_without_reusing_episode_ids(tmp_path, capsys):
    xpolicylab = tmp_path / "RoboDojo" / "XPolicyLab"
    seed_root = tmp_path / "RoboDojo" / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5"
    for seed in (0, 1, 2):
        (seed_root / str(seed)).mkdir(parents=True)
    output = tmp_path / "campaign"
    output.mkdir()
    (output / "campaign_state.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "campaign_id": "test",
                "profile": "policy",
                "next_pass": 7,
                "no_progress_passes": 0,
            }
        )
    )

    run_campaign(
        profile="policy",
        xpolicylab=xpolicylab,
        policy_env="RoboDojoPolicy",
        eval_env="RoboDojo",
        output=output,
        policy_seed=0,
        dry_run=True,
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["layout_seed"] == 1
    assert payload["episode_offset"] == 700_000
    assert payload["trajectory_variant"] == 7


def test_policy_campaign_reaches_exactly_50_successes_and_writes_splits(tmp_path, monkeypatch):
    output = tmp_path / "campaign"

    def fake_export(_dataset, _profile, **_kwargs):
        destination = output / "lerobot"
        destination.mkdir(parents=True)
        return destination

    def fake_run(command, env, check):
        assert check is True
        assert env["ROBODOJOSIM_TWO_BOTTLE_TASK"] == "1"
        count = int(env["ROBODOJOSIM_EVAL_NUM"])
        offset = int(env["ROBODOJOSIM_EPISODE_OFFSET"])
        profile = env["ROBODOJOSIM_COLLECTION_PROFILE"]
        for layout_seed in range(count):
            _one_frame_episode(output, offset + layout_seed, layout_seed, profile=profile)

    monkeypatch.setattr(campaign_module.subprocess, "run", fake_run)
    monkeypatch.setattr(campaign_module, "_export_campaign_lerobot", fake_export)
    stats = run_campaign(
        profile="policy",
        xpolicylab=tmp_path / "XPolicyLab",
        policy_env="RoboDojo",
        eval_env="RoboDojo",
        output=output,
        target_episodes=50,
    )
    assert stats.successful_episodes == 50
    state = json.loads((output / "campaign_state.json").read_text())
    assert state["status"] == "complete"
    assert state["next_pass"] == 2
    assert (output / "splits.json").exists()


def test_campaign_refreshes_existing_lerobot_export(tmp_path, monkeypatch):
    output = tmp_path / "campaign"
    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        assert check is True

    monkeypatch.setattr(campaign_module.subprocess, "run", fake_run)
    result = campaign_module._export_campaign_lerobot(output, "policy", lerobot_env="LeRobot")
    assert result == output / "lerobot"
    assert "--overwrite" in captured["command"]
