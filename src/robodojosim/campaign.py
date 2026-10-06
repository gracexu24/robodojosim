from __future__ import annotations

import json
import math
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from .datasets import DatasetStats, summarize_dataset, write_splits
from .lerobot_export import export_lerobot


@dataclass(frozen=True)
class CampaignTarget:
    profile: str
    target_episodes: int | None
    target_seconds: float | None

    def reached(self, stats: DatasetStats) -> bool:
        if self.target_episodes is not None:
            return stats.successful_episodes >= self.target_episodes
        assert self.target_seconds is not None
        return stats.complete_seconds >= self.target_seconds

    def remaining_batch_size(self, stats: DatasetStats, maximum: int) -> int:
        if self.target_episodes is not None:
            return max(1, min(maximum, self.target_episodes - stats.successful_episodes))
        assert self.target_seconds is not None
        remaining = self.target_seconds - stats.complete_seconds
        if stats.mean_complete_seconds <= 0:
            return maximum
        return max(1, min(maximum, math.ceil(remaining / stats.mean_complete_seconds)))


def campaign_target(profile: str, *, target_episodes: int = 50, target_hours: float = 6.0) -> CampaignTarget:
    if profile == "policy":
        if target_episodes < 1:
            raise ValueError("target_episodes must be positive")
        return CampaignTarget(profile, target_episodes, None)
    if profile == "world_model":
        if target_hours <= 0:
            raise ValueError("target_hours must be positive")
        return CampaignTarget(profile, None, target_hours * 3600.0)
    raise ValueError(f"unknown campaign profile: {profile}")


def default_config(profile: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "configs" / f"{profile}_data.json"


def default_runner() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "run_collection.sh"


def _load_state(path: Path, target: CampaignTarget) -> dict:
    if not path.exists():
        return {
            "format_version": 1,
            "campaign_id": str(uuid.uuid4()),
            "profile": target.profile,
            "next_pass": 0,
            "no_progress_passes": 0,
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("profile") != target.profile:
        raise ValueError(f"existing campaign profile is {state.get('profile')}, requested {target.profile}")
    return state


def _save_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_campaign(
    *,
    profile: str,
    xpolicylab: str | Path,
    policy_env: str,
    eval_env: str,
    output: str | Path,
    policy_seed: int = 0,
    controller_config: str | Path | None = None,
    runner: str | Path | None = None,
    target_episodes: int = 50,
    target_hours: float = 6.0,
    batch_size: int = 25,
    max_passes: int | None = None,
    dry_run: bool = False,
) -> DatasetStats:
    target = campaign_target(profile, target_episodes=target_episodes, target_hours=target_hours)
    if batch_size < 1 or batch_size > 25:
        raise ValueError("batch_size must be between 1 and RoboDojo's 25 available layouts")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "campaign_state.json"
    state = _load_state(state_path, target)
    config_path = Path(controller_config).resolve() if controller_config else default_config(profile)
    runner_path = Path(runner).resolve() if runner else default_runner()
    passes_this_run = 0

    while True:
        stats = summarize_dataset(output, profile=profile)
        if target.reached(stats):
            write_splits(output, profile=profile)
            lerobot_output = export_lerobot(
                output,
                output / "lerobot",
                profile=profile,
                repo_id=f"robodojosim/bottle-{profile.replace('_', '-')}",
            )
            state["status"] = "complete"
            state["stats"] = stats.to_dict()
            state["lerobot_output"] = str(lerobot_output)
            _save_state(state_path, state)
            return stats
        if max_passes is not None and passes_this_run >= max_passes:
            state["status"] = "paused"
            state["stats"] = stats.to_dict()
            _save_state(state_path, state)
            return stats

        pass_index = int(state["next_pass"])
        evaluation_count = target.remaining_batch_size(stats, batch_size)
        environment = os.environ.copy()
        environment.update(
            {
                "ROBODOJOSIM_COLLECTION_PROFILE": profile,
                "ROBODOJOSIM_CAMPAIGN_ID": str(state["campaign_id"]),
                "ROBODOJOSIM_TRAJECTORY_VARIANT": str(pass_index),
                "ROBODOJOSIM_EPISODE_OFFSET": str(pass_index * 100_000),
                "ROBODOJOSIM_CONFIG": str(config_path),
                "ROBODOJOSIM_EVAL_NUM": str(evaluation_count),
                "ROBODOJOSIM_JPEG_QUALITY": "90",
            }
        )
        command = [
            "bash",
            str(runner_path),
            str(Path(xpolicylab).resolve()),
            str(policy_seed + pass_index),
            policy_env,
            eval_env,
            str(output),
        ]
        if dry_run:
            print(
                json.dumps(
                    {
                        "command": command,
                        "profile": profile,
                        "pass": pass_index,
                        "evaluation_count": evaluation_count,
                        "episode_offset": pass_index * 100_000,
                        "trajectory_variant": pass_index,
                        "current_stats": stats.to_dict(),
                    },
                    indent=2,
                )
            )
            return stats

        before_progress = stats.successful_episodes if profile == "policy" else stats.complete_episodes
        subprocess.run(command, env=environment, check=True)
        updated = summarize_dataset(output, profile=profile)
        state["next_pass"] = pass_index + 1
        state["no_progress_passes"] = (
            int(state.get("no_progress_passes", 0)) + 1
            if (updated.successful_episodes if profile == "policy" else updated.complete_episodes) == before_progress
            else 0
        )
        state["status"] = "running"
        state["stats"] = updated.to_dict()
        _save_state(state_path, state)
        if state["no_progress_passes"] >= 3:
            raise RuntimeError("three consecutive passes produced no usable episodes; calibrate before continuing")
        passes_this_run += 1
