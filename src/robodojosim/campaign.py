from __future__ import annotations

import json
import math
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from .datasets import WORLD_MODEL_PROFILES, DatasetStats, summarize_dataset, write_splits
from .lerobot_export import export_lerobot


@dataclass(frozen=True)
class CampaignTarget:
    profile: str
    target_episodes: int | None
    target_seconds: float | None
    successful_seconds_only: bool = False

    def progress_seconds(self, stats: DatasetStats) -> float:
        return stats.successful_seconds if self.successful_seconds_only else stats.complete_seconds

    def progress_episodes(self, stats: DatasetStats) -> int:
        return stats.successful_episodes if self.successful_seconds_only else stats.complete_episodes

    def reached(self, stats: DatasetStats) -> bool:
        if self.target_episodes is not None:
            return stats.successful_episodes >= self.target_episodes
        assert self.target_seconds is not None
        return self.progress_seconds(stats) >= self.target_seconds

    def remaining_batch_size(self, stats: DatasetStats, maximum: int) -> int:
        if self.target_episodes is not None:
            return max(1, min(maximum, self.target_episodes - stats.successful_episodes))
        assert self.target_seconds is not None
        remaining = self.target_seconds - self.progress_seconds(stats)
        mean_seconds = stats.mean_successful_seconds if self.successful_seconds_only else stats.mean_complete_seconds
        if mean_seconds <= 0:
            return maximum
        return max(1, min(maximum, math.ceil(remaining / mean_seconds)))


def campaign_target(profile: str, *, target_episodes: int = 50, target_hours: float = 6.0) -> CampaignTarget:
    if profile == "policy":
        if target_episodes < 1:
            raise ValueError("target_episodes must be positive")
        return CampaignTarget(profile, target_episodes, None)
    if profile in WORLD_MODEL_PROFILES:
        if target_hours <= 0:
            raise ValueError("target_hours must be positive")
        return CampaignTarget(
            profile,
            None,
            target_hours * 3600.0,
            successful_seconds_only=profile == "world_model_success",
        )
    raise ValueError(f"unknown campaign profile: {profile}")


def default_config(profile: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "configs" / f"{profile}_data.json"


def default_runner() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "run_collection.sh"


def _evaluation_seed(xpolicylab: str | Path, initial_seed: int, pass_index: int) -> int:
    """Cycle through the layout-seed directories installed by RoboDojo.

    Pass indices remain monotonic for episode IDs and trajectory variants, but
    RoboDojo's public asset bundle contains only a bounded set of seed
    directories. Falling back to the historical arithmetic keeps dry runs and
    custom runners usable when the asset tree is not locally available.
    """

    seed_root = (
        Path(xpolicylab).resolve().parent / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5"
    )
    if not seed_root.is_dir():
        return initial_seed + pass_index
    seeds = sorted(int(path.name) for path in seed_root.iterdir() if path.is_dir() and path.name.isdigit())
    if not seeds:
        return initial_seed + pass_index
    try:
        start = seeds.index(initial_seed)
    except ValueError:
        start = initial_seed % len(seeds)
    return seeds[(start + pass_index) % len(seeds)]


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
    lerobot_env: str | None = "RoboDojoLeRobot",
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
            lerobot_output = _export_campaign_lerobot(output, profile, lerobot_env=lerobot_env)
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
        evaluation_seed = _evaluation_seed(xpolicylab, policy_seed, pass_index)
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
                # Supported policy and world-model campaigns always use the
                # brown/yellow two-bottle scene.  Do not inherit an old shell
                # override that silently restores the deprecated four-bottle
                # handoff task.
                "ROBODOJOSIM_TWO_BOTTLE_TASK": "1",
            }
        )
        command = [
            "bash",
            str(runner_path),
            str(Path(xpolicylab).resolve()),
            str(evaluation_seed),
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
                        "layout_seed": evaluation_seed,
                        "scene_variant": "two_bottle_brown_yellow",
                        "current_stats": stats.to_dict(),
                    },
                    indent=2,
                )
            )
            return stats

        before_progress = (
            stats.successful_episodes if profile == "policy" else target.progress_episodes(stats)
        )
        subprocess.run(command, env=environment, check=True)
        updated = summarize_dataset(output, profile=profile)
        state["next_pass"] = pass_index + 1
        state["no_progress_passes"] = (
            int(state.get("no_progress_passes", 0)) + 1
            if (updated.successful_episodes if profile == "policy" else target.progress_episodes(updated))
            == before_progress
            else 0
        )
        state["status"] = "running"
        state["stats"] = updated.to_dict()
        _save_state(state_path, state)
        if state["no_progress_passes"] >= 3:
            raise RuntimeError("three consecutive passes produced no usable episodes; calibrate before continuing")
        passes_this_run += 1


def _export_campaign_lerobot(output: Path, profile: str, *, lerobot_env: str | None) -> Path:
    destination = output / "lerobot"
    repo_id = f"robodojosim/bottle-{profile.replace('_', '-')}"
    if lerobot_env:
        subprocess.run(
            [
                "conda",
                "run",
                "-n",
                lerobot_env,
                "robodojosim",
                "export-lerobot",
                "--dataset",
                str(output),
                "--output",
                str(destination),
                "--profile",
                profile,
                "--repo-id",
                repo_id,
                "--overwrite",
            ],
            check=True,
        )
        return destination
    return export_lerobot(output, destination, profile=profile, repo_id=repo_id, overwrite=True)


def run_balanced_world_model_campaign(
    *,
    xpolicylab: str | Path,
    policy_env: str,
    eval_env: str,
    output: str | Path,
    total_hours: float = 6.0,
    policy_seed: int = 0,
    runner: str | Path | None = None,
    batch_size: int = 25,
    max_passes: int | None = None,
    lerobot_env: str | None = "RoboDojoLeRobot",
    dry_run: bool = False,
) -> dict[str, DatasetStats]:
    """Collect equal recorded-time success, failure-event, and miscellaneous datasets."""

    if total_hours <= 0:
        raise ValueError("total_hours must be positive")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    per_category_hours = total_hours / 3.0
    categories = (
        ("success", "world_model_success"),
        ("failure", "world_model_failure"),
        ("miscellaneous", "world_model_misc"),
    )
    results: dict[str, DatasetStats] = {}
    for index, (directory, profile) in enumerate(categories):
        results[directory] = run_campaign(
            profile=profile,
            xpolicylab=xpolicylab,
            policy_env=policy_env,
            eval_env=eval_env,
            output=root / directory,
            policy_seed=policy_seed + index,
            runner=runner,
            target_hours=per_category_hours,
            batch_size=batch_size,
            max_passes=max_passes,
            lerobot_env=lerobot_env,
            dry_run=dry_run,
        )
    if not dry_run:
        payload = {
            "format_version": 1,
            "target_total_hours": total_hours,
            "target_hours_per_category": per_category_hours,
            "categories": {name: stats.to_dict() for name, stats in results.items()},
        }
        _save_state(root / "balanced_campaign_state.json", payload)
    return results
