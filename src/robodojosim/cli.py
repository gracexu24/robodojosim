from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import run_campaign
from .controller import BottleController, ControllerConfig
from .datasets import summarize_dataset, write_splits
from .layouts import centralize_dustbin, normalize_bottle_mass
from .lerobot_export import export_lerobot
from .mock_env import MockBottleEnv
from .recording import EpisodeRecorder, validate_episode


def _dry_run(args: argparse.Namespace) -> int:
    config = ControllerConfig.from_json(args.config) if args.config else ControllerConfig()
    output = Path(args.output)
    results = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        env = MockBottleEnv(seed)
        controller = BottleController(config)
        controller.reset(env.snapshot())
        if EpisodeRecorder.is_complete(output, episode) and not args.overwrite:
            results.append({"episode": episode, "skipped": True})
            continue
        recorder = EpisodeRecorder(
            output,
            episode,
            seed=seed,
            instruction=env.snapshot().instruction,
            metadata={"environment": "kinematic_mock", "not_physics_data": True},
        )
        while not controller.done:
            observation = env.observation()
            planned = controller.next_action(env.snapshot())
            recorder.append(observation, planned.action, teacher=planned.privileged)
            env.step(planned.action)
        path = recorder.close(success=env.success, reason="mock_success" if env.success else "mock_failure")
        errors = validate_episode(path)
        results.append(
            {
                "episode": episode,
                "seed": seed,
                "steps": recorder.length,
                "success": env.success,
                "validation_errors": errors,
                "path": str(path),
            }
        )
    print(json.dumps(results, indent=2))
    return (
        0 if all(item.get("skipped") or (item["success"] and not item["validation_errors"]) for item in results) else 1
    )


def _validate(args: argparse.Namespace) -> int:
    failed = False
    for name in args.paths:
        errors = validate_episode(name)
        print(f"{name}: {'OK' if not errors else '; '.join(errors)}")
        failed = failed or bool(errors)
    return int(failed)


def _plan(args: argparse.Namespace) -> int:
    env = MockBottleEnv(args.seed)
    config = ControllerConfig.from_json(args.config) if args.config else ControllerConfig()
    controller = BottleController(config)
    controller.reset(env.snapshot())
    counts: dict[str, int] = {}
    while not controller.done:
        phase = controller.next_action().phase.value
        counts[phase] = counts.get(phase, 0) + 1
    print(json.dumps({"total_actions": sum(counts.values()), "actions_by_phase": counts}, indent=2))
    return 0


def _report(args: argparse.Namespace) -> int:
    stats = summarize_dataset(args.dataset, profile=args.profile)
    data = stats.to_dict()
    if args.field:
        print(data[args.field])
    else:
        print(json.dumps(data, indent=2))
    return 0


def _split(args: argparse.Namespace) -> int:
    path = write_splits(
        args.dataset,
        output=args.output,
        profile=args.profile,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )
    print(path)
    return 0


def _campaign(args: argparse.Namespace) -> int:
    stats = run_campaign(
        profile=args.profile,
        xpolicylab=args.xpolicylab,
        policy_env=args.policy_env,
        eval_env=args.eval_env,
        output=args.output,
        policy_seed=args.policy_seed,
        controller_config=args.config,
        runner=args.runner,
        target_episodes=args.target_episodes,
        target_hours=args.target_hours,
        batch_size=args.batch_size,
        max_passes=args.max_passes,
        lerobot_env=args.lerobot_env or None,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(json.dumps(stats.to_dict(), indent=2))
    return 0


def _fix_layouts(args: argparse.Namespace) -> int:
    changed = normalize_bottle_mass(args.robodojo_root, dry_run=args.dry_run)
    dustbins = (
        centralize_dustbin(args.robodojo_root, target_x=args.dustbin_x, dry_run=args.dry_run)
        if args.centralize_dustbin
        else []
    )
    print(
        json.dumps(
            {
                "mass_layouts": len(changed),
                "training_dustbin_layouts": len(dustbins),
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )
    return 0


def _export_lerobot(args: argparse.Namespace) -> int:
    output = export_lerobot(
        args.dataset,
        args.output or Path(args.dataset) / "lerobot",
        profile=args.profile,
        repo_id=args.repo_id,
        overwrite=args.overwrite,
    )
    print(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoboDojo scripted-controller utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry = subparsers.add_parser("dry-run", help="run the controller and recorder against the kinematic mock")
    dry.add_argument("--episodes", type=int, default=1)
    dry.add_argument("--seed", type=int, default=0)
    dry.add_argument("--output", default="outputs/mock_dataset")
    dry.add_argument("--config")
    dry.add_argument("--overwrite", action="store_true")
    dry.set_defaults(func=_dry_run)
    validate = subparsers.add_parser("validate", help="validate one or more recorded HDF5 files")
    validate.add_argument("paths", nargs="+")
    validate.set_defaults(func=_validate)
    plan = subparsers.add_parser("plan", help="show a mock plan summary without recording")
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--config")
    plan.set_defaults(func=_plan)
    report = subparsers.add_parser("report", help="summarize complete and successful dataset duration")
    report.add_argument("--dataset", required=True)
    report.add_argument("--profile", choices=("policy", "world_model"))
    report.add_argument(
        "--field",
        choices=(
            "complete_episodes",
            "complete_frames",
            "complete_seconds",
            "complete_hours",
            "mean_complete_seconds",
            "successful_episodes",
            "failed_episodes",
            "successful_frames",
            "successful_seconds",
            "successful_hours",
            "mean_successful_seconds",
            "corrupt_or_partial_files",
        ),
    )
    report.set_defaults(func=_report)
    split = subparsers.add_parser("split", help="create layout-grouped train/validation/test splits")
    split.add_argument("--dataset", required=True)
    split.add_argument("--output")
    split.add_argument("--profile", choices=("policy", "world_model"))
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--train-ratio", type=float, default=0.8)
    split.add_argument("--validation-ratio", type=float, default=0.1)
    split.set_defaults(func=_split)
    campaign = subparsers.add_parser("campaign", help="collect until a policy-episode or world-model-hour quota")
    campaign.add_argument("--profile", required=True, choices=("policy", "world_model"))
    campaign.add_argument("--xpolicylab", required=True)
    campaign.add_argument("--policy-env", default="RoboDojo")
    campaign.add_argument("--eval-env", default="RoboDojo")
    campaign.add_argument("--output", required=True)
    campaign.add_argument("--policy-seed", type=int, default=0)
    campaign.add_argument("--config")
    campaign.add_argument("--runner")
    campaign.add_argument("--target-episodes", type=int, default=50)
    campaign.add_argument("--target-hours", type=float, default=6.0)
    campaign.add_argument("--batch-size", type=int, default=25)
    campaign.add_argument("--max-passes", type=int)
    campaign.add_argument(
        "--lerobot-env",
        default="RoboDojoLeRobot",
        help="Conda environment for final LeRobot conversion; pass an empty string to use the current environment",
    )
    campaign.add_argument("--dry-run", action="store_true")
    campaign.set_defaults(func=_campaign)
    layouts = subparsers.add_parser("fix-layouts", help="correct the known 22 kg bottle layout typo")
    layouts.add_argument("--robodojo-root", required=True)
    layouts.add_argument(
        "--centralize-dustbin",
        action="store_true",
        help="install the shared tabletop training-bin variant (legacy flag name)",
    )
    layouts.add_argument("--dustbin-x", type=float, default=0.0)
    layouts.add_argument("--dry-run", action="store_true")
    layouts.set_defaults(func=_fix_layouts)
    lerobot = subparsers.add_parser("export-lerobot", help="convert complete episodes to LeRobotDataset v3.0")
    lerobot.add_argument("--dataset", required=True)
    lerobot.add_argument("--output")
    lerobot.add_argument("--profile", choices=("policy", "world_model"))
    lerobot.add_argument("--repo-id", default="robodojosim/bottle-task")
    lerobot.add_argument("--overwrite", action="store_true")
    lerobot.set_defaults(func=_export_lerobot)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
