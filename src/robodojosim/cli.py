from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import run_balanced_world_model_campaign, run_campaign
from .controller import BottleController, ControllerConfig
from .datasets import WORLD_MODEL_PROFILES, summarize_dataset, write_splits
from .evaluation import EvaluationSpec, run_evaluation, summarize_evaluation
from .hub import format_push_plan, push_dataset, push_model
from .layouts import centralize_dustbin, normalize_bottle_mass
from .lerobot_export import export_lerobot
from .mock_env import MockBottleEnv
from .pose_analysis import analyze_pose_distribution
from .recording import EpisodeRecorder, validate_episode
from .showcase import render_skill_video
from .training import TrainingSpec, inspect_lerobot_dataset, run_training
from .world_model_export import (
    WorldModelExportConfig,
    export_3dws_dataset,
    inspect_3dws_episode,
    upgrade_3dws_sim_depth_metadata,
    validate_3dws_packed,
)
from .world_model_qa import analyze_3dws_quality, render_3dws_review
from .world_model_setup import configure_world_model_observations
from .world_model_training import parse_training_root, stage_3dws_latents, stage_3dws_training


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
    paths: list[Path] = []
    for name in args.paths:
        candidate = Path(name)
        paths.extend(sorted(candidate.glob("episode_*.hdf5")) if candidate.is_dir() else [candidate])
    if not paths:
        print("no episode_*.hdf5 files found")
        return 1
    for path in paths:
        errors = validate_episode(path)
        print(f"{path}: {'OK' if not errors else '; '.join(errors)}")
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
        bottle_side_order=args.bottle_side_order,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(json.dumps(stats.to_dict(), indent=2))
    return 0


def _balanced_campaign(args: argparse.Namespace) -> int:
    results = run_balanced_world_model_campaign(
        xpolicylab=args.xpolicylab,
        policy_env=args.policy_env,
        eval_env=args.eval_env,
        output=args.output,
        total_hours=args.target_hours,
        policy_seed=args.policy_seed,
        runner=args.runner,
        batch_size=args.batch_size,
        max_passes=args.max_passes,
        lerobot_env=args.lerobot_env or None,
        bottle_side_order=args.bottle_side_order,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(json.dumps({name: stats.to_dict() for name, stats in results.items()}, indent=2))
    return 0


def _fix_layouts(args: argparse.Namespace) -> int:
    changed = normalize_bottle_mass(args.robodojo_root, dry_run=args.dry_run)
    dustbins = (
        centralize_dustbin(
            args.robodojo_root,
            target_x=args.dustbin_x,
            two_bottle_brown_yellow=args.two_bottle_brown_yellow,
            bottle_side_order=args.bottle_side_order,
            dry_run=args.dry_run,
        )
        if args.centralize_dustbin
        else []
    )
    print(
        json.dumps(
            {
                "mass_layouts": len(changed),
                "training_dustbin_layouts": len(dustbins),
                "two_bottle_brown_yellow": args.two_bottle_brown_yellow,
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


def _world_model_config(args: argparse.Namespace) -> WorldModelExportConfig:
    return WorldModelExportConfig(
        bbox=tuple(args.bbox),
        target_fps=args.target_fps,
        jpeg_quality=args.jpeg_quality,
        source_extrinsic=args.source_extrinsic,
        source_camera_axes=args.source_camera_axes,
        object_mask_margin=args.object_mask_margin,
        require_object_masks=not args.geometry_only,
    )


def _inspect_3dws(args: argparse.Namespace) -> int:
    paths: list[Path] = []
    for name in args.paths:
        candidate = Path(name)
        paths.extend(sorted(candidate.glob("episode_*.hdf5")) if candidate.is_dir() else [candidate])
    reports = [inspect_3dws_episode(path).to_dict() for path in paths]
    print(json.dumps(reports, indent=2, sort_keys=True))
    return int(not reports or any(not report["compatible"] for report in reports))


def _export_3dws(args: argparse.Namespace) -> int:
    output = export_3dws_dataset(args.dataset, args.output, _world_model_config(args))
    print(output)
    return 0


def _validate_3dws(args: argparse.Namespace) -> int:
    failed = False
    paths: list[Path] = []
    for name in args.paths:
        candidate = Path(name)
        paths.extend(sorted(candidate.glob("**/episode_*.hdf5")) if candidate.is_dir() else [candidate])
    for path in paths:
        errors = validate_3dws_packed(path)
        print(f"{path}: {'OK' if not errors else '; '.join(errors)}")
        failed = failed or bool(errors)
    return int(failed or not paths)


def _upgrade_3dws_depth_metadata(args: argparse.Namespace) -> int:
    totals = {"checked": 0, "upgraded": 0, "current": 0}
    for name in args.paths:
        result = upgrade_3dws_sim_depth_metadata(name, dry_run=args.dry_run)
        for key in totals:
            totals[key] += result[key]
    totals["dry_run"] = args.dry_run
    print(json.dumps(totals, indent=2, sort_keys=True))
    return 0


def _configure_world_model(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            configure_world_model_observations(
                args.robodojo_root,
                restore=args.restore,
                dry_run=args.dry_run,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _review_3dws(args: argparse.Namespace) -> int:
    report = analyze_3dws_quality(args.episode)
    video = render_3dws_review(
        args.episode,
        args.output,
        fps=args.fps,
        maximum_frames=args.maximum_frames,
    )
    report_path = Path(args.report) if args.report else Path(args.output).with_suffix(".quality.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"video": str(video), "report": str(report_path), **report}, indent=2, sort_keys=True))
    return int(not report["passed"])


def _stage_3dws_training(args: argparse.Namespace) -> int:
    manifest = stage_3dws_training(
        [parse_training_root(value) for value in args.root],
        args.output,
    )
    print(manifest)
    return 0


def _stage_3dws_latents(args: argparse.Namespace) -> int:
    manifest = stage_3dws_latents(args.manifest, args.source_root, args.output)
    print(manifest)
    return 0


def _episode_indices(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 0 for item in result):
        raise ValueError("episodes must be a comma-separated list of non-negative LeRobot indices")
    return result


def _train(args: argparse.Namespace) -> int:
    result = run_training(
        TrainingSpec(
            policy=args.policy,
            dataset=Path(args.dataset),
            output=Path(args.output),
            repo_id=args.dataset_repo_id,
            device=args.device,
            gpu=args.gpu,
            steps=args.steps,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            episodes=_episode_indices(args.episodes),
            resume_from=Path(args.resume_from) if args.resume_from else None,
            wandb=args.wandb,
            push_to_hub=args.push_to_hub,
            model_repo_id=args.model_repo_id,
        ),
        dry_run=args.dry_run,
        executable=args.executable,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def _inspect_lerobot(args: argparse.Namespace) -> int:
    print(json.dumps(inspect_lerobot_dataset(args.dataset), indent=2, sort_keys=True))
    return 0


def _evaluate_policy(args: argparse.Namespace) -> int:
    result = run_evaluation(
        EvaluationSpec(
            xpolicylab=Path(args.xpolicylab),
            checkpoint=Path(args.checkpoint),
            output=Path(args.output),
            episodes=args.episodes,
            seed=args.seed,
            policy_gpu=args.policy_gpu,
            environment_gpu=args.env_gpu,
            policy_env=args.policy_env,
            eval_env=args.eval_env,
            gui=args.gui,
        ),
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def _summarize_evaluation(args: argparse.Namespace) -> int:
    print(json.dumps(summarize_evaluation(args.output), indent=2, sort_keys=True))
    return 0


def _render_skill_video(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            render_skill_video(args.episode, args.output, camera=args.camera),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _analyze_poses(args: argparse.Namespace) -> int:
    print(json.dumps(analyze_pose_distribution(args.dataset, args.output), indent=2, sort_keys=True))
    return 0


def _push_dataset(args: argparse.Namespace) -> int:
    result = push_dataset(
        args.dataset,
        args.repo_id,
        private=not args.public,
        revision=args.revision,
        dry_run=args.dry_run,
    )
    print(format_push_plan(result))
    return 0


def _push_model(args: argparse.Namespace) -> int:
    result = push_model(
        args.checkpoint,
        args.repo_id,
        private=not args.public,
        revision=args.revision,
        dry_run=args.dry_run,
    )
    print(format_push_plan(result))
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
    validate = subparsers.add_parser("validate", help="validate HDF5 files or every episode in a dataset directory")
    validate.add_argument("paths", nargs="+")
    validate.set_defaults(func=_validate)
    plan = subparsers.add_parser("plan", help="show a mock plan summary without recording")
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--config")
    plan.set_defaults(func=_plan)
    report = subparsers.add_parser("report", help="summarize complete and successful dataset duration")
    report.add_argument("--dataset", required=True)
    profiles = ("policy", *WORLD_MODEL_PROFILES)
    report.add_argument("--profile", choices=profiles)
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
    split.add_argument("--profile", choices=profiles)
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--train-ratio", type=float, default=0.8)
    split.add_argument("--validation-ratio", type=float, default=0.1)
    split.set_defaults(func=_split)
    campaign = subparsers.add_parser("campaign", help="collect until a policy-episode or world-model-hour quota")
    campaign.add_argument("--profile", required=True, choices=profiles)
    campaign.add_argument("--xpolicylab", required=True)
    campaign.add_argument("--policy-env", default="RoboDojoPolicy")
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
        "--bottle-side-order",
        choices=("alternating", "brown_left", "yellow_left"),
        default="alternating",
        help="force bottle colors to one side or alternate them by layout",
    )
    campaign.add_argument(
        "--lerobot-env",
        default="RoboDojoLeRobot",
        help="Conda environment for final LeRobot conversion; pass an empty string to use the current environment",
    )
    campaign.add_argument("--dry-run", action="store_true")
    campaign.set_defaults(func=_campaign)
    balanced = subparsers.add_parser(
        "balanced-world-model",
        help="collect equal recorded time in success, recoverable-failure, and miscellaneous datasets",
    )
    balanced.add_argument("--xpolicylab", required=True)
    balanced.add_argument("--policy-env", default="RoboDojoPolicy")
    balanced.add_argument("--eval-env", default="RoboDojo")
    balanced.add_argument("--output", required=True)
    balanced.add_argument("--policy-seed", type=int, default=0)
    balanced.add_argument("--runner")
    balanced.add_argument("--target-hours", type=float, default=6.0)
    balanced.add_argument("--batch-size", type=int, default=25)
    balanced.add_argument("--max-passes", type=int)
    balanced.add_argument(
        "--bottle-side-order",
        choices=("alternating", "brown_left", "yellow_left"),
        default="alternating",
        help="force bottle colors to one side or alternate them by layout",
    )
    balanced.add_argument("--lerobot-env", default="RoboDojoLeRobot")
    balanced.add_argument("--dry-run", action="store_true")
    balanced.set_defaults(func=_balanced_campaign)
    poses = subparsers.add_parser(
        "analyze-bottle-poses",
        help="export every-frame bottle positions and render balanced-campaign distribution plots",
    )
    poses.add_argument("--dataset", required=True)
    poses.add_argument("--output", required=True)
    poses.set_defaults(func=_analyze_poses)
    layouts = subparsers.add_parser("fix-layouts", help="correct the known 22 kg bottle layout typo")
    layouts.add_argument("--robodojo-root", required=True)
    layouts.add_argument(
        "--centralize-dustbin",
        action="store_true",
        help="install the shared tabletop training-bin variant",
    )
    layouts.add_argument("--dustbin-x", type=float, default=0.0)
    layouts.add_argument(
        "--two-bottle-brown-yellow",
        action="store_true",
        help="install the supported brown/yellow two-bottle layouts in alternating left/right order",
    )
    layouts.add_argument(
        "--bottle-side-order",
        choices=("alternating", "brown_left", "yellow_left"),
        default="alternating",
        help="force bottle colors to one side or alternate them by layout",
    )
    layouts.add_argument("--dry-run", action="store_true")
    layouts.set_defaults(func=_fix_layouts)
    lerobot = subparsers.add_parser("export-lerobot", help="convert complete episodes to LeRobotDataset v3.0")
    lerobot.add_argument("--dataset", required=True)
    lerobot.add_argument("--output")
    lerobot.add_argument("--profile", choices=profiles)
    lerobot.add_argument("--repo-id", default="robodojosim/bottle-task")
    lerobot.add_argument("--overwrite", action="store_true")
    lerobot.set_defaults(func=_export_lerobot)
    wm_setup = subparsers.add_parser(
        "configure-world-model-observations",
        help="enable or restore RoboDojo metric depth and camera calibration capture",
    )
    wm_setup.add_argument("--robodojo-root", required=True)
    wm_setup.add_argument("--restore", action="store_true")
    wm_setup.add_argument("--dry-run", action="store_true")
    wm_setup.set_defaults(func=_configure_world_model)
    inspect_3dws = subparsers.add_parser(
        "inspect-3dws",
        help="report whether source episodes contain the geometry needed by 3DWS",
    )
    inspect_3dws.add_argument("paths", nargs="+")
    inspect_3dws.set_defaults(func=_inspect_3dws)
    export_3dws = subparsers.add_parser(
        "export-3dws",
        help="convert enhanced RoboDojo HDF5 episodes to the 3DWS packed_v2 contract",
    )
    export_3dws.add_argument("--dataset", required=True)
    export_3dws.add_argument("--output", required=True)
    export_3dws.add_argument(
        "--bbox",
        required=True,
        nargs=6,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    export_3dws.add_argument("--target-fps", type=float, default=10.0)
    export_3dws.add_argument("--jpeg-quality", type=int, default=95)
    export_3dws.add_argument("--source-extrinsic", choices=("camera_to_world", "world_to_camera"), default="camera_to_world")
    export_3dws.add_argument("--source-camera-axes", choices=("isaac_usd", "opencv"), default="isaac_usd")
    export_3dws.add_argument("--object-mask-margin", type=float, default=0.005)
    export_3dws.add_argument("--geometry-only", action="store_true", help="allow a pack without object labels")
    export_3dws.set_defaults(func=_export_3dws)
    validate_3dws = subparsers.add_parser("validate-3dws", help="validate one or more packed_v2 episode files")
    validate_3dws.add_argument("paths", nargs="+")
    validate_3dws.set_defaults(func=_validate_3dws)
    upgrade_depth = subparsers.add_parser(
        "upgrade-3dws-depth-metadata",
        help="upgrade legacy RoboDojo simulator packs to the upstream full-scene depth marker",
    )
    upgrade_depth.add_argument("paths", nargs="+")
    upgrade_depth.add_argument("--dry-run", action="store_true")
    upgrade_depth.set_defaults(func=_upgrade_3dws_depth_metadata)
    review_3dws = subparsers.add_parser(
        "review-3dws",
        help="run numerical quality gates and render an RGB/depth/mask review video",
    )
    review_3dws.add_argument("--episode", required=True)
    review_3dws.add_argument("--output", required=True)
    review_3dws.add_argument("--report")
    review_3dws.add_argument("--fps", type=float)
    review_3dws.add_argument("--maximum-frames", type=int, default=600)
    review_3dws.set_defaults(func=_review_3dws)
    stage_3dws = subparsers.add_parser(
        "stage-3dws-training",
        help="stage several packed_v2 roots under globally unique bottle-training episode IDs",
    )
    stage_3dws.add_argument(
        "--root",
        action="append",
        required=True,
        metavar="LABEL=/ABSOLUTE/PATH",
        help="packed root containing train/ and val/; repeat for every category and side order",
    )
    stage_3dws.add_argument("--output", required=True)
    stage_3dws.set_defaults(func=_stage_3dws_training)
    stage_latents = subparsers.add_parser(
        "stage-3dws-latents",
        help="stage per-root 3DWS latent exports under the manifest's globally unique episode IDs",
    )
    stage_latents.add_argument("--manifest", required=True)
    stage_latents.add_argument("--source-root", required=True)
    stage_latents.add_argument("--output", required=True)
    stage_latents.set_defaults(func=_stage_3dws_latents)
    inspect_dataset = subparsers.add_parser("inspect-lerobot", help="validate a local training-facing LeRobot tree")
    inspect_dataset.add_argument("--dataset", required=True)
    inspect_dataset.set_defaults(func=_inspect_lerobot)
    train = subparsers.add_parser("train", help="train ACT or Diffusion Policy with LeRobot")
    train.add_argument("--policy", required=True, choices=("act", "diffusion"))
    train.add_argument("--dataset", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--dataset-repo-id", default="robodojosim/bottle-policy")
    train.add_argument("--device", default="cuda", choices=("cuda", "cpu", "mps"))
    train.add_argument("--gpu", type=int)
    train.add_argument("--steps", type=int, default=20_000)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--episodes", help="comma-separated LeRobot episode indices")
    train.add_argument("--resume-from", help="checkpoint train_config.json")
    train.add_argument("--wandb", action="store_true")
    train.add_argument("--push-to-hub", action="store_true")
    train.add_argument("--model-repo-id")
    train.add_argument("--executable", default="lerobot-train")
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(func=_train)
    evaluate = subparsers.add_parser("evaluate-policy", help="run a LeRobot checkpoint in RoboDojo/Isaac Sim")
    evaluate.add_argument("--xpolicylab", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--episodes", type=int, default=10)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--policy-gpu", type=int, default=0)
    evaluate.add_argument("--env-gpu", type=int, default=0)
    evaluate.add_argument("--policy-env", default="RoboDojoLeRobotTrain")
    evaluate.add_argument("--eval-env", default="RoboDojo")
    evaluate.add_argument("--gui", action="store_true")
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.set_defaults(func=_evaluate_policy)
    eval_report = subparsers.add_parser("summarize-evaluation", help="write JSON, CSV, and Markdown evaluation reports")
    eval_report.add_argument("--output", required=True)
    eval_report.set_defaults(func=_summarize_evaluation)
    skill_video = subparsers.add_parser(
        "render-skill-video",
        help="render a labeled MP4 from a world-model HDF5 episode",
    )
    skill_video.add_argument("--episode", required=True)
    skill_video.add_argument("--output", required=True)
    skill_video.add_argument("--camera", default="cam_head")
    skill_video.set_defaults(func=_render_skill_video)
    dataset_push = subparsers.add_parser("push-dataset", help="upload a validated LeRobot tree to Hugging Face")
    dataset_push.add_argument("--dataset", required=True)
    dataset_push.add_argument("--repo-id", required=True)
    dataset_push.add_argument("--revision", default="main")
    dataset_push.add_argument("--public", action="store_true", help="create a public repository (private by default)")
    dataset_push.add_argument("--dry-run", action="store_true")
    dataset_push.set_defaults(func=_push_dataset)
    model_push = subparsers.add_parser("push-model", help="upload a validated checkpoint to Hugging Face")
    model_push.add_argument("--checkpoint", required=True)
    model_push.add_argument("--repo-id", required=True)
    model_push.add_argument("--revision", default="main")
    model_push.add_argument("--public", action="store_true", help="create a public repository (private by default)")
    model_push.add_argument("--dry-run", action="store_true")
    model_push.set_defaults(func=_push_model)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
