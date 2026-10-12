from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REQUIRED_FEATURES = {
    "observation.state": (16,),
    "action": (16,),
    "observation.images.cam_head": (3, 480, 640),
    "observation.images.cam_left_wrist": (3, 480, 640),
    "observation.images.cam_right_wrist": (3, 480, 640),
}


@dataclass(frozen=True)
class TrainingSpec:
    policy: str
    dataset: Path
    output: Path
    repo_id: str = "robodojosim/bottle-policy"
    device: str = "cuda"
    gpu: int | None = None
    steps: int = 20_000
    batch_size: int = 8
    num_workers: int = 4
    episodes: tuple[int, ...] | None = None
    resume_from: Path | None = None
    wandb: bool = False
    push_to_hub: bool = False
    model_repo_id: str | None = None

    def __post_init__(self) -> None:
        if self.policy not in {"act", "diffusion"}:
            raise ValueError("policy must be 'act' or 'diffusion'")
        if self.steps < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("steps and batch_size must be positive; num_workers cannot be negative")
        if self.gpu is not None and self.gpu < 0:
            raise ValueError("gpu must be non-negative")
        if self.push_to_hub and not self.model_repo_id:
            raise ValueError("model_repo_id is required when push_to_hub is enabled")


def inspect_lerobot_dataset(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"not a complete LeRobot dataset: missing {info_path}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid LeRobot metadata: {info_path}: {exc}") from exc

    if int(info.get("total_episodes", 0)) < 1 or int(info.get("total_frames", 0)) < 1:
        raise ValueError("LeRobot metadata reports no episodes or frames")
    if int(info.get("fps", 0)) <= 0:
        raise ValueError("LeRobot metadata reports an invalid frame rate")
    features = info.get("features")
    if not isinstance(features, dict):
        raise TypeError("LeRobot metadata does not contain a feature map")
    for name, expected_shape in REQUIRED_FEATURES.items():
        feature = features.get(name)
        if not isinstance(feature, dict):
            raise TypeError(f"LeRobot dataset is missing required feature {name!r}")
        shape = tuple(int(value) for value in feature.get("shape", ()))
        if shape != expected_shape:
            raise ValueError(f"feature {name!r} has shape {shape}, expected {expected_shape}")

    data_files = sorted(root.glob("data/**/*.parquet"))
    video_files = sorted(root.glob("videos/**/*.mp4"))
    if not data_files:
        raise ValueError("LeRobot dataset has no Parquet data files")
    if len(video_files) < 3:
        raise ValueError("LeRobot dataset must contain all three camera video streams")
    return {
        "root": str(root),
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "total_episodes": int(info["total_episodes"]),
        "total_frames": int(info["total_frames"]),
        "fps": int(info["fps"]),
        "data_files": len(data_files),
        "video_files": len(video_files),
    }


def build_training_command(spec: TrainingSpec, executable: str = "lerobot-train") -> list[str]:
    if spec.resume_from is not None:
        return [executable, f"--config_path={spec.resume_from.resolve()}", "--resume=true"]

    command = [
        executable,
        f"--dataset.repo_id={spec.repo_id}",
        f"--dataset.root={spec.dataset.resolve()}",
        f"--policy.type={spec.policy}",
        f"--policy.device={spec.device}",
        f"--output_dir={spec.output.resolve()}",
        f"--job_name=bottle-{spec.policy}",
        f"--batch_size={spec.batch_size}",
        f"--steps={spec.steps}",
        f"--num_workers={spec.num_workers}",
        "--save_checkpoint=true",
        f"--save_freq={max(1, min(2_000, spec.steps))}",
        "--log_freq=100",
        "--dataset.image_transforms.enable=true",
        f"--wandb.enable={str(spec.wandb).lower()}",
        f"--policy.push_to_hub={str(spec.push_to_hub).lower()}",
    ]
    if spec.episodes is not None:
        command.append(f"--dataset.episodes={json.dumps(list(spec.episodes), separators=(',', ':'))}")
    if spec.model_repo_id:
        command.append(f"--policy.repo_id={spec.model_repo_id}")
    if spec.policy == "act":
        command.extend(("--policy.chunk_size=50", "--policy.n_action_steps=10"))
    else:
        command.extend(
            (
                "--policy.n_obs_steps=2",
                "--policy.horizon=32",
                "--policy.n_action_steps=8",
                "--policy.num_inference_steps=10",
            )
        )
    return command


def run_training(spec: TrainingSpec, *, dry_run: bool = False, executable: str = "lerobot-train") -> dict[str, Any]:
    dataset = inspect_lerobot_dataset(spec.dataset)
    command = build_training_command(spec, executable=executable)
    environment = os.environ.copy()
    if spec.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(spec.gpu)

    output = spec.output.resolve()
    manifest_path = output / "robodojosim_training.json"
    manifest: dict[str, Any] = {
        "format_version": 1,
        "status": "planned" if dry_run else "running",
        "created_unix": time.time(),
        "spec": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(spec).items()},
        "dataset": dataset,
        "command": command,
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
    }
    if dry_run:
        return manifest

    if spec.resume_from is None and output.exists():
        raise FileExistsError(f"training output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = output.parent / f".{output.name}.robodojosim_training.json"
    tracking_path = manifest_path if output.is_dir() else sidecar_path
    _write_json_atomic(tracking_path, manifest)
    try:
        subprocess.run(command, env=environment, check=True)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["finished_unix"] = time.time()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        failure_path = manifest_path if output.is_dir() else sidecar_path
        _write_json_atomic(failure_path, manifest)
        if failure_path != sidecar_path:
            sidecar_path.unlink(missing_ok=True)
        raise
    manifest["status"] = "complete"
    manifest["finished_unix"] = time.time()
    manifest["checkpoint"] = str(output / "checkpoints" / "last" / "pretrained_model")
    _write_json_atomic(manifest_path, manifest)
    sidecar_path.unlink(missing_ok=True)
    return manifest


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=list) + "\n", encoding="utf-8")
    os.replace(temporary, path)
