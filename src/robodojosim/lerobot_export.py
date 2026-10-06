from __future__ import annotations

import inspect
import json
import os
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .datasets import EpisodeInfo, list_episodes
from .image_codec import decode_jpeg

STATE_NAMES = [
    "left_x",
    "left_y",
    "left_z",
    "left_qw",
    "left_qx",
    "left_qy",
    "left_qz",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_qw",
    "right_qx",
    "right_qy",
    "right_qz",
    "right_gripper",
]


def _eligible_episodes(dataset_dir: Path, profile: str | None) -> list[EpisodeInfo]:
    records, _ = list_episodes(dataset_dir, profile=profile)
    return [
        record
        for record in records
        if record.complete and (record.success or profile != "policy")
    ]


def _read_text(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _vector(handle: h5py.File, group: str, index: int) -> np.ndarray:
    parts = []
    for arm in ("left", "right"):
        pose_key = f"{group}/{arm}_ee_poses"
        gripper_key = f"{group}/{arm}_ee_joint_states"
        if pose_key not in handle or gripper_key not in handle:
            raise ValueError(f"{handle.filename}: missing /{pose_key} or /{gripper_key}")
        pose = np.asarray(handle[pose_key][index], dtype=np.float32).reshape(-1)
        gripper = np.asarray(handle[gripper_key][index], dtype=np.float32).reshape(-1)
        if pose.shape != (7,) or gripper.shape != (1,):
            raise ValueError(
                f"{handle.filename}: expected {arm} pose/gripper dimensions 7+1, "
                f"got {pose.shape}+{gripper.shape}"
            )
        parts.extend((pose, gripper))
    return np.concatenate(parts).astype(np.float32, copy=False)


def _camera_datasets(handle: h5py.File) -> dict[str, h5py.Dataset]:
    result: dict[str, h5py.Dataset] = {}
    vision = handle.get("vision")
    if not isinstance(vision, h5py.Group):
        return result
    for camera, camera_group in vision.items():
        if isinstance(camera_group, h5py.Group) and "colors" in camera_group:
            result[camera] = camera_group["colors"]
    return result


def _image(dataset: h5py.Dataset, index: int) -> np.ndarray:
    value = np.asarray(dataset[index])
    image = decode_jpeg(value) if value.ndim == 1 else value
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{dataset.file.filename}: invalid RGB frame {dataset.name}[{index}] {image.shape} {image.dtype}")
    return np.ascontiguousarray(image)


def _inspect_source(episodes: Iterable[EpisodeInfo]) -> tuple[int, dict[str, tuple[int, int]]]:
    frequency: int | None = None
    cameras: dict[str, tuple[int, int]] | None = None
    for episode in episodes:
        with h5py.File(episode.path, "r") as handle:
            episode_frequency = round(float(handle["additional_info/frequency"][()]))
            if episode_frequency <= 0:
                raise ValueError(f"{episode.path}: frequency must be positive")
            if frequency is None:
                frequency = episode_frequency
            elif frequency != episode_frequency:
                raise ValueError(f"mixed frequencies are unsupported: {frequency} and {episode_frequency}")
            episode_cameras = {
                name: tuple(_image(dataset, 0).shape[:2])
                for name, dataset in _camera_datasets(handle).items()
            }
            if not episode_cameras:
                raise ValueError(f"{episode.path}: no RGB cameras found")
            if cameras is None:
                cameras = episode_cameras
            elif cameras != episode_cameras:
                raise ValueError(f"{episode.path}: camera names or resolutions differ from earlier episodes")
    assert frequency is not None and cameras is not None
    return frequency, cameras


def _features(cameras: dict[str, tuple[int, int]]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": [STATE_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": [STATE_NAMES],
        },
    }
    for camera, (height, width) in cameras.items():
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }
    return features


def _default_factory(**kwargs: Any) -> Any:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is required for export; install lerobot>=0.4,<0.5 in the conversion environment"
        ) from exc
    supported = inspect.signature(LeRobotDataset.create).parameters
    return LeRobotDataset.create(**{key: value for key, value in kwargs.items() if key in supported})


def _source_signature(episodes: Iterable[EpisodeInfo]) -> list[dict[str, int | str]]:
    return [
        {
            "episode_id": episode.episode_id,
            "file": episode.path.name,
            "frames": episode.frames,
            "size": episode.path.stat().st_size,
        }
        for episode in episodes
    ]


def export_lerobot(
    dataset_dir: str | Path,
    output: str | Path,
    *,
    profile: str | None = None,
    repo_id: str = "robodojosim/bottle-task",
    overwrite: bool = False,
    dataset_factory: Callable[..., Any] | None = None,
) -> Path:
    """Convert complete HDF5 episodes into a local LeRobotDataset v3 tree."""

    dataset_dir = Path(dataset_dir).resolve()
    output = Path(output).resolve()
    episodes = _eligible_episodes(dataset_dir, profile)
    if not episodes:
        raise ValueError("dataset contains no eligible complete episodes")
    signature = _source_signature(episodes)
    manifest_path = output / "robodojosim_export.json"
    if output.exists() and not overwrite:
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if existing and existing.get("source_episodes") == signature and existing.get("profile") == profile:
            return output
        raise FileExistsError(f"output already exists and is not current: {output}; pass --overwrite")

    frequency, cameras = _inspect_source(episodes)
    partial = output.with_name(f".{output.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.parent.mkdir(parents=True, exist_ok=True)
    factory = dataset_factory or _default_factory
    dataset = factory(
        repo_id=repo_id,
        root=partial,
        fps=frequency,
        robot_type="dual_arx_x5_cartesian",
        features=_features(cameras),
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
        streaming_encoding=False,
    )
    try:
        for episode in episodes:
            with h5py.File(episode.path, "r") as handle:
                instruction = _read_text(handle["instruction"])
                length = int(handle.attrs["length"])
                camera_data = _camera_datasets(handle)
                for index in range(length):
                    frame: dict[str, Any] = {
                        "observation.state": _vector(handle, "state", index),
                        "action": _vector(handle, "action", index),
                        "task": instruction,
                    }
                    for camera, images in camera_data.items():
                        frame[f"observation.images.{camera}"] = _image(images, index)
                    dataset.add_frame(frame)
                dataset.save_episode()
        if hasattr(dataset, "finalize"):
            dataset.finalize()
        elif hasattr(dataset, "stop_image_writer"):
            dataset.stop_image_writer()
    except BaseException:
        if hasattr(dataset, "clear_episode_buffer"):
            dataset.clear_episode_buffer()
        raise

    manifest = {
        "format_version": 1,
        "lerobot_format": "v3.0",
        "repo_id": repo_id,
        "profile": profile,
        "fps": frequency,
        "state_representation": "left_xyz_qwxyz_gripper_right_xyz_qwxyz_gripper",
        "source_dataset": str(dataset_dir),
        "source_episodes": signature,
    }
    (partial / "robodojosim_export.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if output.exists():
        shutil.rmtree(output)
    os.replace(partial, output)
    return output
