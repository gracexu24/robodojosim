from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py


@dataclass(frozen=True)
class EpisodeInfo:
    path: Path
    episode_id: int
    layout_seed: int
    frames: int
    frequency: float
    success: bool
    complete: bool
    profile: str
    trajectory_variant: int

    @property
    def seconds(self) -> float:
        return self.frames / self.frequency if self.frequency > 0 else 0.0


@dataclass(frozen=True)
class DatasetStats:
    complete_episodes: int
    complete_frames: int
    complete_seconds: float
    complete_hours: float
    mean_complete_seconds: float
    successful_episodes: int
    failed_episodes: int
    successful_frames: int
    successful_seconds: float
    successful_hours: float
    mean_successful_seconds: float
    corrupt_or_partial_files: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def read_episode(path: str | Path) -> EpisodeInfo:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        metadata: dict[str, Any] = {}
        if "metadata_json" in handle:
            try:
                metadata = json.loads(_decode_scalar(handle["metadata_json"][()]))
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        frequency = 0.0
        if "additional_info/frequency" in handle:
            frequency = float(handle["additional_info/frequency"][()])
        episode_id = int(handle.attrs.get("episode_id", -1))
        layout_seed = int(metadata.get("layout_seed", handle.attrs.get("seed", -1)))
        return EpisodeInfo(
            path=path,
            episode_id=episode_id,
            layout_seed=layout_seed,
            frames=int(handle.attrs.get("length", 0)),
            frequency=frequency,
            success=bool(handle.attrs.get("success", False)),
            complete=bool(handle.attrs.get("complete", False)),
            profile=str(metadata.get("collection_profile", "unknown")),
            trajectory_variant=int(metadata.get("trajectory_variant", 0)),
        )


def list_episodes(dataset_dir: str | Path, *, profile: str | None = None) -> tuple[list[EpisodeInfo], int]:
    records: list[EpisodeInfo] = []
    invalid = 0
    for path in sorted(Path(dataset_dir).glob("episode_*.hdf5")):
        try:
            record = read_episode(path)
        except (OSError, ValueError, TypeError):
            invalid += 1
            continue
        if profile is None or record.profile == profile:
            records.append(record)
    invalid += len(list(Path(dataset_dir).glob(".*.partial.hdf5")))
    return records, invalid


def summarize_dataset(dataset_dir: str | Path, *, profile: str | None = None) -> DatasetStats:
    records, invalid = list_episodes(dataset_dir, profile=profile)
    complete = [record for record in records if record.complete]
    successful = [record for record in complete if record.success]
    complete_frames = sum(record.frames for record in complete)
    complete_seconds = sum(record.seconds for record in complete)
    frames = sum(record.frames for record in successful)
    seconds = sum(record.seconds for record in successful)
    return DatasetStats(
        complete_episodes=len(complete),
        complete_frames=complete_frames,
        complete_seconds=complete_seconds,
        complete_hours=complete_seconds / 3600.0,
        mean_complete_seconds=complete_seconds / len(complete) if complete else 0.0,
        successful_episodes=len(successful),
        failed_episodes=sum(not record.success for record in complete),
        successful_frames=frames,
        successful_seconds=seconds,
        successful_hours=seconds / 3600.0,
        mean_successful_seconds=seconds / len(successful) if successful else 0.0,
        corrupt_or_partial_files=invalid,
    )


def write_splits(
    dataset_dir: str | Path,
    *,
    output: str | Path | None = None,
    profile: str | None = None,
    seed: int = 0,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> Path:
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("ratios must satisfy train > 0, validation >= 0, and train + validation < 1")
    dataset_dir = Path(dataset_dir)
    records, _ = list_episodes(dataset_dir, profile=profile)
    eligible = [record for record in records if record.complete and (record.success or profile == "world_model")]
    if not eligible:
        raise ValueError("dataset contains no eligible complete episodes")

    by_layout: dict[int, list[EpisodeInfo]] = {}
    for record in eligible:
        by_layout.setdefault(record.layout_seed, []).append(record)
    layouts = sorted(by_layout)
    random.Random(seed).shuffle(layouts)
    count = len(layouts)
    train_count = max(1, int(count * train_ratio))
    validation_count = int(count * validation_ratio)
    if validation_ratio > 0 and count >= 3:
        validation_count = max(1, validation_count)
    if train_count + validation_count >= count:
        train_count = max(1, count - validation_count - 1)

    layout_splits = {
        "train": layouts[:train_count],
        "validation": layouts[train_count : train_count + validation_count],
        "test": layouts[train_count + validation_count :],
    }
    payload: dict[str, Any] = {
        "format_version": 1,
        "grouped_by": "layout_seed",
        "seed": seed,
        "profile": profile,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": 1.0 - train_ratio - validation_ratio,
        },
        "splits": {},
    }
    for split_name, split_layouts in layout_splits.items():
        split_records = [record for layout in split_layouts for record in by_layout[layout]]
        payload["splits"][split_name] = {
            "layout_seeds": split_layouts,
            "episodes": [record.path.name for record in sorted(split_records, key=lambda item: item.episode_id)],
            "episode_count": len(split_records),
            "hours": sum(record.seconds for record in split_records) / 3600.0,
        }

    output_path = Path(output) if output else dataset_dir / "splits.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return output_path
