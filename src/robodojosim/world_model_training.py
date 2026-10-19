from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

_LABEL = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class PackedTrainingRoot:
    label: str
    path: Path

    def __post_init__(self) -> None:
        if not _LABEL.fullmatch(self.label):
            raise ValueError(f"invalid root label {self.label!r}; use lowercase letters, numbers, '-' or '_'")


def parse_training_root(value: str) -> PackedTrainingRoot:
    label, separator, path = value.partition("=")
    if not separator or not path:
        raise ValueError("packed roots must use LABEL=/absolute/path syntax")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_absolute():
        raise ValueError("packed root paths must be absolute")
    return PackedTrainingRoot(label=label, path=resolved)


def _episode_number(path: Path) -> int:
    try:
        return int(path.stem.split("_")[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid packed episode filename: {path.name}") from exc


def _packed_metadata(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("format_version", "") != "packed_v2":
            raise ValueError(f"{path} is not a packed_v2 episode")
        if "action/cartesian_action" not in handle:
            raise ValueError(f"{path} is missing /action/cartesian_action")
        action = handle["action/cartesian_action"]
        if action.ndim != 2 or action.shape[1] != 14:
            raise ValueError(f"{path} has action shape {action.shape}; expected (T, 14)")
        frames = int(handle.attrs.get("num_frames", action.shape[0]))
        if frames < 1 or action.shape[0] != frames:
            raise ValueError(f"{path} has inconsistent frame counts")
        bbox = np.asarray(handle.attrs.get("bbox", ()), dtype=np.float32)
        if bbox.shape != (6,) or not np.isfinite(bbox).all():
            raise ValueError(f"{path} has no finite six-value bbox")
        episode = int(handle.attrs.get("episode", _episode_number(path)))
        frequency = float(handle.attrs.get("frequency", 0.0))
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError(f"{path} has an invalid frequency")
        return {
            "original_episode": episode,
            "frames": frames,
            "frequency": frequency,
            "bbox": [float(value) for value in bbox],
            "camera_ids": [int(value) for value in handle.attrs.get("cam_ids", ())],
        }


def _relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source, destination.parent)
    if destination.is_symlink():
        if os.readlink(destination) == target:
            return
        raise FileExistsError(f"existing symlink points elsewhere: {destination}")
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing path: {destination}")
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    os.replace(temporary, destination)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stage_3dws_training(
    roots: Iterable[PackedTrainingRoot],
    output: str | Path,
) -> Path:
    roots = tuple(roots)
    if not roots:
        raise ValueError("at least one packed training root is required")
    labels = [root.label for root in roots]
    if len(labels) != len(set(labels)):
        raise ValueError("packed training root labels must be unique")

    output = Path(output).expanduser().resolve()
    manifest_path = output / "training_manifest.json"
    if output.exists() and not manifest_path.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to use non-empty unowned staging directory: {output}")

    records: list[dict[str, Any]] = []
    global_episode = 0
    reference_bbox: list[float] | None = None
    for root in roots:
        if not root.path.is_dir():
            raise FileNotFoundError(f"packed root does not exist: {root.path}")
        for split in ("train", "val"):
            split_dir = root.path / split
            episodes = sorted(split_dir.glob("episode_*.hdf5"), key=_episode_number)
            if not episodes:
                raise FileNotFoundError(f"no packed episodes under {split_dir}")
            for source in episodes:
                metadata = _packed_metadata(source)
                if reference_bbox is None:
                    reference_bbox = metadata["bbox"]
                elif not np.allclose(reference_bbox, metadata["bbox"], atol=1e-6, rtol=0):
                    raise ValueError(f"bbox mismatch in {source}: {metadata['bbox']} != {reference_bbox}")
                staged_name = f"episode_{global_episode:06d}.hdf5"
                destination = output / "packed" / split / staged_name
                _relative_symlink(source.resolve(), destination)
                records.append(
                    {
                        "global_episode": global_episode,
                        "label": root.label,
                        "root": str(root.path),
                        "split": split,
                        "source": str(source.resolve()),
                        "staged": str(destination),
                        **metadata,
                    }
                )
                global_episode += 1

    payload = {
        "format_version": 1,
        "kind": "robodojosim_3dws_bottle_training_stage",
        "bbox": reference_bbox,
        "roots": [{"label": root.label, "path": str(root.path)} for root in roots],
        "episodes": records,
        "counts": {
            "total": len(records),
            "train": sum(record["split"] == "train" for record in records),
            "val": sum(record["split"] == "val" for record in records),
            "frames": sum(int(record["frames"]) for record in records),
        },
    }
    _write_json_atomic(manifest_path, payload)
    return manifest_path


def stage_3dws_latents(
    manifest: str | Path,
    source_root: str | Path,
    output: str | Path,
) -> Path:
    manifest = Path(manifest).expanduser().resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("kind") != "robodojosim_3dws_bottle_training_stage":
        raise ValueError(f"not a RoboDojoSim 3DWS training manifest: {manifest}")
    source_root = Path(source_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    staged: list[dict[str, Any]] = []
    for record in payload.get("episodes", []):
        source = (
            source_root / record["label"] / record["split"] / f"episode_{int(record['original_episode'])}_latents.h5"
        )
        if not source.is_file():
            raise FileNotFoundError(f"missing latent export: {source}")
        destination = output / record["split"] / f"episode_{int(record['global_episode']):06d}_latents.h5"
        _relative_symlink(source.resolve(), destination)
        staged.append(
            {
                "global_episode": int(record["global_episode"]),
                "source": str(source),
                "staged": str(destination),
                "split": record["split"],
            }
        )
    result = {
        "format_version": 1,
        "training_manifest": str(manifest),
        "source_root": str(source_root),
        "episodes": staged,
        "counts": {
            "total": len(staged),
            "train": sum(record["split"] == "train" for record in staged),
            "val": sum(record["split"] == "val" for record in staged),
        },
    }
    result_path = output / "latent_manifest.json"
    _write_json_atomic(result_path, result)
    return result_path
