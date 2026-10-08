from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import cv2
import h5py
import numpy as np

from .image_codec import decode_jpeg, encode_xpolicylab_jpeg

_PLURAL_KEYS = {
    "left_arm_joint_state": "left_arm_joint_states",
    "left_ee_joint_state": "left_ee_joint_states",
    "left_ee_pose": "left_ee_poses",
    "left_tcp_pose": "left_tcp_poses",
    "left_delta_ee_pose": "left_delta_ee_poses",
    "right_arm_joint_state": "right_arm_joint_states",
    "right_ee_joint_state": "right_ee_joint_states",
    "right_ee_pose": "right_ee_poses",
    "right_tcp_pose": "right_tcp_poses",
    "right_delta_ee_pose": "right_delta_ee_poses",
    "arm_joint_state": "arm_joint_states",
    "ee_joint_state": "ee_joint_states",
    "ee_pose": "ee_poses",
    "tcp_pose": "tcp_poses",
    "delta_ee_pose": "delta_ee_poses",
    "base_pose": "base_poses",
    "base_twist": "base_twists",
    "color": "colors",
    "depth": "depths",
}


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class EpisodeRecorder:
    """Streaming, atomic writer for XPolicyLab trajectory HDF5 files."""

    def __init__(
        self,
        output_dir: str | Path,
        episode_id: int,
        *,
        seed: int,
        instruction: str,
        frequency: int = 25,
        metadata: Mapping[str, Any] | None = None,
        jpeg_quality: int | None = None,
        flush_interval: int = 25,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_id = int(episode_id)
        self.seed = int(seed)
        if jpeg_quality is not None and not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.jpeg_quality = jpeg_quality
        if flush_interval < 1:
            raise ValueError("flush_interval must be positive")
        self.flush_interval = int(flush_interval)
        stem = f"episode_{self.episode_id:06d}"
        self.final_path = self.output_dir / f"{stem}.hdf5"
        self.partial_path = self.output_dir / f".{stem}.partial.hdf5"
        self._file = h5py.File(self.partial_path, "w")
        self._file.attrs.update(
            {
                "complete": False,
                "episode_id": self.episode_id,
                "seed": self.seed,
                "created_unix": time.time(),
                "generator": "robodojosim",
                "image_encoding": "xpolicylab_jpeg" if jpeg_quality is not None else "raw",
            }
        )
        self._file.create_dataset("data_format_version", data="v1.0")
        self._file.create_dataset("instruction", data=instruction)
        self._file.require_group("additional_info").create_dataset("frequency", data=int(frequency))
        if metadata:
            self._file.create_dataset("metadata_json", data=json.dumps(dict(metadata), sort_keys=True))
        self._length = 0
        self._closed = False

    @classmethod
    def is_complete(cls, output_dir: str | Path, episode_id: int, *, require_success: bool = False) -> bool:
        path = Path(output_dir) / f"episode_{int(episode_id):06d}.hdf5"
        if not path.exists():
            return False
        try:
            with h5py.File(path, "r") as handle:
                complete = bool(handle.attrs.get("complete", False))
                return complete and (not require_success or bool(handle.attrs.get("success", False)))
        except OSError:
            return False

    @property
    def length(self) -> int:
        return self._length

    def append(
        self,
        observation: Mapping[str, Any],
        action: Mapping[str, Any],
        *,
        teacher: Mapping[str, Any] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed recorder")
        state = observation.get("state")
        if not isinstance(state, Mapping):
            raise TypeError("observation must contain a state mapping")
        self._append_mapping("state", state)
        self._append_mapping("action", action)
        vision = observation.get("vision", {})
        if isinstance(vision, Mapping):
            for camera, camera_data in vision.items():
                if not isinstance(camera_data, Mapping):
                    continue
                for key, value in camera_data.items():
                    # Camera calibration and shape are episode constants.
                    if key in {"intrinsic_matrix", "extrinsic_matrix", "shape"}:
                        self._write_once(f"vision/{camera}/{key}", value)
                    elif key in {"color", "colors"} and self.jpeg_quality is not None:
                        self._append_encoded_image(f"vision/{camera}/{_PLURAL_KEYS.get(key, key)}", value)
                    else:
                        self._append_value(f"vision/{camera}/{_PLURAL_KEYS.get(key, key)}", value)
        if teacher:
            self._append_mapping("teacher", teacher, pluralize=False)
        self._length += 1
        if self._length % self.flush_interval == 0:
            self._file.flush()

    def close(self, *, success: bool, score: float | None = None, reason: str = "completed") -> Path:
        if self._closed:
            return self.final_path
        self._file.attrs["length"] = self._length
        self._file.attrs["success"] = bool(success)
        self._file.attrs["termination_reason"] = reason
        if score is not None:
            self._file.attrs["score"] = float(score)
        self._file.attrs["complete"] = True
        self._file.flush()
        self._file.close()
        os.replace(self.partial_path, self.final_path)
        self._closed = True
        self._update_manifest(success=success, score=score, reason=reason)
        return self.final_path

    def abort(self, reason: str) -> None:
        if self._closed:
            return
        self._file.attrs["abort_reason"] = reason
        self._file.flush()
        self._file.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._closed:
            self.abort("exception" if exc_value is not None else "not_finalized")

    def _append_mapping(self, prefix: str, mapping: Mapping[str, Any], *, pluralize: bool = True) -> None:
        for key, value in mapping.items():
            output_key = _PLURAL_KEYS.get(key, key) if pluralize else key
            path = f"{prefix}/{output_key}"
            if isinstance(value, Mapping):
                self._append_mapping(path, value, pluralize=pluralize)
            else:
                self._append_value(path, value)

    def _append_value(self, path: str, value: Any) -> None:
        group_path, name = path.rsplit("/", 1)
        group = self._file.require_group(group_path)
        if isinstance(value, (str, bytes)) or value is None:
            dtype = h5py.string_dtype("utf-8")
            text = (
                ""
                if value is None
                else (value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value)
            )
            if name not in group:
                dataset = group.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=True)
            else:
                dataset = group[name]
            dataset.resize(dataset.shape[0] + 1, axis=0)
            dataset[-1] = text
            return
        array = _as_numpy(value)
        if array.dtype.kind in {"O", "U", "S"}:
            return self._append_value(path, json.dumps(array.tolist()))
        if name not in group:
            compression = "gzip" if array.ndim >= 2 else None
            dataset = group.create_dataset(
                name,
                shape=(0, *array.shape),
                maxshape=(None, *array.shape),
                dtype=array.dtype,
                chunks=(1, *array.shape) if array.shape else True,
                compression=compression,
                compression_opts=1 if compression else None,
            )
        else:
            dataset = group[name]
            if dataset.shape[1:] != array.shape:
                raise ValueError(f"shape changed for {path}: {dataset.shape[1:]} -> {array.shape}")
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = array

    def _write_once(self, path: str, value: Any) -> None:
        group_path, name = path.rsplit("/", 1)
        group = self._file.require_group(group_path)
        if name not in group:
            group.create_dataset(name, data=_as_numpy(value))

    def _append_encoded_image(self, path: str, value: Any) -> None:
        group_path, name = path.rsplit("/", 1)
        group = self._file.require_group(group_path)
        encoded = encode_xpolicylab_jpeg(_as_numpy(value), quality=self.jpeg_quality or 90)
        if name not in group:
            dataset = group.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.vlen_dtype(np.dtype("uint8")),
                chunks=True,
            )
        else:
            dataset = group[name]
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = encoded

    def _update_manifest(self, *, success: bool, score: float | None, reason: str) -> None:
        path = self.output_dir / "manifest.json"
        records: dict[str, Any] = {"format_version": 1, "episodes": []}
        if path.exists():
            try:
                records = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        episodes = [entry for entry in records.get("episodes", []) if entry.get("episode_id") != self.episode_id]
        episodes.append(
            {
                "episode_id": self.episode_id,
                "seed": self.seed,
                "file": self.final_path.name,
                "length": self._length,
                "success": bool(success),
                "score": score,
                "termination_reason": reason,
            }
        )
        records["episodes"] = sorted(episodes, key=lambda entry: entry["episode_id"])
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)


def validate_episode(path: str | Path) -> list[str]:
    """Return validation errors; an empty list means the file is usable."""

    errors: list[str] = []
    try:
        with h5py.File(path, "r") as handle:
            for key in ("data_format_version", "instruction", "additional_info", "state", "action"):
                if key not in handle:
                    errors.append(f"missing /{key}")
            if not bool(handle.attrs.get("complete", False)):
                errors.append("file is not marked complete")
            state_datasets = _datasets_below(handle.get("state"))
            action_datasets = _datasets_below(handle.get("action"))
            state_lengths = [dataset.shape[0] for _, dataset in state_datasets]
            action_lengths = [dataset.shape[0] for _, dataset in action_datasets]
            if not state_lengths:
                errors.append("state has no time-series datasets")
            if not action_lengths:
                errors.append("action has no time-series datasets")
            lengths = state_lengths + action_lengths
            if lengths and len(set(lengths)) != 1:
                errors.append(f"time-series lengths do not match: {sorted(set(lengths))}")
            expected = int(handle.attrs.get("length", -1))
            if lengths and expected != lengths[0]:
                errors.append(f"length attribute is {expected}, datasets have {lengths[0]}")
            if expected <= 0:
                errors.append(f"invalid recorded length: {expected}")
            if "success" not in handle.attrs:
                errors.append("missing success attribute")
            frequency = handle.get("additional_info/frequency")
            if frequency is not None and int(frequency[()]) <= 0:
                errors.append("recording frequency must be positive")

            timed = [(f"state/{name}", dataset) for name, dataset in state_datasets]
            timed += [(f"action/{name}", dataset) for name, dataset in action_datasets]
            timed += [(f"teacher/{name}", dataset) for name, dataset in _datasets_below(handle.get("teacher"))]
            for camera_name, camera in _groups_below(handle.get("vision")):
                for key in ("colors", "depths"):
                    if key in camera and isinstance(camera[key], h5py.Dataset):
                        timed.append((f"vision/{camera_name}/{key}", camera[key]))
            for name, dataset in timed:
                if dataset.shape and expected >= 0 and dataset.shape[0] != expected:
                    errors.append(f"/{name} has {dataset.shape[0]} frames, expected {expected}")
                if dataset.dtype.kind in {"f", "c"} and not _all_finite(dataset):
                    errors.append(f"/{name} contains NaN or Inf")

            if handle.attrs.get("image_encoding", "raw") == "xpolicylab_jpeg":
                for camera_name, camera in _groups_below(handle.get("vision")):
                    colors = camera.get("colors")
                    if not isinstance(colors, h5py.Dataset) or not len(colors):
                        continue
                    sample_indices = sorted({0, len(colors) // 2, len(colors) - 1})
                    for index in sample_indices:
                        try:
                            frame = decode_jpeg(colors[index])
                            if frame.ndim != 3 or frame.shape[2] != 3:
                                errors.append(
                                    f"/vision/{camera_name}/colors[{index}] decoded to invalid shape {frame.shape}"
                                )
                        except (cv2.error, TypeError, ValueError) as exc:
                            errors.append(f"/vision/{camera_name}/colors[{index}] is not decodable: {exc}")
    except OSError as exc:
        errors.append(f"cannot open HDF5: {exc}")
    return errors


def _datasets_below(group: h5py.Group | None) -> list[tuple[str, h5py.Dataset]]:
    if group is None:
        return []
    result: list[tuple[str, h5py.Dataset]] = []

    def collect(name: str, item: h5py.Group | h5py.Dataset) -> None:
        if isinstance(item, h5py.Dataset):
            result.append((name, item))

    group.visititems(collect)
    return result


def _groups_below(group: h5py.Group | None) -> list[tuple[str, h5py.Group]]:
    if group is None:
        return []
    return [(name, item) for name, item in group.items() if isinstance(item, h5py.Group)]


def _all_finite(dataset: h5py.Dataset, chunk_size: int = 1024) -> bool:
    if not dataset.shape:
        return bool(np.all(np.isfinite(dataset[()])))
    for start in range(0, dataset.shape[0], chunk_size):
        if not np.all(np.isfinite(dataset[start : start + chunk_size])):
            return False
    return True
