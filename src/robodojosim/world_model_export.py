from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from .image_codec import decode_jpeg

_VLEN_UINT8 = h5py.vlen_dtype(np.dtype("uint8"))
_DEPTH_KEYS = ("depths", "distance_to_image_planes", "approximate_depths")
_EXTRINSIC_KEYS = ("extrinsic_matrices", "extrinsics_matrices", "extrinsic_matrix", "extrinsics_matrix")
_LABEL_KEYS = (
    "object_labels",
    "object_masks",
    "semantic_segmentations",
    "instance_segmentations",
    "instance_id_segmentations",
)
_USD_FROM_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


@dataclass(frozen=True)
class CompatibilityReport:
    path: str
    compatible: bool
    cameras: tuple[str, ...]
    frames: int
    frequency: float
    action_dim: int | None
    direct: tuple[str, ...]
    converted: tuple[str, ...]
    missing: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorldModelExportConfig:
    bbox: tuple[float, float, float, float, float, float]
    target_fps: float = 10.0
    jpeg_quality: int = 95
    source_extrinsic: str = "camera_to_world"
    source_camera_axes: str = "isaac_usd"
    object_mask_margin: float = 0.005
    require_object_masks: bool = True

    def __post_init__(self) -> None:
        xmin, xmax, ymin, ymax, zmin, zmax = self.bbox
        if not (xmin < xmax and ymin < ymax and zmin < zmax):
            raise ValueError("bbox must be xmin xmax ymin ymax zmin zmax with strictly increasing pairs")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if self.source_extrinsic not in {"camera_to_world", "world_to_camera"}:
            raise ValueError("source_extrinsic must be camera_to_world or world_to_camera")
        if self.source_camera_axes not in {"isaac_usd", "opencv"}:
            raise ValueError("source_camera_axes must be isaac_usd or opencv")


def _camera_names(handle: h5py.File) -> tuple[str, ...]:
    vision = handle.get("vision")
    if not isinstance(vision, h5py.Group):
        return ()
    return tuple(sorted(name for name, value in vision.items() if isinstance(value, h5py.Group)))


def _has_any(group: h5py.Group, keys: tuple[str, ...]) -> bool:
    return any(key in group for key in keys)


def inspect_3dws_episode(path: str | Path) -> CompatibilityReport:
    path = Path(path)
    direct: list[str] = []
    converted: list[str] = []
    missing: list[str] = []
    with h5py.File(path, "r") as handle:
        frames = int(handle.attrs.get("length", 0))
        frequency = float(handle.get("additional_info/frequency", np.asarray(0))[()])
        cameras = _camera_names(handle)
        if cameras:
            direct.append("synchronized multi-camera RGB")
        else:
            missing.append("vision camera groups")
        for camera in cameras:
            group = handle[f"vision/{camera}"]
            for description, present in (
                (f"{camera}: RGB", "colors" in group),
                (f"{camera}: metric depth", _has_any(group, _DEPTH_KEYS)),
                (f"{camera}: intrinsics", "intrinsic_matrix" in group),
                (f"{camera}: per-frame extrinsics", _has_any(group, _EXTRINSIC_KEYS)),
            ):
                (direct if present else missing).append(description)
        action_dim = 14 if all(f"action/{arm}_ee_poses" in handle for arm in ("left", "right")) else None
        if action_dim is None:
            missing.append("bimanual Cartesian pose actions")
        elif all(f"action/{arm}_ee_joint_states" in handle for arm in ("left", "right")):
            converted.append("16D quaternion+gripper commands -> 14D xyz+rpy+gripper Cartesian action")
        else:
            missing.append("bimanual gripper actions")
        has_labels = any(
            _has_any(handle[f"vision/{camera}"], _LABEL_KEYS) for camera in cameras
        ) if cameras else False
        has_boxes = all(f"teacher/bottle{index}_pose" in handle and f"teacher/bottle{index}_bbox" in handle for index in (0, 1))
        if has_labels:
            direct.append("object label images")
        elif has_boxes:
            converted.append("privileged bottle poses/bounds -> per-pixel object labels")
        else:
            missing.append("object masks or teacher bottle poses plus local bounds")
        if all(f"state/{arm}_arm_joint_states" in handle for arm in ("left", "right")):
            converted.append("per-arm joint/gripper state -> 14D robot joint state")
        else:
            missing.append("both six-joint arm state streams")
        converted.extend(
            (
                "Isaac metric depth -> packed_v2 full-scene depth (FoundationStereo bypassed)",
                "aligned RGB/depth -> rgb_in_ir (identity RGB-to-IR transform)",
                "Isaac/USD camera axes -> OpenCV camera-to-world extrinsics",
                "25 Hz or source-rate frames -> timestamp-based target-rate sampling",
            )
        )
    return CompatibilityReport(
        str(path), not missing, cameras, frames, frequency, action_dim, tuple(direct), tuple(converted), tuple(missing)
    )


def _select_indices(handle: h5py.File, target_fps: float) -> tuple[np.ndarray, np.ndarray]:
    length = int(handle.attrs["length"])
    frequency = float(handle["additional_info/frequency"][()])
    if "meta/timestamp" in handle:
        timestamp = np.asarray(handle["meta/timestamp"][:], dtype=np.float64)
    else:
        timestamp = np.arange(length, dtype=np.float64) / frequency
    timestamp = timestamp - timestamp[0]
    if target_fps >= frequency - 1e-8:
        return np.arange(length, dtype=np.int64), timestamp
    targets = np.arange(0.0, timestamp[-1] + 0.5 / target_fps, 1.0 / target_fps)
    right = np.searchsorted(timestamp, targets, side="left")
    right = np.clip(right, 0, length - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(timestamp[left] - targets) <= np.abs(timestamp[right] - targets)
    indices = np.where(choose_left, left, right)
    indices = np.unique(indices.astype(np.int64))
    return indices, timestamp[indices]


def _at(dataset: h5py.Dataset, index: int, source_length: int) -> np.ndarray:
    if dataset.ndim >= 1 and dataset.shape[0] == source_length:
        return np.asarray(dataset[index])
    return np.asarray(dataset[()])


def _dataset(group: h5py.Group, keys: tuple[str, ...]) -> h5py.Dataset:
    for key in keys:
        value = group.get(key)
        if isinstance(value, h5py.Dataset):
            return value
    raise KeyError(f"none of {keys} found under {group.name}")


def _rgb(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    image = decode_jpeg(value) if value.ndim == 1 else value
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"invalid RGB frame shape {image.shape}")
    return np.asarray(image[..., :3], dtype=np.uint8)


def _depth(dataset: h5py.Dataset, index: int, source_length: int) -> np.ndarray:
    value = _at(dataset, index, source_length)
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError(f"{dataset.name}: depth must be HxW, got {value.shape}")
    result = value.astype(np.float32)
    if dataset.name.endswith("approximate_depths") or value.dtype == np.uint16:
        result /= 1000.0
    return result


def _jpeg(image_rgb: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image_rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("OpenCV failed to JPEG-encode a frame")
    return encoded.astype(np.uint8)


def _opencv_cam_to_world(raw: np.ndarray, convention: str, axes: str) -> np.ndarray:
    transform = np.asarray(raw, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"camera extrinsic must be 4x4, got {transform.shape}")
    if convention == "world_to_camera":
        transform = np.linalg.inv(transform)
    if axes == "isaac_usd":
        transform = transform @ _USD_FROM_OPENCV
    return transform.astype(np.float32)


def _world_points(depth: np.ndarray, intrinsic: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    z = depth
    x = (xx - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (yy - intrinsic[1, 2]) * z / intrinsic[1, 1]
    camera = np.stack((x, y, z, np.ones_like(z)), axis=-1)
    return camera @ cam_to_world.T


def _bbox_valid(points: np.ndarray, bbox: tuple[float, ...]) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    return (
        (points[..., 0] >= xmin)
        & (points[..., 0] <= xmax)
        & (points[..., 1] >= ymin)
        & (points[..., 1] <= ymax)
        & (points[..., 2] >= zmin)
        & (points[..., 2] <= zmax)
    )


def _quat_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = np.linalg.norm([w, x, y, z])
    if norm == 0:
        raise ValueError("zero quaternion")
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _object_labels_from_bounds(
    handle: h5py.File,
    source_index: int,
    points_world: np.ndarray,
    valid: np.ndarray,
    margin: float,
) -> np.ndarray:
    labels = np.zeros(valid.shape, dtype=np.uint8)
    for object_index in (0, 1):
        pose = np.asarray(handle[f"teacher/bottle{object_index}_pose"][source_index], dtype=np.float64)
        bbox = np.asarray(handle[f"teacher/bottle{object_index}_bbox"][source_index], dtype=np.float64).reshape(-1)
        if bbox.size != 6:
            raise ValueError(f"teacher/bottle{object_index}_bbox must contain six local AABB values")
        local = (points_world[..., :3] - pose[:3]) @ _quat_rotation(pose[3:]).astype(np.float32)
        inside = np.all(local >= bbox[:3] - margin, axis=-1) & np.all(local <= bbox[3:] + margin, axis=-1)
        labels[inside & valid] = object_index + 1
    return labels


def _canonical_label_image(group: h5py.Group, index: int, source_length: int) -> np.ndarray | None:
    for key in _LABEL_KEYS:
        dataset = group.get(key)
        if not isinstance(dataset, h5py.Dataset):
            continue
        labels = _at(dataset, index, source_length)
        if labels.ndim == 3 and labels.shape[-1] == 1:
            labels = labels[..., 0]
        if labels.ndim == 2:
            return labels.astype(np.uint8)
    return None


def _quat_to_euler(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(np.asarray(quaternion, dtype=np.float64), -1, 0)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1)


def _cartesian_action(handle: h5py.File, indices: np.ndarray) -> np.ndarray:
    arms = []
    for arm in ("left", "right"):
        pose = np.asarray(handle[f"action/{arm}_ee_poses"][indices], dtype=np.float64)
        grip = np.asarray(handle[f"action/{arm}_ee_joint_states"][indices], dtype=np.float64).reshape(len(indices), -1)
        euler = np.unwrap(_quat_to_euler(pose[:, 3:7]), axis=0)
        arms.append(np.concatenate((pose[:, :3], euler, grip[:, :1]), axis=1))
    return np.concatenate(arms, axis=1).astype(np.float32)


def _joint_state(handle: h5py.File, indices: np.ndarray) -> np.ndarray | None:
    arms = []
    for arm in ("left", "right"):
        joint_key = f"state/{arm}_arm_joint_states"
        grip_key = f"state/{arm}_ee_joint_states"
        if joint_key not in handle or grip_key not in handle:
            return None
        joints = np.asarray(handle[joint_key][indices], dtype=np.float32).reshape(len(indices), -1)
        grip = np.asarray(handle[grip_key][indices], dtype=np.float32).reshape(len(indices), -1)
        arms.append(np.concatenate((joints, grip[:, :1]), axis=1))
    return np.concatenate(arms, axis=1)


def _phase_stage(handle: h5py.File, indices: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    if "teacher/skill" not in handle:
        return np.zeros(len(indices), dtype=np.int64), {"unknown": 0}
    raw = handle["teacher/skill"][indices]
    text = [value.decode() if isinstance(value, bytes) else str(value) for value in raw]
    mapping = {name: number for number, name in enumerate(sorted(set(text)))}
    return np.asarray([mapping[name] for name in text], dtype=np.int64), mapping


def export_3dws_episode(
    source: str | Path,
    output: str | Path,
    config: WorldModelExportConfig,
) -> dict[str, Any]:
    source, output = Path(source), Path(output)
    report = inspect_3dws_episode(source)
    blockers = [item for item in report.missing if not item.startswith("object masks")]
    if config.require_object_masks:
        blockers = list(report.missing)
    if blockers:
        raise ValueError(f"{source} is not 3DWS-exportable: " + "; ".join(blockers))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with h5py.File(source, "r") as source_file, h5py.File(temporary, "w") as packed:
        source_length = int(source_file.attrs["length"])
        indices, timestamps = _select_indices(source_file, config.target_fps)
        cameras = _camera_names(source_file)
        camera_ids = tuple(range(len(cameras)))
        action = _cartesian_action(source_file, indices)
        joint_state = _joint_state(source_file, indices)
        stage, stage_labels = _phase_stage(source_file, indices)
        has_object_masks = True
        camera_shapes: dict[str, tuple[int, int]] = {}
        for camera_id, camera_name in zip(camera_ids, cameras):
            camera = source_file[f"vision/{camera_name}"]
            rgb_dataset = camera["colors"]
            depth_dataset = _dataset(camera, _DEPTH_KEYS)
            intrinsic_dataset = camera["intrinsic_matrix"]
            extrinsic_dataset = _dataset(camera, _EXTRINSIC_KEYS)
            first_rgb = _rgb(_at(rgb_dataset, int(indices[0]), source_length))
            height, width = first_rgb.shape[:2]
            camera_shapes[camera_name] = (height, width)
            depth_out = packed.create_dataset(
                f"img/camera_{camera_id}/depth", (len(indices), height, width), dtype=np.float16, compression="gzip"
            )
            rgb_out = packed.create_dataset(f"img/camera_{camera_id}/rgb_in_ir", (len(indices),), dtype=_VLEN_UINT8)
            packed_width = (height * width + 7) // 8
            valid_out = packed.create_dataset(
                f"img/camera_{camera_id}/valid_mask", (len(indices), packed_width), dtype=np.uint8, compression="gzip"
            )
            object_out = packed.create_dataset(
                f"img/camera_{camera_id}/obj_mask",
                (len(indices), height, width),
                dtype=np.uint8,
                compression="gzip",
            )
            intrinsic_out = packed.create_dataset(f"cam/camera_{camera_id}/K_ir", (len(indices), 3, 3), dtype=np.float32)
            extrinsic_out = packed.create_dataset(
                f"cam/camera_{camera_id}/E_cam2world", (len(indices), 4, 4), dtype=np.float32
            )
            render_jpg = packed.create_dataset(f"render/camera_{camera_id}/jpg", (len(indices),), dtype=_VLEN_UINT8)
            render_k = packed.create_dataset(f"render/camera_{camera_id}/K", (len(indices), 3, 3), dtype=np.float32)
            render_view = packed.create_dataset(
                f"render/camera_{camera_id}/viewmat", (len(indices), 4, 4), dtype=np.float32
            )
            render_obj = packed.create_dataset(
                f"render/camera_{camera_id}/obj_mask", (len(indices), packed_width), dtype=np.uint8, compression="gzip"
            )
            render_valid = packed.create_dataset(
                f"render/camera_{camera_id}/valid_mask", (len(indices), packed_width), dtype=np.uint8, compression="gzip"
            )
            for output_index, source_index in enumerate(indices):
                rgb = _rgb(_at(rgb_dataset, int(source_index), source_length))
                depth = _depth(depth_dataset, int(source_index), source_length)
                if rgb.shape[:2] != depth.shape:
                    raise ValueError(f"{camera.name}: RGB {rgb.shape[:2]} and depth {depth.shape} are not aligned")
                intrinsic = _at(intrinsic_dataset, int(source_index), source_length).astype(np.float32)
                extrinsic = _opencv_cam_to_world(
                    _at(extrinsic_dataset, int(source_index), source_length),
                    config.source_extrinsic,
                    config.source_camera_axes,
                )
                points = _world_points(depth, intrinsic, extrinsic)
                valid_depth = np.isfinite(depth) & (depth > 0)
                valid = valid_depth & _bbox_valid(points, config.bbox)
                labels = _canonical_label_image(camera, int(source_index), source_length)
                if labels is None and all(
                    f"teacher/bottle{object_index}_bbox" in source_file for object_index in (0, 1)
                ):
                    labels = _object_labels_from_bounds(
                        source_file, int(source_index), points, valid_depth, config.object_mask_margin
                    )
                if labels is None:
                    labels = np.zeros(depth.shape, dtype=np.uint8)
                    has_object_masks = False
                encoded = _jpeg(rgb, config.jpeg_quality)
                stored_depth = np.where(valid_depth, depth, 0).astype(np.float16)
                depth_out[output_index] = stored_depth
                rgb_out[output_index] = encoded
                valid_out[output_index] = np.packbits(valid.reshape(-1))
                object_out[output_index] = labels
                intrinsic_out[output_index] = intrinsic
                extrinsic_out[output_index] = extrinsic
                render_jpg[output_index] = encoded
                render_k[output_index] = intrinsic
                render_view[output_index] = np.linalg.inv(extrinsic).astype(np.float32)
                render_obj[output_index] = np.packbits((labels > 0).reshape(-1))
                render_valid[output_index] = np.packbits(valid_depth.reshape(-1))

        packed.create_dataset("action/cartesian_action", data=action)
        if joint_state is not None:
            packed.create_dataset("robot/joint_pos", data=joint_state)
            packed.create_dataset("robot/full_joint_pos", data=joint_state)
        packed.create_dataset("meta/stage", data=stage)
        packed.create_dataset("meta/timestamp", data=timestamps.astype(np.float64))
        packed.create_dataset("meta/source_frame_index", data=indices.astype(np.int64))
        for object_index in (0, 1):
            for suffix in ("pose", "bbox"):
                key = f"teacher/bottle{object_index}_{suffix}"
                if key in source_file:
                    packed.create_dataset(key, data=source_file[key][indices])
        height, width = next(iter(camera_shapes.values()))
        packed.attrs.update(
            dataset="robodojosim_two_bottle",
            episode=int(source_file.attrs.get("episode_id", -1)),
            num_frames=len(indices),
            H=height,
            W=width,
            cam_ids=np.asarray(camera_ids, dtype=np.int32),
            camera_names=json.dumps({str(index): name for index, name in zip(camera_ids, cameras)}),
            camera_shapes=json.dumps({name: list(shape) for name, shape in camera_shapes.items()}),
            n_arms=2,
            bbox=np.asarray(config.bbox, dtype=np.float32),
            robot_base="{}",
            has_obj_mask=bool(has_object_masks),
            obj_mask_kind="label",
            obj_labels=json.dumps({"1": "brown_cola_bottle", "2": "yellow_tea_bottle"}),
            stage_labels=json.dumps(stage_labels, sort_keys=True),
            action_layout=json.dumps(
                [
                    "left_x", "left_y", "left_z", "left_roll", "left_pitch", "left_yaw", "left_gripper",
                    "right_x", "right_y", "right_z", "right_roll", "right_pitch", "right_yaw", "right_gripper",
                ]
            ),
            action_rotation="static_xyz_euler_radians_unwrapped",
            source_action_layout="per-arm xyz+quaternion_wxyz+gripper",
            source_frequency=float(source_file["additional_info/frequency"][()]),
            frequency=float(config.target_fps),
            # Upstream 3DWS uses this exact contract marker before initializing
            # its static background bank.  Isaac Sim supplies full-scene metric
            # ground-truth depth, so it satisfies the geometric contract without
            # requiring the stereo left-right-consistency filtering that produced
            # the original real-camera packs.
            depth_semantics="full_scene_lrc",
            depth_source="isaac_sim_ground_truth",
            depth_semantics_detail="metric distance-to-image-plane; 0=invalid; simulator ground truth",
            rgb_depth_alignment="same simulated pinhole camera; T_rgb_to_ir=identity",
            extrinsic_semantics="OpenCV camera_to_world (+x right,+y down,+z forward)",
            mask_semantics="valid=finite_positive_depth&world_bbox; obj=privileged local-AABB labels",
            src_episode=source.name,
            format_version="packed_v2",
        )
    os.replace(temporary, output)
    errors = validate_3dws_packed(output)
    if errors:
        raise ValueError(f"exported file failed validation: {'; '.join(errors)}")
    return {
        "source": str(source),
        "output": str(output),
        "source_frames": report.frames,
        "exported_frames": len(indices),
        "cameras": list(cameras),
        "action_dim": int(action.shape[1]),
        "has_object_masks": bool(has_object_masks),
    }


def _split_lookup(dataset: Path) -> dict[str, str]:
    split_path = dataset / "splits.json"
    if not split_path.exists():
        return {}
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    return {
        episode: split_name
        for split_name, split in payload.get("splits", {}).items()
        for episode in split.get("episodes", [])
    }


def export_3dws_dataset(
    dataset: str | Path,
    output: str | Path,
    config: WorldModelExportConfig,
) -> Path:
    dataset, output = Path(dataset), Path(output)
    episodes = sorted(dataset.glob("episode_*.hdf5"))
    if not episodes:
        raise ValueError(f"no episode_*.hdf5 files under {dataset}")
    lookup = _split_lookup(dataset)
    results = []
    for source in episodes:
        with h5py.File(source, "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                continue
            episode_id = int(handle.attrs.get("episode_id", len(results)))
        split = lookup.get(source.name, "train")
        destination = output / split / f"episode_{episode_id}.hdf5"
        results.append(export_3dws_episode(source, destination, config))
    if not results:
        raise ValueError("dataset contains no complete episodes")
    manifest = {
        "format_version": "3dws_packed_v2",
        "source": str(dataset),
        "config": asdict(config),
        "episodes": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    _manifest = output / "export_manifest.json"
    temporary = _manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, _manifest)
    return output


def validate_3dws_packed(path: str | Path) -> list[str]:
    errors: list[str] = []
    try:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("format_version", "") != "packed_v2":
                errors.append("format_version is not packed_v2")
            frames = int(handle.attrs.get("num_frames", 0))
            camera_ids = [int(value) for value in handle.attrs.get("cam_ids", [])]
            if frames <= 0:
                errors.append("num_frames must be positive")
            for key in ("action/cartesian_action", "meta/timestamp", "meta/stage"):
                if key not in handle:
                    errors.append(f"missing /{key}")
                elif handle[key].shape[0] != frames:
                    errors.append(f"/{key} has the wrong time dimension")
            for camera_id in camera_ids:
                for key in (
                    f"img/camera_{camera_id}/depth",
                    f"img/camera_{camera_id}/rgb_in_ir",
                    f"img/camera_{camera_id}/valid_mask",
                    f"cam/camera_{camera_id}/K_ir",
                    f"cam/camera_{camera_id}/E_cam2world",
                    f"render/camera_{camera_id}/jpg",
                    f"render/camera_{camera_id}/K",
                    f"render/camera_{camera_id}/viewmat",
                ):
                    if key not in handle:
                        errors.append(f"missing /{key}")
                    elif handle[key].shape[0] != frames:
                        errors.append(f"/{key} has the wrong time dimension")
    except OSError as exc:
        errors.append(f"cannot open HDF5: {exc}")
    return errors
