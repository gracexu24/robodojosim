from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from .image_codec import decode_jpeg

_LABEL_COLORS = {
    1: np.asarray((70, 110, 190), dtype=np.uint8),  # brown bottle, RGB
    2: np.asarray((245, 215, 45), dtype=np.uint8),  # yellow bottle, RGB
}


def _camera_ids(handle: h5py.File) -> list[int]:
    return [int(value) for value in handle.attrs.get("cam_ids", [])]


def _camera_names(handle: h5py.File) -> dict[int, str]:
    raw = handle.attrs.get("camera_names", "{}")
    if isinstance(raw, bytes):
        raw = raw.decode()
    parsed = json.loads(str(raw))
    return {int(key): str(value) for key, value in parsed.items()}


def _decode_rgb(dataset: h5py.Dataset, index: int) -> np.ndarray:
    value = np.asarray(dataset[index])
    image = decode_jpeg(value) if value.ndim == 1 else value
    return np.asarray(image[..., :3], dtype=np.uint8)


def _unpack_mask(value: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.unpackbits(np.asarray(value, dtype=np.uint8), count=height * width).reshape(height, width).astype(bool)


def analyze_3dws_quality(path: str | Path, *, maximum_samples: int = 120) -> dict[str, Any]:
    """Run numerical gates that catch unusable depth, calibration, masks, or timing."""

    path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    cameras: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        frames = int(handle.attrs.get("num_frames", 0))
        if frames <= 0:
            return {"path": str(path), "passed": False, "errors": ["episode has no frames"], "warnings": []}
        indices = np.unique(np.linspace(0, frames - 1, min(frames, maximum_samples), dtype=np.int64))
        timestamps = np.asarray(handle["meta/timestamp"][:], dtype=np.float64)
        deltas = np.diff(timestamps)
        if not np.all(np.isfinite(timestamps)) or (len(deltas) and not np.all(deltas > 0)):
            errors.append("timestamps are not finite and strictly increasing")
        action = np.asarray(handle["action/cartesian_action"][:], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != 14 or not np.all(np.isfinite(action)):
            errors.append("Cartesian action must be finite T x 14")

        names = _camera_names(handle)
        camera_ids = _camera_ids(handle)
        if len(camera_ids) != 3:
            errors.append(f"expected three cameras, found {len(camera_ids)}")
        object_totals = {1: 0, 2: 0}
        for camera_id in camera_ids:
            name = names.get(camera_id, f"camera_{camera_id}")
            depth = np.asarray(handle[f"img/camera_{camera_id}/depth"][indices], dtype=np.float32)
            labels = np.asarray(handle[f"img/camera_{camera_id}/obj_mask"][indices], dtype=np.uint8)
            intrinsic = np.asarray(handle[f"cam/camera_{camera_id}/K_ir"][indices], dtype=np.float64)
            extrinsic = np.asarray(handle[f"cam/camera_{camera_id}/E_cam2world"][indices], dtype=np.float64)
            height, width = depth.shape[1:]
            valid_bbox = np.stack(
                [_unpack_mask(handle[f"img/camera_{camera_id}/valid_mask"][int(index)], height, width) for index in indices]
            )
            positive = np.isfinite(depth) & (depth > 0)
            positive_values = depth[positive]
            positive_fraction = float(np.mean(positive))
            bbox_fraction = float(np.mean(valid_bbox))
            median_depth = float(np.median(positive_values)) if len(positive_values) else 0.0
            rotation_determinants = np.linalg.det(extrinsic[:, :3, :3])
            rotation_error = float(np.max(np.abs(rotation_determinants - 1.0)))
            translation_span = float(np.linalg.norm(np.ptp(extrinsic[:, :3, 3], axis=0)))
            k_finite = bool(np.all(np.isfinite(intrinsic)))
            focal_min = float(np.min(intrinsic[:, (0, 1), (0, 1)]))
            counts = {str(label): int(np.count_nonzero(labels == label)) for label in (1, 2)}
            for label in object_totals:
                object_totals[label] += counts[str(label)]

            if positive_fraction < 0.01 or not 0.03 < median_depth < 20.0:
                errors.append(f"{name}: depth is missing or has implausible metric scale")
            if bbox_fraction < 0.001:
                errors.append(f"{name}: almost no depth survives the configured world bounds")
            if not k_finite or focal_min <= 0:
                errors.append(f"{name}: invalid camera intrinsics")
            if not np.all(np.isfinite(extrinsic)) or rotation_error > 0.03:
                errors.append(f"{name}: invalid camera-to-world rotations")
            if "wrist" in name and translation_span < 0.001:
                warnings.append(f"{name}: camera moved less than 1 mm in sampled frames")
            cameras[name] = {
                "positive_depth_fraction": positive_fraction,
                "world_bbox_fraction": bbox_fraction,
                "median_depth_m": median_depth,
                "rotation_determinant_max_error": rotation_error,
                "camera_translation_span_m": translation_span,
                "object_label_pixels": counts,
            }

        for label, total in object_totals.items():
            if total < 20:
                errors.append(f"bottle label {label} has fewer than 20 visible pixels across sampled cameras")

        expected_dt = 1.0 / float(handle.attrs.get("frequency", 10.0))
        median_dt = float(np.median(deltas)) if len(deltas) else 0.0
        if len(deltas) and abs(median_dt - expected_dt) > max(0.03, expected_dt * 0.35):
            errors.append(f"median timestamp interval {median_dt:.4f}s does not match {expected_dt:.4f}s")
        report = {
            "path": str(path),
            "passed": not errors,
            "frames": frames,
            "sampled_frames": len(indices),
            "frequency_hz": float(handle.attrs.get("frequency", 0.0)),
            "median_timestamp_interval_s": median_dt,
            "object_label_pixels_all_cameras": {str(key): value for key, value in object_totals.items()},
            "cameras": cameras,
            "errors": errors,
            "warnings": warnings,
        }
    return report


def _annotate_rgb(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = rgb.copy()
    for label, color in _LABEL_COLORS.items():
        mask = labels == label
        result[mask] = (0.55 * result[mask] + 0.45 * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, tuple(int(value) for value in color[::-1]), 2)
    return result


def _depth_panel(depth: np.ndarray, labels: np.ndarray, valid: np.ndarray, near: float, far: float) -> np.ndarray:
    normalized = np.clip((depth - near) / max(far - near, 1e-6), 0, 1)
    colored = cv2.applyColorMap((255 * (1 - normalized)).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
    colored[~np.isfinite(depth) | (depth <= 0)] = 0
    colored[~valid] = (0.18 * colored[~valid]).astype(np.uint8)
    for label, color in _LABEL_COLORS.items():
        mask = labels == label
        colored[mask] = (0.35 * colored[mask] + 0.65 * color).astype(np.uint8)
    return colored


def render_3dws_review(
    path: str | Path,
    output: str | Path,
    *,
    fps: float | None = None,
    maximum_frames: int = 600,
) -> Path:
    """Render RGB+mask and depth+workspace panels for a packed episode."""

    path, output = Path(path), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".intermediate.mp4")
    with h5py.File(path, "r") as handle:
        total = min(int(handle.attrs["num_frames"]), maximum_frames)
        camera_ids = _camera_ids(handle)
        names = _camera_names(handle)
        if not camera_ids:
            raise ValueError("packed episode has no cameras")
        source_height = int(handle.attrs["H"])
        source_width = int(handle.attrs["W"])
        panel_width = 480
        panel_height = round(source_height * panel_width / source_width)
        canvas_size = (panel_width * len(camera_ids), 60 + 2 * panel_height)
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps or float(handle.attrs["frequency"]), canvas_size)
        if not writer.isOpened():
            raise OSError(f"failed to open video writer for {temporary}")

        scales: dict[int, tuple[float, float]] = {}
        sample_indices = np.unique(np.linspace(0, total - 1, min(total, 40), dtype=np.int64))
        for camera_id in camera_ids:
            sample = np.asarray(handle[f"img/camera_{camera_id}/depth"][sample_indices], dtype=np.float32)
            sample = sample[np.isfinite(sample) & (sample > 0)]
            scales[camera_id] = tuple(np.percentile(sample, (2, 98))) if len(sample) else (0.1, 3.0)

        for frame_index in range(total):
            canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
            timestamp = float(handle["meta/timestamp"][frame_index])
            title = f"3DWS calibration review | frame {frame_index + 1}/{total} | t={timestamp:.2f}s"
            cv2.putText(canvas, title, (22, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (245, 245, 245), 2, cv2.LINE_AA)
            for column, camera_id in enumerate(camera_ids):
                rgb = _decode_rgb(handle[f"img/camera_{camera_id}/rgb_in_ir"], frame_index)
                depth = np.asarray(handle[f"img/camera_{camera_id}/depth"][frame_index], dtype=np.float32)
                labels = np.asarray(handle[f"img/camera_{camera_id}/obj_mask"][frame_index], dtype=np.uint8)
                height, width = depth.shape
                valid = _unpack_mask(handle[f"img/camera_{camera_id}/valid_mask"][frame_index], height, width)
                rgb_panel = _annotate_rgb(rgb, labels)
                near, far = scales[camera_id]
                depth_rgb = _depth_panel(depth, labels, valid, near, far)
                for row, panel in enumerate((rgb_panel, depth_rgb)):
                    resized = cv2.resize(panel, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
                    top = 60 + row * panel_height
                    left = column * panel_width
                    canvas[top : top + panel_height, left : left + panel_width] = resized[..., ::-1]
                label = names.get(camera_id, f"camera_{camera_id}")
                cv2.putText(canvas, label, (column * panel_width + 12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "depth | dim=outside workspace | color=bottle mask", (column * panel_width + 12, 60 + panel_height + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(canvas)
        writer.release()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(temporary), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
            check=True,
        )
        temporary.unlink()
    else:
        temporary.replace(output)
    return output
