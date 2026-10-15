from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


CATEGORIES = ("success", "failure", "miscellaneous")
BOTTLES = ("bottle0", "bottle1")


@dataclass(frozen=True)
class PoseTrajectory:
    category: str
    episode_id: int
    success: bool
    frequency: float
    start_seconds: float
    bottle: str
    positions: np.ndarray


def _read_trajectories(root: Path) -> tuple[list[PoseTrajectory], dict[str, np.ndarray]]:
    trajectories: list[PoseTrajectory] = []
    dustbin_positions: dict[str, list[np.ndarray]] = {category: [] for category in CATEGORIES}
    elapsed = 0.0
    for category in CATEGORIES:
        category_elapsed = 0.0
        for path in sorted((root / category).glob("episode_*.hdf5")):
            with h5py.File(path, "r") as handle:
                if not bool(handle.attrs.get("complete", False)):
                    continue
                success = bool(handle.attrs.get("success", False))
                if category == "success" and not success:
                    continue
                frequency = float(handle["additional_info/frequency"][()])
                if frequency <= 0:
                    raise ValueError(f"{path}: recording frequency must be positive")
                length = int(handle.attrs.get("length", 0))
                for bottle in BOTTLES:
                    key = f"teacher/{bottle}_pose"
                    if key not in handle:
                        raise ValueError(f"{path}: missing /{key}")
                    positions = np.asarray(handle[key][:, :3], dtype=np.float64)
                    if positions.shape != (length, 3):
                        raise ValueError(f"{path}: {key} has shape {positions.shape}, expected {(length, 3)}")
                    trajectories.append(
                        PoseTrajectory(
                            category=category,
                            episode_id=int(handle.attrs.get("episode_id", -1)),
                            success=success,
                            frequency=frequency,
                            start_seconds=elapsed + category_elapsed,
                            bottle=bottle,
                            positions=positions,
                        )
                    )
                dustbin = handle.get("teacher/dustbin_pose")
                if isinstance(dustbin, h5py.Dataset) and len(dustbin):
                    dustbin_positions[category].append(np.asarray(dustbin[0, :3], dtype=np.float64))
                category_elapsed += length / frequency
        elapsed += category_elapsed
    return trajectories, {
        category: np.vstack(values) if values else np.empty((0, 3), dtype=np.float64)
        for category, values in dustbin_positions.items()
    }


def _write_csv(path: Path, trajectories: list[PoseTrajectory]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("category", "episode_id", "frame", "recorded_time_s", "bottle", "x", "y", "z", "success"))
        for trajectory in trajectories:
            for frame, (x, y, z) in enumerate(trajectory.positions):
                writer.writerow(
                    (
                        trajectory.category,
                        trajectory.episode_id,
                        frame,
                        f"{trajectory.start_seconds + frame / trajectory.frequency:.6f}",
                        trajectory.bottle,
                        f"{x:.8f}",
                        f"{y:.8f}",
                        f"{z:.8f}",
                        int(trajectory.success),
                    )
                )


def _global_xy_limits(trajectories: list[PoseTrajectory]) -> tuple[tuple[float, float], tuple[float, float]]:
    points = np.vstack([trajectory.positions[:, :2] for trajectory in trajectories if len(trajectory.positions)])
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    padding = np.maximum((high - low) * 0.06, 0.03)
    return (float(low[0] - padding[0]), float(high[0] + padding[0])), (
        float(low[1] - padding[1]),
        float(high[1] + padding[1]),
    )


def _map_xy(
    x: float,
    y: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    left: int,
    top: int,
    width: int,
    height: int,
) -> tuple[int, int]:
    px = left + int((x - xlim[0]) / max(xlim[1] - xlim[0], 1e-9) * width)
    py = top + height - int((y - ylim[0]) / max(ylim[1] - ylim[0], 1e-9) * height)
    return px, py


def _render_heatmap(
    path: Path,
    category: str,
    trajectories: list[PoseTrajectory],
    dustbin_positions: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    canvas = np.full((820, 1000, 3), 248, dtype=np.uint8)
    left, top, width, height = 110, 90, 820, 620
    plot = np.full((height, width, 3), 255, dtype=np.float32)
    colors = {"bottle0": np.array([55, 105, 180], dtype=np.float32), "bottle1": np.array([30, 205, 235], dtype=np.float32)}
    for bottle in BOTTLES:
        points = np.vstack(
            [
                trajectory.positions[:, :2]
                for trajectory in trajectories
                if trajectory.category == category and trajectory.bottle == bottle and len(trajectory.positions)
            ]
        )
        histogram, _, _ = np.histogram2d(
            points[:, 1],
            points[:, 0],
            bins=(height, width),
            range=((ylim[0], ylim[1]), (xlim[0], xlim[1])),
        )
        density = np.log1p(histogram)
        if np.max(density) > 0:
            density /= np.max(density)
        density = np.flipud(density)[..., None]
        alpha = 0.82 * density
        plot = plot * (1.0 - alpha) + colors[bottle] * alpha
    canvas[top : top + height, left : left + width] = np.clip(plot, 0, 255).astype(np.uint8)
    cv2.rectangle(canvas, (left, top), (left + width, top + height), (80, 80, 80), 1)
    if len(dustbin_positions):
        center = np.median(dustbin_positions[:, :2], axis=0)
        corner0 = _map_xy(center[0] - 0.29, center[1] - 0.31, xlim, ylim, left, top, width, height)
        corner1 = _map_xy(center[0] + 0.29, center[1] + 0.31, xlim, ylim, left, top, width, height)
        cv2.rectangle(canvas, corner0, corner1, (30, 30, 30), 3)
        cv2.putText(canvas, "bin footprint", (corner0[0] + 8, min(corner0[1] - 8, corner1[1] - 8)), 0, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    for index in range(6):
        x = xlim[0] + index * (xlim[1] - xlim[0]) / 5
        px, _ = _map_xy(x, ylim[0], xlim, ylim, left, top, width, height)
        cv2.putText(canvas, f"{x:.2f}", (px - 22, top + height + 28), 0, 0.48, (50, 50, 50), 1, cv2.LINE_AA)
        y = ylim[0] + index * (ylim[1] - ylim[0]) / 5
        _, py = _map_xy(xlim[0], y, xlim, ylim, left, top, width, height)
        cv2.putText(canvas, f"{y:.2f}", (35, py + 5), 0, 0.48, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{category.title()} bottle XY distribution", (110, 48), 0, 1.05, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, "world X (m)", (450, 785), 0, 0.65, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(canvas, "world Y (m)", (8, 65), 0, 0.65, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.circle(canvas, (700, 48), 8, tuple(int(value) for value in colors["bottle0"]), -1)
    cv2.putText(canvas, "brown bottle", (715, 54), 0, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.circle(canvas, (835, 48), 8, tuple(int(value) for value in colors["bottle1"]), -1)
    cv2.putText(canvas, "yellow bottle", (850, 54), 0, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"failed to write {path}")


def _render_xyz(path: Path, trajectories: list[PoseTrajectory]) -> None:
    canvas = np.full((1120, 1500, 3), 248, dtype=np.uint8)
    left, width = 120, 1320
    panel_height = 265
    colors = {"bottle0": (55, 105, 180), "bottle1": (30, 180, 220)}
    total_seconds = max(
        trajectory.start_seconds + len(trajectory.positions) / trajectory.frequency for trajectory in trajectories
    )
    category_ranges: dict[str, list[float]] = {}
    for trajectory in trajectories:
        end = trajectory.start_seconds + len(trajectory.positions) / trajectory.frequency
        bounds = category_ranges.setdefault(trajectory.category, [trajectory.start_seconds, end])
        bounds[0] = min(bounds[0], trajectory.start_seconds)
        bounds[1] = max(bounds[1], end)
    cv2.putText(canvas, "Every-frame bottle world positions", (120, 52), 0, 1.15, (25, 25, 25), 2, cv2.LINE_AA)
    for axis, label in enumerate(("X", "Y", "Z")):
        top = 105 + axis * 325
        values = np.concatenate([trajectory.positions[:, axis] for trajectory in trajectories])
        low, high = float(np.min(values)), float(np.max(values))
        padding = max((high - low) * 0.08, 0.02)
        low -= padding
        high += padding
        for category_index, category in enumerate(CATEGORIES):
            if category not in category_ranges:
                continue
            start, end = category_ranges[category]
            x0 = left + int(start / total_seconds * width)
            x1 = left + int(end / total_seconds * width)
            shade = 246 - category_index * 5
            cv2.rectangle(canvas, (x0, top), (x1, top + panel_height), (shade, shade, shade), -1)
            cv2.putText(canvas, category, (x0 + 8, top + 22), 0, 0.48, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (left, top), (left + width, top + panel_height), (75, 75, 75), 1)
        for trajectory in trajectories:
            points: list[tuple[int, int]] = []
            for frame, value in enumerate(trajectory.positions[:, axis]):
                seconds = trajectory.start_seconds + frame / trajectory.frequency
                px = left + int(seconds / total_seconds * width)
                py = top + panel_height - int((float(value) - low) / max(high - low, 1e-9) * panel_height)
                points.append((px, py))
            if len(points) >= 2:
                cv2.polylines(
                    canvas,
                    [np.asarray(points, dtype=np.int32)],
                    False,
                    colors[trajectory.bottle],
                    1,
                    cv2.LINE_AA,
                )
        cv2.putText(canvas, f"world {label} (m)", (12, top + panel_height // 2), 0, 0.58, (35, 35, 35), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{high:.2f}", (68, top + 8), 0, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{low:.2f}", (68, top + panel_height), 0, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    for minute in range(int(total_seconds // 600) + 2):
        seconds = min(total_seconds, minute * 600.0)
        px = left + int(seconds / total_seconds * width)
        cv2.putText(canvas, f"{seconds / 60:.0f}", (px - 8, 1085), 0, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, "cumulative recorded time (min)", (650, 1110), 0, 0.62, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.line(canvas, (1110, 52), (1150, 52), colors["bottle0"], 3, cv2.LINE_AA)
    cv2.putText(canvas, "brown bottle", (1160, 58), 0, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.line(canvas, (1290, 52), (1330, 52), colors["bottle1"], 3, cv2.LINE_AA)
    cv2.putText(canvas, "yellow bottle", (1340, 58), 0, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"failed to write {path}")


def analyze_pose_distribution(dataset_root: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectories, dustbins = _read_trajectories(root)
    if not trajectories:
        raise ValueError(f"no complete balanced world-model episodes found below {root}")
    _write_csv(output / "bottle_positions_every_frame.csv", trajectories)
    xlim, ylim = _global_xy_limits(trajectories)
    for category in CATEGORIES:
        if any(trajectory.category == category for trajectory in trajectories):
            _render_heatmap(
                output / f"{category}_bottle_xy_heatmap.png",
                category,
                trajectories,
                dustbins[category],
                xlim,
                ylim,
            )
    _render_xyz(output / "all_bottle_xyz_positions.png", trajectories)
    summary: dict[str, Any] = {
        "format_version": 1,
        "dataset_root": str(root),
        "position_csv": "bottle_positions_every_frame.csv",
        "xyz_plot": "all_bottle_xyz_positions.png",
        "xy_limits": {"x": list(xlim), "y": list(ylim)},
        "categories": {},
    }
    for category in CATEGORIES:
        selected = [trajectory for trajectory in trajectories if trajectory.category == category]
        frames = sum(len(trajectory.positions) for trajectory in selected if trajectory.bottle == "bottle0")
        seconds = sum(
            len(trajectory.positions) / trajectory.frequency
            for trajectory in selected
            if trajectory.bottle == "bottle0"
        )
        summary["categories"][category] = {
            "episodes": len({trajectory.episode_id for trajectory in selected}),
            "frames_per_bottle": frames,
            "seconds": seconds,
            "heatmap": f"{category}_bottle_xy_heatmap.png",
        }
    (output / "pose_distribution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
