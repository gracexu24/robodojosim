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
IDENTITIES = ("brown", "yellow")
_IDENTITY_LABELS = {"brown": "Brown bottle", "yellow": "Yellow bottle"}
_IDENTITY_COLORS = {
    "brown": np.array([45, 95, 185], dtype=np.float32),
    "yellow": np.array([20, 170, 230], dtype=np.float32),
}
_HEATMAP_CELL_METERS = 0.02


@dataclass(frozen=True)
class PoseTrajectory:
    category: str
    episode_id: int
    success: bool
    frequency: float
    start_seconds: float
    bottle: str
    identity: str
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
                metadata: dict[str, Any] = {}
                if "metadata_json" in handle:
                    raw_metadata = handle["metadata_json"][()]
                    if isinstance(raw_metadata, bytes):
                        raw_metadata = raw_metadata.decode("utf-8")
                    metadata = json.loads(str(raw_metadata))
                for bottle in BOTTLES:
                    key = f"teacher/{bottle}_pose"
                    if key not in handle:
                        raise ValueError(f"{path}: missing /{key}")
                    positions = np.asarray(handle[key][:, :3], dtype=np.float64)
                    if positions.shape != (length, 3):
                        raise ValueError(f"{path}: {key} has shape {positions.shape}, expected {(length, 3)}")
                    identity = str(
                        metadata.get(
                            f"{bottle}_identity",
                            "brown" if bottle == "bottle0" else "yellow",
                        )
                    )
                    if identity not in IDENTITIES:
                        identity = "brown" if bottle == "bottle0" else "yellow"
                    trajectories.append(
                        PoseTrajectory(
                            category=category,
                            episode_id=int(handle.attrs.get("episode_id", -1)),
                            success=success,
                            frequency=frequency,
                            start_seconds=elapsed + category_elapsed,
                            bottle=bottle,
                            identity=identity,
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
        writer.writerow(
            (
                "category",
                "episode_id",
                "frame",
                "recorded_time_s",
                "bottle",
                "identity",
                "x",
                "y",
                "z",
                "success",
            )
        )
        for trajectory in trajectories:
            for frame, (x, y, z) in enumerate(trajectory.positions):
                writer.writerow(
                    (
                        trajectory.category,
                        trajectory.episode_id,
                        frame,
                        f"{trajectory.start_seconds + frame / trajectory.frequency:.6f}",
                        trajectory.bottle,
                        trajectory.identity,
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


def _heatmap_edges(
    xlim: tuple[float, float], ylim: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    x_bins = max(16, int(np.ceil((xlim[1] - xlim[0]) / _HEATMAP_CELL_METERS)))
    y_bins = max(16, int(np.ceil((ylim[1] - ylim[0]) / _HEATMAP_CELL_METERS)))
    return np.linspace(*xlim, x_bins + 1), np.linspace(*ylim, y_bins + 1)


def _heatmap_histogram(
    trajectories: list[PoseTrajectory],
    category: str,
    identity: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> np.ndarray:
    selected = [
        trajectory.positions[:, :2]
        for trajectory in trajectories
        if trajectory.category == category
        and trajectory.identity == identity
        and len(trajectory.positions)
    ]
    if not selected:
        return np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=np.float64)
    points = np.vstack(selected)
    histogram, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=(y_edges, x_edges))
    return histogram


def _heatmap_intensity(histogram: np.ndarray, maximum: float) -> np.ndarray:
    if maximum <= 0:
        return np.zeros_like(histogram, dtype=np.float32)
    # Log scaling retains short transit paths while preventing repeated reset and
    # hold poses from washing out every other occupied cell. The square root is
    # a display gamma only; the colorbar remains labelled in recorded frames.
    normalized = np.log1p(histogram) / np.log1p(maximum)
    return np.sqrt(np.clip(normalized, 0.0, 1.0)).astype(np.float32)


def _draw_heatmap_panel(
    canvas: np.ndarray,
    *,
    left: int,
    top: int,
    width: int,
    height: int,
    category: str,
    identity: str,
    histogram: np.ndarray,
    maximum: float,
    dustbin_positions: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    show_category: bool,
) -> None:
    intensity = np.flipud(_heatmap_intensity(histogram, maximum))
    intensity = cv2.resize(intensity, (width, height), interpolation=cv2.INTER_NEAREST)[..., None]
    background = np.full((height, width, 3), 252.0, dtype=np.float32)
    color = _IDENTITY_COLORS[identity]
    plot = background * (1.0 - intensity) + color * intensity
    canvas[top : top + height, left : left + width] = np.clip(plot, 0, 255).astype(np.uint8)

    for index in range(6):
        x = xlim[0] + index * (xlim[1] - xlim[0]) / 5
        px, _ = _map_xy(x, ylim[0], xlim, ylim, left, top, width, height)
        cv2.line(canvas, (px, top), (px, top + height), (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{x:.2f}", (px - 21, top + height + 25), 0, 0.43, (55, 55, 55), 1, cv2.LINE_AA)
        y = ylim[0] + index * (ylim[1] - ylim[0]) / 5
        _, py = _map_xy(xlim[0], y, xlim, ylim, left, top, width, height)
        cv2.line(canvas, (left, py), (left + width, py), (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{y:.2f}", (left - 55, py + 5), 0, 0.43, (55, 55, 55), 1, cv2.LINE_AA)

    if len(dustbin_positions):
        center = np.median(dustbin_positions[:, :2], axis=0)
        corner0 = _map_xy(center[0] - 0.29, center[1] - 0.31, xlim, ylim, left, top, width, height)
        corner1 = _map_xy(center[0] + 0.29, center[1] + 0.31, xlim, ylim, left, top, width, height)
        x0, x1 = sorted((corner0[0], corner1[0]))
        y0, y1 = sorted((corner0[1], corner1[1]))
        x0, x1 = max(left, x0), min(left + width, x1)
        y0, y1 = max(top, y0), min(top + height, y1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (35, 35, 35), 3, cv2.LINE_AA)
        cv2.putText(canvas, "bin", (max(left + 5, x0 + 7), min(top + height - 8, y0 + 20)), 0, 0.48, (35, 35, 35), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (left, top), (left + width, top + height), (70, 70, 70), 1)
    panel_title = _IDENTITY_LABELS[identity]
    if show_category:
        panel_title = f"{category.title()} - {panel_title.lower()}"
    cv2.putText(canvas, panel_title, (left, top - 18), 0, 0.68, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "World X (m)", (left + width // 2 - 42, top + height + 50), 0, 0.52, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "World Y (m)", (left - 55, top - 43), 0, 0.48, (40, 40, 40), 1, cv2.LINE_AA)

    bar_left, bar_top, bar_width, bar_height = left + 40, top + height + 68, width - 80, 14
    gradient = np.linspace(0.0, 1.0, bar_width, dtype=np.float32)[None, :, None]
    gradient_image = 252.0 * (1.0 - gradient) + color * gradient
    canvas[bar_top : bar_top + bar_height, bar_left : bar_left + bar_width] = np.repeat(
        np.clip(gradient_image, 0, 255).astype(np.uint8), bar_height, axis=0
    )
    cv2.rectangle(canvas, (bar_left, bar_top), (bar_left + bar_width, bar_top + bar_height), (95, 95, 95), 1)
    tick_values = sorted({0, 1, 10, 100, int(maximum)})
    for count in tick_values:
        position = 0.0 if count <= 0 else np.sqrt(np.log1p(count) / np.log1p(max(maximum, 1.0)))
        px = bar_left + int(np.clip(position, 0.0, 1.0) * bar_width)
        cv2.line(canvas, (px, bar_top + bar_height), (px, bar_top + bar_height + 5), (60, 60, 60), 1)
        label = f"{count:,}"
        cv2.putText(canvas, label, (px - min(18, len(label) * 4), bar_top + bar_height + 21), 0, 0.38, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Recorded frames per 2 cm cell (log scale)", (bar_left + 50, bar_top + 45), 0, 0.42, (50, 50, 50), 1, cv2.LINE_AA)


def _render_heatmap(
    path: Path,
    category: str,
    trajectories: list[PoseTrajectory],
    dustbin_positions: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    maximum: float,
) -> None:
    canvas = np.full((760, 1280, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, f"{category.title()} bottle XY occupancy", (65, 48), 0, 1.0, (25, 25, 25), 2, cv2.LINE_AA)
    for identity, left in zip(IDENTITIES, (85, 685), strict=True):
        _draw_heatmap_panel(
            canvas,
            left=left,
            top=115,
            width=510,
            height=440,
            category=category,
            identity=identity,
            histogram=_heatmap_histogram(trajectories, category, identity, x_edges, y_edges),
            maximum=maximum,
            dustbin_positions=dustbin_positions,
            xlim=xlim,
            ylim=ylim,
            show_category=False,
        )
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"failed to write {path}")


def _render_all_heatmaps(
    path: Path,
    trajectories: list[PoseTrajectory],
    dustbins: dict[str, np.ndarray],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    maximum: float,
) -> None:
    canvas = np.full((2190, 1280, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, "Bottle XY occupancy by collection category", (65, 48), 0, 1.0, (25, 25, 25), 2, cv2.LINE_AA)
    for row, category in enumerate(CATEGORIES):
        for identity, left in zip(IDENTITIES, (85, 685), strict=True):
            _draw_heatmap_panel(
                canvas,
                left=left,
                top=115 + row * 700,
                width=510,
                height=440,
                category=category,
                identity=identity,
                histogram=_heatmap_histogram(trajectories, category, identity, x_edges, y_edges),
                maximum=maximum,
                dustbin_positions=dustbins[category],
                xlim=xlim,
                ylim=ylim,
                show_category=True,
            )
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"failed to write {path}")


def _render_xyz(path: Path, trajectories: list[PoseTrajectory]) -> None:
    canvas = np.full((1120, 1500, 3), 248, dtype=np.uint8)
    left, width = 120, 1320
    panel_height = 265
    colors = {"brown": (55, 105, 180), "yellow": (30, 180, 220)}
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
                    colors[trajectory.identity],
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
    cv2.line(canvas, (1110, 52), (1150, 52), colors["brown"], 3, cv2.LINE_AA)
    cv2.putText(canvas, "brown bottle", (1160, 58), 0, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.line(canvas, (1290, 52), (1330, 52), colors["yellow"], 3, cv2.LINE_AA)
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
    x_edges, y_edges = _heatmap_edges(xlim, ylim)
    maximum = max(
        float(np.max(_heatmap_histogram(trajectories, category, identity, x_edges, y_edges)))
        for category in CATEGORIES
        for identity in IDENTITIES
    )
    for category in CATEGORIES:
        if any(trajectory.category == category for trajectory in trajectories):
            _render_heatmap(
                output / f"{category}_bottle_xy_heatmap.png",
                category,
                trajectories,
                dustbins[category],
                xlim,
                ylim,
                x_edges,
                y_edges,
                maximum,
            )
    _render_all_heatmaps(
        output / "bottle_xy_heatmaps_all_categories.png",
        trajectories,
        dustbins,
        xlim,
        ylim,
        x_edges,
        y_edges,
        maximum,
    )
    _render_xyz(output / "all_bottle_xyz_positions.png", trajectories)
    summary: dict[str, Any] = {
        "format_version": 1,
        "dataset_root": str(root),
        "position_csv": "bottle_positions_every_frame.csv",
        "xyz_plot": "all_bottle_xyz_positions.png",
        "combined_xy_heatmap": "bottle_xy_heatmaps_all_categories.png",
        "heatmap_cell_meters": _HEATMAP_CELL_METERS,
        "heatmap_max_frames_per_cell": maximum,
        "xy_limits": {"x": list(xlim), "y": list(ylim)},
        "categories": {},
    }
    for category in CATEGORIES:
        selected = [trajectory for trajectory in trajectories if trajectory.category == category]
        frames = sum(len(trajectory.positions) for trajectory in selected if trajectory.identity == "brown")
        seconds = sum(
            len(trajectory.positions) / trajectory.frequency
            for trajectory in selected
            if trajectory.identity == "brown"
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
