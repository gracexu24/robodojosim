from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path


def _layout_candidates(root: Path) -> dict[Path, Path]:
    layout_roots = (
        root / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5",
        root / ".cache" / "robodojo_assets_repo" / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5",
    )
    candidates: dict[Path, Path] = {}
    for layout_root in layout_roots:
        if layout_root.exists():
            for path in layout_root.glob("*/put_bottles_into_dustbin_*.json"):
                candidates[path.resolve()] = path
    return candidates


def _write_layout(path: Path, data: dict) -> None:
    backup = path.with_suffix(path.suffix + ".robodojosim-original")
    if not backup.exists():
        shutil.copy2(path, backup)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def normalize_bottle_mass(robodojo_root: str | Path, *, dry_run: bool = False) -> list[Path]:
    """Correct RoboDojo bottle asset 22's 100x layout-mass typo.

    The asset metadata declares 0.22 kg, while generated bottle-task layouts
    encode 22 kg. Only that exact asset/value pair is changed. Original JSON
    files are retained beside the layout on the first update.
    """

    root = Path(robodojo_root).expanduser().resolve()
    candidates = _layout_candidates(root)

    changed: list[Path] = []
    for resolved, path in sorted(candidates.items(), key=lambda item: str(item[0])):
        with resolved.open(encoding="utf-8") as handle:
            data = json.load(handle)
        bottles = data.get("Rigid", {}).get("bottle", [])
        matches = [
            bottle
            for bottle in bottles
            if bottle.get("category_idx") == 22 and bottle.get("physics", {}).get("mass") == 22
        ]
        if not matches:
            continue
        changed.append(path)
        if dry_run:
            continue
        for bottle in matches:
            bottle["physics"]["mass"] = 0.22
        _write_layout(resolved, data)
    return changed


def configure_training_dustbin(
    robodojo_root: str | Path,
    *,
    target_x: float = 0.0,
    target_y: float = -0.10,
    width_scale: float = 0.8,
    depth_scale: float = 0.4,
    height_scale: float = 0.2,
    dry_run: bool = False,
) -> list[Path]:
    """Create a shared, collision-free tabletop receptacle for both X5 arms.

    The public layout's 47 cm-wide floor bin sits at x=-0.63, outside the
    right arm's measured top-down workspace. Merely moving that full-height
    bin to x=0 intersects the table. The training variant reduces the bin to a
    shallow central receptacle at the measured shared reachable point
    (x=0, y=-0.10). Bottles keep their calibrated y positions and are remapped
    only far enough outward in x to avoid the receptacle footprint. Original
    JSON is backed up by :func:`_write_layout`.

    This also recognizes the short-lived x=0 floor-bin transform so machines
    updated by an older robodojosim revision are migrated safely.
    """

    if min(width_scale, depth_scale, height_scale) <= 0:
        raise ValueError("dustbin scale values must be positive")

    root = Path(robodojo_root).expanduser().resolve()
    changed: list[Path] = []
    for resolved, path in sorted(_layout_candidates(root).items(), key=lambda item: str(item[0])):
        with resolved.open(encoding="utf-8") as handle:
            data = json.load(handle)
        bottles = data.get("Rigid", {}).get("bottle", [])
        needs_bottle_relayout = any(
            bottle.get("xlim") in ([0.05, 0.45], [-0.35, 0.05], [0.30, 0.48], [-0.38, -0.30])
            or bottle.get("ylim") in ([0.10, 0.20], [0.06, 0.12])
            for bottle in bottles
        )
        dustbins = data.get("Geometry", {}).get("dustbin", [])
        matches = []
        for item in dustbins:
            if item.get("category_idx") != 0 or item.get("label") != "dustbin":
                continue
            plane = item.get("relative_plane", "Ground").lower()
            position = item.get("default_pos", [None, None])
            scale = item.get("scale", [1.0, 1.0, 1.0])
            public_or_floor_center = plane == "ground" and position[0] in (-0.63, float(target_x))
            old_tabletop_variant = (
                plane == "table"
                and position[0] == float(target_x)
                and position[1] in (0.30, float(target_y))
                and scale
                in ([1.0, 1.0, 0.5], [1.0, 1.0, 0.2], [1.0, 0.6, 0.2], [1.0, 0.4, 0.2])
            )
            desired_bin_needs_bottle_migration = (
                plane == "table"
                and position[:2] == [float(target_x), float(target_y)]
                and scale == [float(width_scale), float(depth_scale), float(height_scale)]
                and needs_bottle_relayout
            )
            if public_or_floor_center or old_tabletop_variant or desired_bin_needs_bottle_migration:
                matches.append(item)
        if not matches:
            continue
        changed.append(path)
        if dry_run:
            continue
        table = data.get("Table", {})
        table_pos = table.get("default_pos", [0.0, 0.0, 0.74])
        table_scale = table.get("scale", [1.0, 1.0, 0.05])
        table_top = float(table_pos[2]) + float(table_scale[2]) / 2.0
        # Asset metadata reports a 0.65 m total height around its origin.
        bin_center_z = round(table_top + 0.325 * float(height_scale), 6)
        for dustbin in matches:
            dustbin["default_pos"] = [float(target_x), float(target_y), bin_center_z]
            dustbin["xlim"] = [float(target_x), float(target_x)]
            dustbin["ylim"] = [float(target_y), float(target_y)]
            dustbin["zlim"] = [bin_center_z, bin_center_z]
            dustbin["relative_plane"] = "Table"
            dustbin["scale"] = [float(width_scale), float(depth_scale), float(height_scale)]
            dustbin.setdefault("physics", {})["collision"] = True
        for bottle in bottles:
            xlim = bottle.get("xlim")
            ylim = bottle.get("ylim")
            position = bottle.get("default_pos")
            if not isinstance(position, list) or len(position) < 2:
                continue
            if xlim == [0.30, 0.48]:
                fraction = (float(position[0]) - 0.30) / 0.18
                position[0] = round(0.05 + 0.40 * fraction, 6)
                xlim = [0.05, 0.45]
                bottle["xlim"] = xlim
            elif xlim == [-0.38, -0.30]:
                fraction = (float(position[0]) + 0.38) / 0.08
                position[0] = round(-0.35 + 0.40 * fraction, 6)
                xlim = [-0.35, 0.05]
                bottle["xlim"] = xlim
            if xlim == [0.05, 0.45]:
                fraction = (float(position[0]) - 0.05) / 0.40
                position[0] = round(0.24 + 0.21 * fraction, 6)
                bottle["xlim"] = [0.24, 0.45]
            elif xlim == [-0.35, 0.05]:
                fraction = (float(position[0]) + 0.35) / 0.40
                position[0] = round(-0.37 + 0.08 * fraction, 6)
                bottle["xlim"] = [-0.37, -0.29]
            if ylim == [0.10, 0.20]:
                fraction = (float(position[1]) - 0.10) / 0.10
                position[1] = round(-0.25 + 0.27 * fraction, 6)
                bottle["ylim"] = [-0.25, 0.02]
            elif ylim == [0.06, 0.12]:
                fraction = (float(position[1]) - 0.06) / 0.06
                position[1] = round(-0.25 + 0.27 * fraction, 6)
                bottle["ylim"] = [-0.25, 0.02]
        _write_layout(resolved, data)
    return changed


def centralize_dustbin(
    robodojo_root: str | Path, *, target_x: float = 0.0, dry_run: bool = False
) -> list[Path]:
    """Backward-compatible name for the tabletop training-bin transform."""

    return configure_training_dustbin(robodojo_root, target_x=target_x, dry_run=dry_run)
