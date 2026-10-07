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
    target_y: float = 0.30,
    height_scale: float = 0.5,
    dry_run: bool = False,
) -> list[Path]:
    """Create a shared, collision-free tabletop receptacle for both X5 arms.

    The public layout's 47 cm-wide floor bin sits at x=-0.63, outside the
    right arm's measured top-down workspace. Merely moving that full-height
    bin to x=0 intersects the table. The training variant therefore preserves
    the bin's full opening, halves only its height, and fixes its base to the
    rear of the tabletop. Original JSON is backed up by :func:`_write_layout`.

    This also recognizes the short-lived x=0 floor-bin transform so machines
    updated by an older robodojosim revision are migrated safely.
    """

    if height_scale <= 0:
        raise ValueError("height_scale must be positive")

    root = Path(robodojo_root).expanduser().resolve()
    changed: list[Path] = []
    for resolved, path in sorted(_layout_candidates(root).items(), key=lambda item: str(item[0])):
        with resolved.open(encoding="utf-8") as handle:
            data = json.load(handle)
        dustbins = data.get("Geometry", {}).get("dustbin", [])
        matches = [
            item
            for item in dustbins
            if item.get("category_idx") == 0
            and item.get("label") == "dustbin"
            and item.get("relative_plane", "Ground").lower() == "ground"
            and item.get("default_pos", [None])[0] in (-0.63, float(target_x))
        ]
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
        bin_center_z = table_top + 0.325 * float(height_scale)
        for dustbin in matches:
            dustbin["default_pos"] = [float(target_x), float(target_y), bin_center_z]
            dustbin["xlim"] = [float(target_x), float(target_x)]
            dustbin["ylim"] = [float(target_y), float(target_y)]
            dustbin["zlim"] = [bin_center_z, bin_center_z]
            dustbin["relative_plane"] = "Table"
            dustbin["scale"] = [1.0, 1.0, float(height_scale)]
            dustbin.setdefault("physics", {})["collision"] = True
        _write_layout(resolved, data)
    return changed


def centralize_dustbin(
    robodojo_root: str | Path, *, target_x: float = 0.0, dry_run: bool = False
) -> list[Path]:
    """Backward-compatible name for the tabletop training-bin transform."""

    return configure_training_dustbin(robodojo_root, target_x=target_x, dry_run=dry_run)
