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


def centralize_dustbin(
    robodojo_root: str | Path, *, target_x: float = 0.0, dry_run: bool = False
) -> list[Path]:
    """Move the bottle-task dustbin into both X5 arms' reachable workspace.

    The public layout places the bin at x=-0.63, outside the right arm's
    top-down workspace. Training campaigns use x=0 so each arm can complete a
    direct pick-and-place without an unreachable mid-air handoff. Only dustbin
    entries still at the public x=-0.63 position are changed.
    """

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
            and item.get("default_pos", [None])[0] == -0.63
        ]
        if not matches:
            continue
        changed.append(path)
        if dry_run:
            continue
        for dustbin in matches:
            dustbin["default_pos"][0] = float(target_x)
            if dustbin.get("xlim") == [-0.63, -0.63]:
                dustbin["xlim"] = [float(target_x), float(target_x)]
        _write_layout(resolved, data)
    return changed
