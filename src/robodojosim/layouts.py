from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path


def normalize_bottle_mass(robodojo_root: str | Path, *, dry_run: bool = False) -> list[Path]:
    """Correct RoboDojo bottle asset 22's 100x layout-mass typo.

    The asset metadata declares 0.22 kg, while generated bottle-task layouts
    encode 22 kg. Only that exact asset/value pair is changed. Original JSON
    files are retained beside the layout on the first update.
    """

    root = Path(robodojo_root).expanduser().resolve()
    layout_roots = (
        root / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5",
        root / ".cache" / "robodojo_assets_repo" / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5",
    )
    candidates: dict[Path, Path] = {}
    for layout_root in layout_roots:
        if layout_root.exists():
            for path in layout_root.glob("*/put_bottles_into_dustbin_*.json"):
                candidates[path.resolve()] = path

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
        backup = resolved.with_suffix(resolved.suffix + ".robodojosim-original")
        if not backup.exists():
            shutil.copy2(resolved, backup)
        for bottle in matches:
            bottle["physics"]["mass"] = 0.22
        with tempfile.NamedTemporaryFile("w", dir=resolved.parent, delete=False, encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, resolved)
    return changed
