from __future__ import annotations

import os
from pathlib import Path

_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
_BACKUP_SUFFIX = ".robodojosim-world-model-original"


def _backup(path: Path) -> Path:
    backup = path.with_name(path.name + _BACKUP_SUFFIX)
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    return backup


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _enable_observation_fields(text: str) -> str:
    for key in ("depth", "intrinsic_matrix", "extrinsic_matrix"):
        old = f"    {key}: false"
        new = f"    {key}: true"
        if old in text:
            text = text.replace(old, new, 1)
        elif new not in text:
            raise ValueError(f"RoboDojo observation config has no expected '{key}' field")
    return text


def _enable_camera_depth(text: str) -> str:
    lines = text.splitlines()
    for camera in _CAMERAS:
        start = next((index for index, line in enumerate(lines) if line == f"  {camera}:"), None)
        if start is None:
            raise ValueError(f"RoboDojo camera config has no {camera} annotator section")
        end = next(
            (index for index in range(start + 1, len(lines)) if lines[index] and not lines[index].startswith(" ")),
            len(lines),
        )
        if any(line.strip() == "type: distance_to_image_plane" for line in lines[start:end]):
            continue
        insert = next(
            (
                index + 1
                for index in range(start, end)
                if lines[index].strip() == "device: cpu" and index and lines[index - 1].strip() == "type: rgb"
            ),
            None,
        )
        if insert is None:
            raise ValueError(f"RoboDojo camera config has no RGB capture block for {camera}")
        lines[insert:insert] = [
            "    distance_to_image_plane_capture:",
            "      type: distance_to_image_plane",
            "      device: cpu",
        ]
    return "\n".join(lines) + "\n"


def configure_world_model_observations(
    robodojo_root: str | Path,
    *,
    restore: bool = False,
    dry_run: bool = False,
) -> dict[str, str | bool]:
    """Temporarily enable synchronized depth and camera calibration in RoboDojo."""

    root = Path(robodojo_root)
    observation_path = root / "env_cfg" / "arx_x5.yml"
    camera_path = root / "env_cfg" / "camera" / "camera_config.yml"
    for path in (observation_path, camera_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if restore:
        restored = []
        for path in (observation_path, camera_path):
            backup = path.with_name(path.name + _BACKUP_SUFFIX)
            if backup.exists():
                if not dry_run:
                    _atomic_write(path, backup.read_text(encoding="utf-8"))
                restored.append(str(path))
        return {"enabled": False, "restored": bool(restored), "dry_run": dry_run}

    observation_text = _enable_observation_fields(observation_path.read_text(encoding="utf-8"))
    camera_text = _enable_camera_depth(camera_path.read_text(encoding="utf-8"))
    if not dry_run:
        _backup(observation_path)
        _backup(camera_path)
        _atomic_write(observation_path, observation_text)
        _atomic_write(camera_path, camera_text)
    return {
        "enabled": True,
        "depth": "metric distance_to_image_plane float32",
        "intrinsics": True,
        "extrinsics": "per-frame camera_to_world in Isaac/USD camera axes",
        "dry_run": dry_run,
    }
