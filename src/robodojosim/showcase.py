from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from .image_codec import decode_jpeg

_SKILL_COLORS = {
    "push": (60, 170, 255),
    "hold": (255, 190, 70),
    "carry": (100, 220, 100),
    "regrasp": (220, 130, 255),
    "rim_hit": (80, 80, 255),
    "pick_place": (255, 255, 255),
}


def _text(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _rgb_frame(dataset: h5py.Dataset, index: int) -> np.ndarray:
    value = np.asarray(dataset[index])
    if value.ndim == 1:
        return decode_jpeg(value)
    if value.ndim == 3 and value.shape[-1] == 3:
        return value.astype(np.uint8, copy=False)
    raise ValueError(f"unsupported recorded image shape: {value.shape}")


def render_skill_video(
    episode: str | Path,
    output: str | Path,
    *,
    camera: str = "cam_head",
) -> dict[str, Any]:
    """Render an annotated MP4 from a recorded world-model episode."""

    episode = Path(episode).resolve()
    output = Path(output).resolve()
    with h5py.File(episode, "r") as handle:
        image_path = f"vision/{camera}/colors"
        if image_path not in handle:
            raise ValueError(f"episode does not contain camera {camera!r}")
        if "teacher/skill" not in handle or "teacher/phase" not in handle:
            raise ValueError("episode does not contain teacher skill/phase annotations")
        images = handle[image_path]
        skills = [_text(value) for value in handle["teacher/skill"][:]]
        phases = [_text(value) for value in handle["teacher/phase"][:]]
        if not (len(images) == len(skills) == len(phases)):
            raise ValueError("camera and teacher annotation lengths do not match")
        if not len(images):
            raise ValueError("cannot render an empty episode")
        fps = int(np.asarray(handle["additional_info/frequency"]).reshape(()))
        first = _rgb_frame(images, 0)
        height, width = first.shape[:2]
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV could not open MP4 writer for {output}")
        try:
            for index in range(len(images)):
                rgb = first if index == 0 else _rgb_frame(images, index)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                skill = skills[index]
                phase = phases[index]
                color = _SKILL_COLORS.get(skill, (220, 220, 220))
                overlay = bgr.copy()
                cv2.rectangle(overlay, (0, 0), (width, 82), (0, 0, 0), thickness=-1)
                bgr = cv2.addWeighted(overlay, 0.62, bgr, 0.38, 0.0)
                cv2.putText(
                    bgr,
                    f"WORLD MODEL SKILL: {skill.upper()}",
                    (20, 33),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.78,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    bgr,
                    f"phase={phase}   frame={index + 1}/{len(images)}",
                    (20, 66),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
                progress = round(width * (index + 1) / len(images))
                cv2.rectangle(bgr, (0, height - 6), (progress, height - 1), color, thickness=-1)
                writer.write(bgr)
        finally:
            writer.release()

    ordered_skills = list(dict.fromkeys(skills))
    return {
        "episode": str(episode),
        "output": str(output),
        "camera": camera,
        "fps": fps,
        "frames": len(skills),
        "seconds": len(skills) / fps,
        "skills": ordered_skills,
        "skill_frames": dict(Counter(skills)),
    }
