from __future__ import annotations

import math

import numpy as np

from .types import Pose


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-path SLERP for quaternions in qw, qx, qy, qz order."""

    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q0 + fraction * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return (math.sin((1 - fraction) * theta) / sin_theta) * q0 + (
        math.sin(fraction * theta) / sin_theta
    ) * q1


def interpolate_pose(start: Pose, target: Pose, max_translation_step: float) -> list[Pose]:
    distance = float(np.linalg.norm(target.position - start.position))
    count = max(1, math.ceil(distance / max_translation_step))
    return [
        Pose(
            start.position + (target.position - start.position) * (index / count),
            quaternion_slerp(start.quaternion, target.quaternion, index / count),
        )
        for index in range(1, count + 1)
    ]


def bbox_top(object_pose: Pose, bbox: np.ndarray | None, fallback_height: float) -> float:
    # The object assets used here are upright and the dustbin pose is fixed.  A
    # full oriented-box transform is unnecessary for the shipped task, but the
    # local z extent is still preferable to a hard-coded asset height.
    if bbox is None:
        return float(object_pose.position[2] + fallback_height)
    return float(object_pose.position[2] + max(0.0, float(bbox[5])))
