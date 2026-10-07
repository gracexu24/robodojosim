from __future__ import annotations

import math

import numpy as np

from .types import Pose


def quaternion_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix for a normalized qw, qx, qy, qz quaternion."""

    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def bbox_world_center(object_pose: Pose, bbox: np.ndarray | None) -> np.ndarray:
    """Transform a mesh-local AABB center into the task's world frame.

    RoboDojo reports the transform of the first mesh together with mesh-local
    bounds.  Some bottle mesh origins are near the neck rather than at the
    geometric center, so targeting the raw transform can make a gripper land
    on one end of an otherwise reachable bottle.
    """

    if bbox is None:
        return object_pose.position.copy()
    bounds = np.asarray(bbox, dtype=np.float64)
    local_center = (bounds[:3] + bounds[3:]) / 2.0
    return object_pose.position + quaternion_rotation_matrix(object_pose.quaternion) @ local_center


def bbox_world_corners(object_pose: Pose, bbox: np.ndarray) -> np.ndarray:
    """Return all eight mesh-local AABB corners transformed to world space."""

    bounds = np.asarray(bbox, dtype=np.float64)
    minimum, maximum = bounds[:3], bounds[3:]
    corners = np.array(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ]
    )
    return object_pose.position + corners @ quaternion_rotation_matrix(object_pose.quaternion).T


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for qw, qx, qy, qz quaternions."""

    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def align_tool_yaw_to_bbox_major_axis(
    grasp_quaternion: np.ndarray,
    object_pose: Pose,
    bbox: np.ndarray | None,
    *,
    tool_length_axis: int = 2,
) -> np.ndarray:
    """Yaw a vertical gripper so its fingers run along an object's major axis.

    The X5 tool's local z axis runs along the fingers when its local x axis is
    pointed down.  Aligning that axis with a bottle's projected longest mesh
    dimension makes the jaws close across the bottle instead of squeezing it
    lengthwise.  A vertical major axis has no useful table-plane direction,
    so the calibrated base orientation is retained in that case.
    """

    quaternion = np.asarray(grasp_quaternion, dtype=np.float64)
    if bbox is None:
        return quaternion.copy()
    bounds = np.asarray(bbox, dtype=np.float64)
    major_axis = int(np.argmax(bounds[3:] - bounds[:3]))
    desired = quaternion_rotation_matrix(object_pose.quaternion)[:2, major_axis]
    current = quaternion_rotation_matrix(quaternion)[:2, tool_length_axis]
    desired_norm = float(np.linalg.norm(desired))
    current_norm = float(np.linalg.norm(current))
    if desired_norm < 1e-4 or current_norm < 1e-4:
        return quaternion.copy()
    desired /= desired_norm
    current /= current_norm
    delta = math.atan2(current[0] * desired[1] - current[1] * desired[0], float(np.dot(current, desired)))
    # Both axes describe unoriented lines, so choose the equivalent yaw with
    # the smallest rotation from the calibrated arm posture.
    if delta > math.pi / 2:
        delta -= math.pi
    elif delta < -math.pi / 2:
        delta += math.pi
    yaw = np.array([math.cos(delta / 2), 0.0, 0.0, math.sin(delta / 2)])
    result = quaternion_multiply(yaw, quaternion)
    return result / np.linalg.norm(result)


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
    return (math.sin((1 - fraction) * theta) / sin_theta) * q0 + (math.sin(fraction * theta) / sin_theta) * q1


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
    if bbox is None:
        return float(object_pose.position[2] + fallback_height)
    return float(np.max(bbox_world_corners(object_pose, bbox)[:, 2]))
