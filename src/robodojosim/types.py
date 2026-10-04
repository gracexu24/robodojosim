from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def _vector(value: Any, length: int, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite numbers; got shape {result.shape}")
    return result


@dataclass(frozen=True)
class Pose:
    """RoboDojo pose convention: xyz followed by qw, qx, qy, qz."""

    position: FloatArray
    quaternion: FloatArray

    def __post_init__(self) -> None:
        position = _vector(self.position, 3, "position")
        quaternion = _vector(self.quaternion, 4, "quaternion")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-9:
            raise ValueError("quaternion cannot be zero")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion", quaternion / norm)

    @classmethod
    def from_array(cls, value: Any) -> Pose:
        array = _vector(value, 7, "pose")
        return cls(array[:3], array[3:])

    def as_array(self, dtype=np.float32) -> FloatArray:
        return np.concatenate([self.position, self.quaternion]).astype(dtype)

    def at(self, position: Any) -> Pose:
        return Pose(position, self.quaternion)


@dataclass(frozen=True)
class ObjectState:
    pose: Pose
    bbox: FloatArray | None = None

    def __post_init__(self) -> None:
        if self.bbox is not None:
            object.__setattr__(self, "bbox", _vector(self.bbox, 6, "bbox"))


@dataclass(frozen=True)
class SceneSnapshot:
    arms: Mapping[str, Pose]
    grippers: Mapping[str, float]
    bottles: Mapping[str, ObjectState]
    dustbin: ObjectState
    instruction: str = "Pick up the bottles and throw them into the dustbin, using handover when needed."
    raw_teacher_state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = {"left", "right"}
        if set(self.arms) != required:
            raise ValueError(f"arms must have exactly {sorted(required)}")
        if set(self.grippers) != required:
            raise ValueError(f"grippers must have exactly {sorted(required)}")
        if not self.bottles:
            raise ValueError("at least one bottle is required")


Action = dict[str, np.ndarray]
