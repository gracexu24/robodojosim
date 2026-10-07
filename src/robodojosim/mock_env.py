from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import ObjectState, Pose, SceneSnapshot


@dataclass
class _Bottle:
    state: ObjectState
    in_bin: bool = False


class MockBottleEnv:
    """Tiny kinematic stand-in for controller/recorder testing (not physics)."""

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.home = {
            "left": Pose([-0.32, -0.25, 1.02], [1, 0, 0, 0]),
            "right": Pose([0.32, -0.25, 1.02], [1, 0, 0, 0]),
        }
        self.arms = dict(self.home)
        self.grippers = {"left": 1.0, "right": 1.0}
        xs = np.array([-0.28, -0.08, 0.18, 0.38]) + rng.uniform(-0.025, 0.025, 4)
        ys = rng.uniform(-0.18, -0.02, 4)
        self.bottles = {
            f"bottle{index}": _Bottle(
                ObjectState(Pose([x, y, 0.78], [1, 0, 0, 0]), np.array([-0.03, -0.03, -0.045, 0.03, 0.03, 0.045]))
            )
            for index, (x, y) in enumerate(zip(xs, ys))
        }
        self.dustbin = ObjectState(Pose([-0.63, -0.10, 0.35], [1, 0, 0, 0]))
        self.attachments: dict[str, str | None] = {"left": None, "right": None}
        self.step_count = 0

    @property
    def success(self) -> bool:
        arms_home = all(np.linalg.norm(self.arms[arm].position - self.home[arm].position) < 1e-6 for arm in self.arms)
        return (
            all(bottle.in_bin for bottle in self.bottles.values())
            and arms_home
            and all(value > 0.8 for value in self.grippers.values())
        )

    def snapshot(self) -> SceneSnapshot:
        return SceneSnapshot(
            dict(self.arms),
            dict(self.grippers),
            {label: bottle.state for label, bottle in self.bottles.items()},
            self.dustbin,
        )

    def observation(self) -> dict:
        # A deterministic RGB pattern catches time-axis and image-shape bugs.
        color = np.full((16, 16, 3), self.step_count % 255, dtype=np.uint8)
        return {
            "data_format_version": "v1.0",
            "instruction": self.snapshot().instruction,
            "additional_info": {"frequency": 25},
            "vision": {"cam_head": {"color": color, "shape": np.array(color.shape[:2])}},
            "state": {
                "left_ee_pose": self.arms["left"].as_array(),
                "right_ee_pose": self.arms["right"].as_array(),
                "left_ee_joint_state": np.asarray([self.grippers["left"]], dtype=np.float32),
                "right_ee_joint_state": np.asarray([self.grippers["right"]], dtype=np.float32),
            },
        }

    def step(self, action: dict[str, np.ndarray]) -> None:
        old_grippers = dict(self.grippers)
        for arm in ("left", "right"):
            self.arms[arm] = Pose.from_array(action[f"{arm}_ee_pose"])
            self.grippers[arm] = float(np.asarray(action[f"{arm}_ee_joint_state"]).reshape(-1)[0])

        for arm in ("left", "right"):
            if old_grippers[arm] > 0.8 and self.grippers[arm] < 0.2:
                self._try_attach(arm)
        for arm, label in self.attachments.items():
            if label is not None:
                bottle = self.bottles[label]
                bottle.state = ObjectState(
                    Pose(self.arms[arm].position, bottle.state.pose.quaternion), bottle.state.bbox
                )
        for arm in ("left", "right"):
            if old_grippers[arm] < 0.2 and self.grippers[arm] > 0.8:
                self._release(arm)
        self.step_count += 1

    def _try_attach(self, arm: str) -> None:
        candidates = []
        for label, bottle in self.bottles.items():
            if bottle.in_bin:
                continue
            distance = np.linalg.norm(self.arms[arm].position - bottle.state.pose.position)
            candidates.append((distance, label))
        if candidates and min(candidates)[0] < 0.10:
            self.attachments[arm] = min(candidates)[1]

    def _release(self, arm: str) -> None:
        label = self.attachments[arm]
        self.attachments[arm] = None
        if label is None:
            return
        # During handover the other arm may still hold the same bottle.
        if label in self.attachments.values():
            return
        bottle = self.bottles[label]
        delta_xy = bottle.state.pose.position[:2] - self.dustbin.pose.position[:2]
        if np.linalg.norm(delta_xy) < 0.20:
            bottle.in_bin = True
