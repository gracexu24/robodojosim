#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

from robodojosim.controller import BottleController
from robodojosim.mock_env import MockBottleEnv
from robodojosim.recording import EpisodeRecorder, validate_episode


def _obj_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [float(value) for value in line.split()[1:4]]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("v ")
        ],
        dtype=np.float64,
    )
    if not len(vertices):
        raise ValueError(f"no vertices in {path}")
    return vertices.min(axis=0), vertices.max(axis=0)


def validate_asset(project_root: Path) -> None:
    asset = project_root / "integration/robodojo/arx_aloha/Assets/Robots/x5_aloha"
    urdf = ET.parse(asset / "X5A_Aloha.urdf").getroot()
    joints = {joint.attrib["name"]: joint for joint in urdf.findall("joint")}
    links = {link.attrib["name"] for link in urdf.findall("link")}
    expected_links = {
        "aloha_left_finger",
        "aloha_right_finger",
        "aloha_left_pad",
        "aloha_right_pad",
    }
    assert expected_links <= links
    assert joints["joint7"].find("limit").attrib["upper"] == "0.042"
    assert joints["joint8"].find("limit").attrib["upper"] == "0.042"
    assert joints["joint8"].find("axis").attrib["xyz"] == "0 -1 0"

    low, high = _obj_bounds(asset / "meshes/aloha/aloha_flat_pad.obj")
    np.testing.assert_allclose(high - low, [0.026, 0.003, 0.06665], atol=1e-7)
    row_centers = np.linspace(-0.011, 0.011, 12)
    column_centers = np.linspace(-0.031, 0.031, 32)
    assert row_centers[0] >= low[0] and row_centers[-1] <= high[0]
    assert column_centers[0] >= low[2] and column_centers[-1] <= high[2]


def validate_recording() -> None:
    env = MockBottleEnv()
    observation = env.observation()
    positions = np.zeros((12, 32, 3), dtype=np.float32)
    forces = np.zeros((12, 32), dtype=np.float32)
    forces[5, 16] = 1.25
    observation["tactile"] = {
        name: {"positions": positions, "normal_forces": forces}
        for name in (
            "left_left_pad",
            "left_right_pad",
            "right_left_pad",
            "right_right_pad",
        )
    }
    with tempfile.TemporaryDirectory(prefix="robodojosim-tactile-") as directory:
        recorder = EpisodeRecorder(directory, 1, seed=0, instruction="validation")
        recorder.append(observation, BottleController().next_action(env.snapshot()).action)
        path = recorder.close(success=False)
        errors = validate_episode(path)
        assert errors == [], errors
        with h5py.File(path, "r") as handle:
            assert handle["tactile/left_left_pad/positions"].shape == (1, 12, 32, 3)
            assert handle["tactile/right_right_pad/normal_forces"].shape == (1, 12, 32)
            assert handle["tactile/right_right_pad/normal_forces"][0, 5, 16] == 1.25


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    validate_asset(root)
    validate_recording()
    print("ARX+ALOHA asset and tactile HDF5 schema validated")


if __name__ == "__main__":
    main()
