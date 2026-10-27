#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def validate_backend_source(project_root: Path) -> None:
    """Keep the installed overlay tied to the spatial VT-Refine model."""

    source = (
        project_root
        / "integration/robodojo/arx_aloha/env/robot_manager/vt_refine_tactile.py"
    ).read_text(encoding="utf-8")
    for required in (
        "mesh_query_point_sign_winding_number",
        "depth = (-best_distance).clamp(min=0.0)",
        "normal_force_norm = self.normal_stiffness * depth",
        "static_friction = self.tangential_stiffness * tangent_speed",
        "dynamic_friction = self.friction_coefficient * normal_force_norm",
        "layout_manager.get_layout_records",
    ):
        assert required in source, required
    assert "force_grid[5:7, 15:17]" not in source


def validate_spatial_episode(path: Path) -> dict[str, dict[str, int | float | bool]]:
    """Validate a real episode and summarize its spatial taxel activation."""

    expected_pads = {
        "left_left_pad",
        "left_right_pad",
        "right_left_pad",
        "right_right_pad",
    }
    summary = {}
    with h5py.File(path, "r") as handle:
        assert set(handle["tactile"]) == expected_pads
        for pad in sorted(expected_pads):
            force = handle[f"tactile/{pad}/normal_forces"][:]
            positions = handle[f"tactile/{pad}/positions"][:]
            assert force.ndim == 3 and force.shape[1:] == (12, 32)
            assert positions.shape == force.shape + (3,)
            assert np.isfinite(force).all() and np.isfinite(positions).all()
            active = force > 1.0e-8
            active_by_frame = active.reshape(force.shape[0], -1).sum(axis=1)
            active_coords = np.argwhere(active)
            unique_taxels = (
                np.unique(active_coords[:, 1:3], axis=0).shape[0]
                if active_coords.size
                else 0
            )
            old_center = np.zeros((12, 32), dtype=bool)
            old_center[5:7, 15:17] = True
            outside_old_center = int((active & ~old_center).sum())
            assert active.any(), f"{pad} never contacts an object"
            assert unique_taxels > 4, f"{pad} does not contain a spatial pressure field"
            assert outside_old_center > 0, f"{pad} still looks like the aggregate placeholder"
            summary[pad] = {
                "finite": True,
                "max_response": float(force.max()),
                "nonzero_frames": int((active_by_frame > 0).sum()),
                "max_active_taxels": int(active_by_frame.max()),
                "unique_active_taxels": int(unique_taxels),
                "outside_old_center_samples": outside_old_center,
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode",
        type=Path,
        help="optional real HDF5 episode whose four spatial tactile fields should be validated",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    validate_asset(root)
    validate_backend_source(root)
    validate_recording()
    print("ARX+ALOHA asset, VT-Refine backend source, and tactile HDF5 schema validated")
    if args.episode is not None:
        print(json.dumps(validate_spatial_episode(args.episode), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
