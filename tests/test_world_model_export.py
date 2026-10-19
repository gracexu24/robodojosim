import json

import h5py
import numpy as np

from robodojosim.recording import EpisodeRecorder, validate_episode
from robodojosim.world_model_export import (
    WorldModelExportConfig,
    export_3dws_dataset,
    inspect_3dws_episode,
    upgrade_3dws_sim_depth_metadata,
    validate_3dws_packed,
)
from robodojosim.world_model_qa import analyze_3dws_quality, render_3dws_review
from robodojosim.world_model_setup import configure_world_model_observations


def _enhanced_episode(directory):
    recorder = EpisodeRecorder(directory, 7, seed=3, instruction="put both bottles in the bin", frequency=25)
    intrinsic = np.asarray([[10.0, 0.0, 2.5], [0.0, 10.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    for frame in range(5):
        color = np.full((4, 6, 3), 20 + frame, dtype=np.uint8)
        observation = {
            "state": {
                "left_arm_joint_state": np.arange(6, dtype=np.float32),
                "left_ee_joint_state": np.asarray([0.1], dtype=np.float32),
                "left_ee_pose": np.asarray([-0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "right_arm_joint_state": np.arange(6, dtype=np.float32) + 1,
                "right_ee_joint_state": np.asarray([0.9], dtype=np.float32),
                "right_ee_pose": np.asarray([0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            },
            "vision": {
                camera: {
                    "color": color,
                    "depth": np.ones((4, 6), dtype=np.float32),
                    "intrinsic_matrix": intrinsic,
                    "extrinsic_matrix": np.eye(4, dtype=np.float32),
                    "shape": np.asarray(color.shape),
                }
                for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist")
            },
        }
        action = {
            "left_ee_pose": np.asarray([-0.2 + frame * 0.01, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "left_ee_joint_state": np.asarray([0.1], dtype=np.float32),
            "right_ee_pose": np.asarray([0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "right_ee_joint_state": np.asarray([0.9], dtype=np.float32),
        }
        teacher = {
            "skill": "carry",
            "bottle0_pose": np.asarray([-0.15, 0, 0, 1, 0, 0, 0], dtype=np.float32),
            "bottle0_bbox": np.asarray([-0.2, -1, 0.5, 0.15, 1, 1.5], dtype=np.float32),
            "bottle1_pose": np.asarray([0.15, 0, 0, 1, 0, 0, 0], dtype=np.float32),
            "bottle1_bbox": np.asarray([-0.15, -1, 0.5, 0.2, 1, 1.5], dtype=np.float32),
        }
        recorder.append(observation, action, teacher=teacher, timestamp=frame / 25)
    path = recorder.close(success=True)
    assert validate_episode(path) == []
    return path


def test_enhanced_recording_exports_exact_3dws_packed_contract(tmp_path):
    source_dir = tmp_path / "source"
    source = _enhanced_episode(source_dir)
    report = inspect_3dws_episode(source)
    assert report.compatible
    assert report.action_dim == 14

    output = tmp_path / "packed"
    export_3dws_dataset(
        source_dir,
        output,
        WorldModelExportConfig(
            bbox=(-1.0, 1.0, -1.0, 1.0, 0.5, 1.5),
            target_fps=10,
            source_camera_axes="opencv",
        ),
    )
    packed = output / "train" / "episode_7.hdf5"
    assert validate_3dws_packed(packed) == []
    with h5py.File(packed, "r") as handle:
        assert handle.attrs["format_version"] == "packed_v2"
        assert handle.attrs["depth_semantics"] == "full_scene_lrc"
        assert handle.attrs["depth_source"] == "isaac_sim_ground_truth"
        assert handle["action/cartesian_action"].shape == (3, 14)
        assert handle["robot/joint_pos"].shape == (3, 14)
        assert handle["img/camera_0/depth"].shape == (3, 4, 6)
        assert set(np.unique(handle["img/camera_0/obj_mask"][0])) == {1, 2}
        assert handle["cam/camera_0/E_cam2world"].shape == (3, 4, 4)
        assert handle["render/camera_0/viewmat"].shape == (3, 4, 4)
        assert handle["meta/source_frame_index"][:].tolist() == [0, 3, 4]
    quality = analyze_3dws_quality(packed)
    assert quality["passed"]
    review = render_3dws_review(packed, output / "review.mp4", maximum_frames=2)
    assert review.stat().st_size > 0
    manifest = json.loads((output / "export_manifest.json").read_text())
    assert manifest["episodes"][0]["has_object_masks"] is True


def test_upgrade_legacy_sim_depth_metadata_is_guarded_and_idempotent(tmp_path):
    source_dir = tmp_path / "source"
    _enhanced_episode(source_dir)
    output = tmp_path / "packed"
    export_3dws_dataset(
        source_dir,
        output,
        WorldModelExportConfig(
            bbox=(-1.0, 1.0, -1.0, 1.0, 0.5, 1.5),
            target_fps=10,
            source_camera_axes="opencv",
        ),
    )
    packed = output / "train" / "episode_7.hdf5"
    legacy = "full_scene_metric_distance_to_image_plane; 0=invalid"
    with h5py.File(packed, "r+") as handle:
        handle.attrs["depth_semantics"] = legacy
        del handle.attrs["depth_source"]
    assert "depth_semantics is not full_scene_lrc" in validate_3dws_packed(packed)
    assert upgrade_3dws_sim_depth_metadata(output, dry_run=True) == {
        "checked": 1,
        "upgraded": 1,
        "current": 0,
    }
    with h5py.File(packed, "r") as handle:
        assert handle.attrs["depth_semantics"] == legacy
    assert upgrade_3dws_sim_depth_metadata(output) == {"checked": 1, "upgraded": 1, "current": 0}
    assert validate_3dws_packed(packed) == []
    assert upgrade_3dws_sim_depth_metadata(output) == {"checked": 1, "upgraded": 0, "current": 1}


def test_rgb_only_episode_reports_geometry_blockers(tmp_path):
    recorder = EpisodeRecorder(tmp_path, 2, seed=0, instruction="test")
    observation = {
        "state": {"left_ee_pose": np.zeros(7), "right_ee_pose": np.zeros(7)},
        "vision": {"cam_head": {"color": np.zeros((4, 4, 3), dtype=np.uint8)}},
    }
    action = {
        "left_ee_pose": np.zeros(7),
        "right_ee_pose": np.zeros(7),
        "left_ee_joint_state": np.ones(1),
        "right_ee_joint_state": np.ones(1),
    }
    recorder.append(observation, action)
    path = recorder.close(success=True)
    report = inspect_3dws_episode(path)
    assert not report.compatible
    assert any("metric depth" in item for item in report.missing)
    assert any("intrinsics" in item for item in report.missing)


def test_world_model_camera_configuration_is_reversible(tmp_path):
    root = tmp_path / "RoboDojo"
    (root / "env_cfg" / "camera").mkdir(parents=True)
    observation = root / "env_cfg" / "arx_x5.yml"
    observation.write_text(
        "observation:\n  vision:\n    approximate_depth: false\n    depth: false\n"
        "    intrinsic_matrix: false\n    extrinsic_matrix: false\n    shape: true\n"
    )
    camera = root / "env_cfg" / "camera" / "camera_config.yml"
    camera.write_text(
        "annotator:\n"
        + "".join(
            f"  {name}:\n    enabled: True\n    rgb_capture:\n      type: rgb\n      device: cpu\n"
            for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
        )
        + "cam_head:\n  camera: {}\n"
    )
    configure_world_model_observations(root)
    assert "depth: true" in observation.read_text()
    assert camera.read_text().count("type: distance_to_image_plane") == 3
    configure_world_model_observations(root, restore=True)
    assert "depth: false" in observation.read_text()
    assert "distance_to_image_plane" not in camera.read_text()
