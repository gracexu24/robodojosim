import json
from pathlib import Path

import pytest

from robodojosim.training import TrainingSpec, build_training_command, inspect_lerobot_dataset, run_training


def make_lerobot_dataset(root: Path) -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": [16]},
        "action": {"dtype": "float32", "shape": [16]},
    }
    for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        features[f"observation.images.{camera}"] = {"dtype": "video", "shape": [3, 480, 640]}
        path = root / "videos" / f"observation.images.{camera}" / "chunk-000"
        path.mkdir(parents=True)
        (path / "file-000.mp4").write_bytes(b"video")
    (root / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"parquet")
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "robot_type": "dual_arx_x5_cartesian",
                "total_episodes": 10,
                "total_frames": 5608,
                "fps": 25,
                "features": features,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_inspect_lerobot_dataset_enforces_training_contract(tmp_path):
    dataset = make_lerobot_dataset(tmp_path / "dataset")
    result = inspect_lerobot_dataset(dataset)
    assert result["total_episodes"] == 10
    assert result["video_files"] == 3
    info = json.loads((dataset / "meta" / "info.json").read_text())
    del info["features"]["action"]
    (dataset / "meta" / "info.json").write_text(json.dumps(info))
    with pytest.raises(TypeError, match="missing required feature 'action'"):
        inspect_lerobot_dataset(dataset)


def test_build_training_commands_include_policy_specific_closed_loop_horizons(tmp_path):
    dataset = tmp_path / "dataset"
    act = build_training_command(
        TrainingSpec("act", dataset, tmp_path / "act", gpu=1, episodes=(0, 2, 4)), executable="train"
    )
    assert "--policy.chunk_size=50" in act
    assert "--policy.n_action_steps=10" in act
    assert "--dataset.episodes=[0,2,4]" in act
    diffusion = build_training_command(TrainingSpec("diffusion", dataset, tmp_path / "diffusion"), executable="train")
    assert "--policy.n_obs_steps=2" in diffusion
    assert "--policy.horizon=32" in diffusion
    assert "--policy.num_inference_steps=10" in diffusion


def test_training_dry_run_validates_without_creating_output(tmp_path):
    dataset = make_lerobot_dataset(tmp_path / "dataset")
    output = tmp_path / "model"
    result = run_training(TrainingSpec("act", dataset, output, gpu=1), dry_run=True)
    assert result["status"] == "planned"
    assert result["cuda_visible_devices"] == "1"
    assert not output.exists()
