import json
from pathlib import Path

import numpy as np

from robodojosim.lerobot_export import STATE_NAMES, export_lerobot
from robodojosim.mock_env import MockBottleEnv
from robodojosim.recording import EpisodeRecorder


class FakeLeRobotDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.root = Path(kwargs["root"])
        self.root.mkdir(parents=True)
        self.frames = []
        self.episodes = []
        self.finalized = False

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.episodes.append(self.frames)
        self.frames = []

    def finalize(self):
        self.finalized = True
        (self.root / "meta").mkdir()
        (self.root / "meta" / "info.json").write_text("{}")


def _write_episode(directory, episode_id, *, success, profile, encoded=False):
    env = MockBottleEnv(episode_id)
    observation = env.observation()
    action = {
        "left_ee_pose": np.arange(7, dtype=np.float32),
        "left_ee_joint_state": np.array([7], dtype=np.float32),
        "right_ee_pose": np.arange(8, 15, dtype=np.float32),
        "right_ee_joint_state": np.array([15], dtype=np.float32),
    }
    recorder = EpisodeRecorder(
        directory,
        episode_id,
        seed=episode_id,
        instruction="put the bottles into the dustbin",
        metadata={"collection_profile": profile, "layout_seed": episode_id},
        jpeg_quality=90 if encoded else None,
    )
    recorder.append(observation, action)
    recorder.close(success=success)


def test_export_lerobot_maps_cartesian_features_and_filters_policy_failures(tmp_path):
    _write_episode(tmp_path, 0, success=True, profile="policy", encoded=True)
    _write_episode(tmp_path, 1, success=False, profile="policy")
    made = []

    def factory(**kwargs):
        dataset = FakeLeRobotDataset(**kwargs)
        made.append(dataset)
        return dataset

    output = export_lerobot(
        tmp_path,
        tmp_path / "lerobot",
        profile="policy",
        repo_id="test/bottles",
        dataset_factory=factory,
    )

    dataset = made[0]
    assert dataset.finalized
    assert len(dataset.episodes) == 1
    frame = dataset.episodes[0][0]
    np.testing.assert_array_equal(frame["action"], np.arange(16, dtype=np.float32))
    assert frame["observation.state"].shape == (16,)
    assert frame["observation.images.cam_head"].shape == (16, 16, 3)
    assert dataset.kwargs["features"]["action"]["names"] == [STATE_NAMES]
    manifest = json.loads((output / "robodojosim_export.json").read_text())
    assert manifest["lerobot_format"] == "v3.0"
    assert [item["episode_id"] for item in manifest["source_episodes"]] == [0]


def test_export_lerobot_includes_failed_complete_world_model_episode(tmp_path):
    _write_episode(tmp_path, 4, success=False, profile="world_model")
    made = []

    def factory(**kwargs):
        made.append(FakeLeRobotDataset(**kwargs))
        return made[-1]

    output = export_lerobot(
        tmp_path,
        tmp_path / "lerobot",
        profile="world_model",
        dataset_factory=factory,
    )
    assert len(made[0].episodes) == 1
    assert export_lerobot(
        tmp_path,
        output,
        profile="world_model",
        dataset_factory=factory,
    ) == output
