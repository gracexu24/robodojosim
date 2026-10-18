import csv
import json

from robodojosim.mock_env import MockBottleEnv
from robodojosim.pose_analysis import _time_axis_ticks, analyze_pose_distribution
from robodojosim.recording import EpisodeRecorder
from robodojosim.robodojo_adapter import teacher_frame


def test_pose_analysis_exports_every_frame_and_distribution_plots(tmp_path):
    profiles = {
        "success": "world_model_success",
        "failure": "world_model_failure",
        "miscellaneous": "world_model_misc",
    }
    for episode, (category, profile) in enumerate(profiles.items()):
        env = MockBottleEnv(seed=episode)
        snapshot = env.snapshot()
        recorder = EpisodeRecorder(
            tmp_path / category,
            episode,
            seed=episode,
            instruction="test",
            metadata={
                "collection_profile": profile,
                "layout_seed": episode,
                "bottle_side_order": "yellow_left",
                "bottle0_identity": "yellow",
                "bottle1_identity": "brown",
            },
        )
        action = {
            "left_ee_pose": snapshot.arms["left"].as_array(),
            "right_ee_pose": snapshot.arms["right"].as_array(),
            "left_ee_joint_state": [1.0],
            "right_ee_joint_state": [1.0],
        }
        for frame in range(2):
            recorder.append(
                env.observation(),
                action,
                teacher=teacher_frame(snapshot, "test", "bottle0", "left", "pick_place"),
            )
            if frame == 0:
                env.bottles["bottle0"].state = env.bottles["bottle0"].state.__class__(
                    env.bottles["bottle0"].state.pose.at(
                        env.bottles["bottle0"].state.pose.position + [0.01, 0.0, 0.0]
                    ),
                    env.bottles["bottle0"].state.bbox,
                )
                snapshot = env.snapshot()
        recorder.close(success=category == "success")

    output = tmp_path / "analysis"
    summary = analyze_pose_distribution(tmp_path, output)
    assert set(summary["categories"]) == set(profiles)
    assert all(summary["categories"][category]["frames_per_bottle"] == 2 for category in profiles)
    with (output / "bottle_positions_every_frame.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert {row["bottle"] for row in rows} == {"bottle0", "bottle1"}
    assert {row["identity"] for row in rows} == {"brown", "yellow"}
    assert (output / "all_bottle_xyz_positions.png").stat().st_size > 0
    assert (output / "bottle_xy_heatmaps_all_categories.png").stat().st_size > 0
    assert all((output / f"{category}_bottle_xy_heatmap.png").stat().st_size > 0 for category in profiles)
    written_summary = json.loads((output / "pose_distribution_summary.json").read_text())
    assert written_summary["format_version"] == 1
    assert written_summary["heatmap_cell_meters"] == 0.02
    assert written_summary["heatmap_max_frames_per_cell"] > 0


def test_time_axis_ticks_do_not_overlap_short_final_remainder():
    assert _time_axis_ticks(3660.0, 1320) == [0.0, 600.0, 1200.0, 1800.0, 2400.0, 3000.0, 3600.0]
    assert _time_axis_ticks(4020.0, 1320)[-1] == 4020.0
