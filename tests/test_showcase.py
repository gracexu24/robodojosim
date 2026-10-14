import cv2

from robodojosim.controller import BottleController
from robodojosim.mock_env import MockBottleEnv
from robodojosim.recording import EpisodeRecorder
from robodojosim.showcase import render_skill_video


def test_render_skill_video_labels_recorded_world_model_episode(tmp_path):
    env = MockBottleEnv(seed=0)
    controller = BottleController()
    controller.reset(env.snapshot())
    planned = controller.next_action()
    episode = tmp_path / "episode_000000.hdf5"
    recorder = EpisodeRecorder(
        tmp_path,
        0,
        seed=0,
        instruction="test",
        jpeg_quality=80,
    )
    skills = ["push", "hold", "carry", "regrasp", "rim_hit", "pick_place"]
    for skill in skills:
        recorder.append(
            env.observation(),
            planned.action,
            teacher={"skill": skill, "phase": skill},
        )
        env.step(planned.action)
    recorder.close(success=False, reason="showcase")

    output = tmp_path / "skills.mp4"
    report = render_skill_video(episode, output)
    assert report["skills"] == skills
    assert report["frames"] == len(skills)
    assert output.stat().st_size > 0
    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(skills)
    finally:
        capture.release()
