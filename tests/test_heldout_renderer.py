from pathlib import Path

import pytest

from scripts.evaluate_3dws_bottle_dynamics import evenly_spaced, prepare_output_dir
from scripts.render_3dws_bottle_video import select_episode


def test_evenly_spaced_covers_complete_sequence():
    assert evenly_spaced(list(range(10)), 4) == [0, 3, 6, 9]
    assert evenly_spaced(list(range(5)), 1) == [2]
    assert evenly_spaced(list(range(3)), 0) == [0, 1, 2]


def test_prepare_output_dir_refuses_prior_artifacts(tmp_path: Path):
    output = prepare_output_dir(tmp_path / "new")
    assert output.is_dir()
    assert prepare_output_dir(output) == output
    (output / "prior.txt").write_text("preserve me")
    with pytest.raises(FileExistsError, match="not empty"):
        prepare_output_dir(output)


def test_video_episode_selection_is_explicit_and_deterministic(tmp_path: Path):
    files = [tmp_path / f"episode_{episode:06d}_latents.h5" for episode in (106, 175, 344)]
    assert select_episode(files, 175) == files[1]
    assert select_episode(files) == files[1]
    with pytest.raises(ValueError, match="not present"):
        select_episode(files, 999)
