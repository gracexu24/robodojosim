from pathlib import Path

import pytest

from scripts.evaluate_3dws_bottle_dynamics import evenly_spaced, prepare_output_dir


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
