import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from robodojosim.world_model_training import (
    PackedTrainingRoot,
    parse_training_root,
    stage_3dws_latents,
    stage_3dws_training,
)

BBOX = np.array([-0.85, 0.75, -0.55, 0.40, 0.30, 1.35], dtype=np.float32)


def make_packed(path: Path, episode: int, *, bbox: np.ndarray = BBOX) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            format_version="packed_v2",
            episode=episode,
            num_frames=3,
            frequency=10.0,
            bbox=bbox,
            cam_ids=np.array([0, 1, 2], dtype=np.int64),
        )
        handle.create_dataset("action/cartesian_action", data=np.zeros((3, 14), dtype=np.float32))
    return path


def make_root(path: Path, episode: int = 0) -> Path:
    make_packed(path / "train" / f"episode_{episode}.hdf5", episode)
    make_packed(path / "val" / f"episode_{episode + 1}.hdf5", episode + 1)
    return path


def make_exported_root(path: Path, episode: int = 0) -> Path:
    make_packed(path / "train" / f"episode_{episode}.hdf5", episode)
    make_packed(path / "validation" / f"episode_{episode + 1}.hdf5", episode + 1)
    return path


def test_parse_training_root_requires_labeled_absolute_path(tmp_path):
    parsed = parse_training_root(f"brown_success={tmp_path / 'packed'}")
    assert parsed.label == "brown_success"
    assert parsed.path.is_absolute()
    with pytest.raises(ValueError, match="LABEL=/absolute/path"):
        parse_training_root(str(tmp_path))
    with pytest.raises(ValueError, match="invalid root label"):
        parse_training_root(f"Brown Success={tmp_path}")


def test_stage_training_assigns_global_ids_across_duplicate_source_ids(tmp_path):
    first = make_root(tmp_path / "first", episode=0)
    second = make_root(tmp_path / "second", episode=0)
    output = tmp_path / "stage"
    manifest_path = stage_3dws_training(
        (
            PackedTrainingRoot("brown_success", first),
            PackedTrainingRoot("yellow_success", second),
        ),
        output,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["counts"] == {"total": 4, "train": 2, "val": 2, "frames": 12}
    assert [record["global_episode"] for record in manifest["episodes"]] == [0, 1, 2, 3]
    assert [record["original_episode"] for record in manifest["episodes"]] == [0, 1, 0, 1]
    assert sorted(path.name for path in (output / "packed" / "train").iterdir()) == [
        "episode_000000.hdf5",
        "episode_000002.hdf5",
    ]
    assert all(path.is_symlink() for path in (output / "packed" / "train").iterdir())
    assert (
        stage_3dws_training(
            (PackedTrainingRoot("brown_success", first), PackedTrainingRoot("yellow_success", second)), output
        )
        == manifest_path
    )


def test_stage_training_maps_exported_validation_split_to_val(tmp_path):
    root = make_exported_root(tmp_path / "exported")
    output = tmp_path / "stage"
    manifest_path = stage_3dws_training((PackedTrainingRoot("exported", root),), output)
    manifest = json.loads(manifest_path.read_text())
    val_record = next(record for record in manifest["episodes"] if record["split"] == "val")
    assert val_record["source_split"] == "validation"
    assert (output / "packed" / "val" / "episode_000001.hdf5").is_symlink()


def test_stage_training_rejects_bbox_mismatch(tmp_path):
    first = make_root(tmp_path / "first")
    second = make_root(tmp_path / "second")
    make_packed(
        second / "val" / "episode_1.hdf5",
        1,
        bbox=np.array([-1.0, 0.75, -0.55, 0.40, 0.30, 1.35], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="bbox mismatch"):
        stage_3dws_training(
            (PackedTrainingRoot("first", first), PackedTrainingRoot("second", second)), tmp_path / "stage"
        )


def test_stage_latents_matches_global_packed_ids(tmp_path):
    first = make_root(tmp_path / "first", episode=0)
    second = make_root(tmp_path / "second", episode=0)
    stage = tmp_path / "stage"
    manifest = stage_3dws_training((PackedTrainingRoot("first", first), PackedTrainingRoot("second", second)), stage)
    raw = tmp_path / "raw_latents"
    for label in ("first", "second"):
        for split, episode in (("train", 0), ("val", 1)):
            path = raw / label / split / f"episode_{episode}_latents.h5"
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("z", data=np.zeros((3, 1, 2, 2, 2, 2), dtype=np.float16))
                handle.create_dataset("action", data=np.zeros((3, 14), dtype=np.float32))

    latent_manifest = stage_3dws_latents(manifest, raw, stage / "latents")
    result = json.loads(latent_manifest.read_text())
    assert result["counts"] == {"total": 4, "train": 2, "val": 2}
    assert sorted(path.name for path in (stage / "latents" / "train").iterdir()) == [
        "episode_000000_latents.h5",
        "episode_000002_latents.h5",
    ]
    assert all(path.is_symlink() for path in (stage / "latents" / "val").iterdir())
