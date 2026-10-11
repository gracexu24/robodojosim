from pathlib import Path

from test_training import make_lerobot_dataset

from robodojosim.hub import inspect_checkpoint, push_dataset, push_model


def test_hub_pushes_are_private_and_dry_run_is_non_mutating(tmp_path):
    dataset = make_lerobot_dataset(tmp_path / "dataset")
    result = push_dataset(dataset, "owner/bottle-data", dry_run=True)
    assert result["private"] is True
    assert result["repo_type"] == "dataset"
    assert result["status"] == "planned"

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    assert inspect_checkpoint(checkpoint)["bytes"] > 0
    model = push_model(checkpoint, "owner/bottle-act", dry_run=True)
    assert model["repo_type"] == "model"
    assert not (Path(model["folder"]) / "hub_push.json").exists()
