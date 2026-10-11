from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from .training import inspect_lerobot_dataset


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {path}")
    configs = sorted(path.glob("*.json"))
    weights = sorted(path.glob("*.safetensors"))
    if not configs or not weights:
        raise ValueError("checkpoint must contain JSON configuration and safetensors weights")
    return {
        "root": str(path),
        "config_files": [item.name for item in configs],
        "weight_files": [item.name for item in weights],
        "bytes": sum(item.stat().st_size for item in path.rglob("*") if item.is_file()),
    }


def push_dataset(
    dataset: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    revision: str = "main",
    dry_run: bool = False,
) -> dict[str, Any]:
    summary = inspect_lerobot_dataset(dataset)
    return _push_folder(
        Path(dataset),
        repo_id,
        repo_type="dataset",
        private=private,
        revision=revision,
        dry_run=dry_run,
        summary=summary,
    )


def push_model(
    checkpoint: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    revision: str = "main",
    dry_run: bool = False,
) -> dict[str, Any]:
    summary = inspect_checkpoint(checkpoint)
    return _push_folder(
        Path(checkpoint),
        repo_id,
        repo_type="model",
        private=private,
        revision=revision,
        dry_run=dry_run,
        summary=summary,
    )


def _push_folder(
    folder: Path,
    repo_id: str,
    *,
    repo_type: str,
    private: bool,
    revision: str,
    dry_run: bool,
    summary: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "folder": str(folder.resolve()),
        "repo_id": repo_id,
        "repo_type": repo_type,
        "private": private,
        "revision": revision,
        "summary": summary,
        "status": "planned" if dry_run else "uploaded",
    }
    if dry_run:
        return result
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required; install robodojosim[train]") from exc
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    uploader = getattr(api, "upload_large_folder", api.upload_folder)
    available = inspect.signature(uploader).parameters
    upload_kwargs = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "folder_path": str(folder.resolve()),
        "revision": revision,
    }
    uploader(**{key: value for key, value in upload_kwargs.items() if key in available})
    return result


def format_push_plan(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True)
