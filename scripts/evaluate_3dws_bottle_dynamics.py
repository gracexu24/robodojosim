#!/usr/bin/env python3
"""Evaluation-only held-out renderer for a trained 3DWS dynamics checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def evenly_spaced(items, count):
    """Return up to ``count`` items spanning the complete ordered sequence."""
    values = list(items)
    if count <= 0 or count >= len(values):
        return values
    if count == 1:
        return [values[len(values) // 2]]
    indices = [round(i * (len(values) - 1) / (count - 1)) for i in range(count)]
    return [values[index] for index in indices]


def prepare_output_dir(path: Path) -> Path:
    """Create a new output directory, refusing to mix with prior artifacts."""
    if path.exists():
        if any(path.iterdir()):
            raise FileExistsError(f"output directory is not empty: {path}")
    else:
        path.mkdir(parents=True)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_number(path: str) -> int:
    match = re.search(r"episode_(\d+)_latents\.h5$", Path(path).name)
    if not match:
        raise ValueError(f"unexpected latent filename: {path}")
    return int(match.group(1))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--threedws-root", type=Path, required=True)
    result.add_argument("--dynamics-checkpoint", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--vae-checkpoint", type=Path)
    result.add_argument("--val-latent-dir", type=Path)
    result.add_argument("--packed-dir", type=Path, action="append", dest="packed_dirs")
    result.add_argument("--episodes", type=int, default=4, help="evenly spaced validation episodes to render")
    result.add_argument("--render-episodes", type=int, default=4, help="selected episodes receiving gsplat camera mosaics")
    result.add_argument("--metric-episodes", type=int, default=10, help="evenly spaced validation episodes per metric sweep; 0 means all")
    result.add_argument("--horizons", type=int, nargs="+", default=[10, 30, 75, 150])
    result.add_argument("--infer-noises", type=float, nargs="+", default=[0.0, 0.075, 0.15])
    result.add_argument("--seed", type=int, default=20261021)
    result.add_argument("--skip-metrics", action="store_true")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    started = time.time()
    root = args.threedws_root.resolve()
    dynamics_path = args.dynamics_checkpoint.resolve()
    out_dir = prepare_output_dir(args.output_dir.resolve())
    running_path = out_dir / "RUNNING.json"

    if not root.is_dir():
        raise FileNotFoundError(f"3DWS root not found: {root}")
    if not dynamics_path.is_file():
        raise FileNotFoundError(f"dynamics checkpoint not found: {dynamics_path}")
    sys.path.insert(0, str(root))

    import h5py
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the gsplat held-out renderer")

    from models.dynamics_coarse3d import CoarseDynamics3D
    from scripts.train_dynamics import eval_horizons, load_vae, make_vis

    device = torch.device("cuda:0")
    checkpoint = torch.load(dynamics_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    vae_path = (args.vae_checkpoint or Path(checkpoint_args["vae_ckpt"])).resolve()
    val_dir = (args.val_latent_dir or Path(checkpoint_args["val_latent_dir"])).resolve()
    packed_dirs = [path.resolve() for path in (args.packed_dirs or [])]
    if not packed_dirs:
        packed_dirs = [Path(path).resolve() for path in checkpoint_args.get("packed_dirs", [])]
        packed_dirs = [path for path in packed_dirs if path.name == "val" or "val" in path.parts]

    if not vae_path.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_path}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"validation latent directory not found: {val_dir}")
    if not packed_dirs or not all(path.is_dir() for path in packed_dirs):
        raise FileNotFoundError(f"validation packed directory not found: {packed_dirs}")

    latent_files = sorted(val_dir.glob("episode_*_latents.h5"), key=lambda path: episode_number(str(path)))
    if not latent_files:
        raise FileNotFoundError(f"no validation latent files found under {val_dir}")
    selected = evenly_spaced(latent_files, args.episodes)
    with h5py.File(selected[0], "r") as handle:
        latent_channels = int(handle["z"].shape[2])
        action_dim = int(handle["action"].shape[1])

    n_history = int(checkpoint_args["n_history"])
    n_future = int(checkpoint_args["n_future"])
    model = CoarseDynamics3D(
        latent_channels=latent_channels,
        action_dim=action_dim,
        n_history=n_history,
        n_future=n_future,
        width=int(checkpoint_args["dim"]),
        action_emb_dim=128,
        n_levels=int(checkpoint_args["n_levels"]),
        film=bool(checkpoint_args.get("film", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    action_mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
    action_std = np.asarray(checkpoint["action_std"], dtype=np.float32)
    vae, seed_head, bg_bank, feat_cache, render_config = load_vae(str(vae_path), device)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    eval_args = SimpleNamespace(
        n_history=n_history,
        n_future=n_future,
        infer_hist_noise=float(checkpoint_args.get("infer_hist_noise", 0.1)),
        render_eps=min(args.render_episodes, len(selected)),
        packed_dirs=[str(path) for path in packed_dirs],
    )
    val_view = SimpleNamespace(
        n_episodes=len(selected),
        episode_files=[str(path) for path in selected],
        action_mean=action_mean,
        action_std=action_std,
    )

    run_info = {
        "status": "running",
        "started_unix": started,
        "device": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "dynamics_checkpoint": str(dynamics_path),
        "dynamics_epoch": int(checkpoint["epoch"]),
        "vae_checkpoint": str(vae_path),
        "val_latent_dir": str(val_dir),
        "packed_dirs": [str(path) for path in packed_dirs],
        "selected_episodes": [episode_number(str(path)) for path in selected],
        "validation_episode_count": len(latent_files),
    }
    running_path.write_text(json.dumps(run_info, indent=2) + "\n")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    render_tuple = (seed_head, bg_bank, feat_cache, render_config)
    written = make_vis(
        model,
        vae,
        val_view,
        eval_args,
        device,
        int(checkpoint["epoch"]),
        out_dir,
        render=render_tuple,
    )

    metrics = {}
    if not args.skip_metrics:
        for infer_noise in args.infer_noises:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            values = eval_horizons(
                model,
                str(val_dir),
                n_history,
                args.horizons,
                action_mean,
                action_std,
                float(checkpoint_args.get("object_alpha", 1.0)),
                int(checkpoint_args.get("obj_band_vox", 0)),
                infer_noise,
                device,
                args.metric_episodes,
            )
            if not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError(f"non-finite metric at inference noise {infer_noise}: {values}")
            metrics[f"{infer_noise:g}"] = {str(horizon): values[horizon] for horizon in args.horizons}

        with (out_dir / "heldout_horizon_metrics.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["infer_noise", *[f"ar{horizon}" for horizon in args.horizons]])
            for infer_noise in args.infer_noises:
                row = metrics[f"{infer_noise:g}"]
                writer.writerow([infer_noise, *[row[str(horizon)] for horizon in args.horizons]])

    image_paths = sorted({Path(path).resolve() for path in written})
    import cv2
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise RuntimeError(f"unreadable output image: {image_path}")

    finished = time.time()
    manifest = {
        **run_info,
        "status": "complete",
        "finished_unix": finished,
        "wall_seconds": finished - started,
        "dynamics_checkpoint_size": dynamics_path.stat().st_size,
        "dynamics_checkpoint_sha256": sha256(dynamics_path),
        "vae_checkpoint_size": vae_path.stat().st_size,
        "vae_checkpoint_sha256": sha256(vae_path),
        "object_alpha": float(checkpoint_args.get("object_alpha", 1.0)),
        "object_band_voxels": int(checkpoint_args.get("obj_band_vox", 0)),
        "metric_episode_count": args.metric_episodes or len(latent_files),
        "horizons": args.horizons,
        "inference_noise_levels": args.infer_noises,
        "metrics": metrics,
        "images": [str(path.relative_to(out_dir)) for path in image_paths],
        "image_count": len(image_paths),
    }
    (out_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    running_path.unlink()
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
