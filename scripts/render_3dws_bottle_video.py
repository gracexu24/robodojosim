#!/usr/bin/env python3
"""Render a frozen 3DWS dynamics checkpoint as a held-out comparison video."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path


def episode_number(path: Path) -> int:
    match = re.search(r"episode_(\d+)_latents\.h5$", path.name)
    if not match:
        raise ValueError(f"unexpected latent filename: {path}")
    return int(match.group(1))


def select_episode(files, requested=None):
    ordered = sorted(files, key=episode_number)
    if not ordered:
        raise FileNotFoundError("no validation latent episodes found")
    if requested is None:
        return ordered[len(ordered) // 2]
    for path in ordered:
        if episode_number(path) == requested:
            return path
    raise ValueError(f"held-out episode {requested} is not present")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--threedws-root", type=Path, required=True)
    result.add_argument("--dynamics-checkpoint", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--vae-checkpoint", type=Path)
    result.add_argument("--val-latent-dir", type=Path)
    result.add_argument("--packed-dir", type=Path)
    result.add_argument("--episode-id", type=int)
    result.add_argument("--frames", type=int, default=30)
    result.add_argument("--fps", type=float, default=10.0)
    result.add_argument("--infer-noise", type=float, default=0.1)
    result.add_argument("--seed", type=int, default=20261021)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    started = time.time()
    root = args.threedws_root.resolve()
    checkpoint_path = args.dynamics_checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prior video: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() != ".mp4":
        raise ValueError("output must have an .mp4 suffix")
    if not root.is_dir() or not checkpoint_path.is_file():
        raise FileNotFoundError("3DWS root or dynamics checkpoint is missing")
    sys.path.insert(0, str(root))

    import cv2
    import h5py
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gsplat video rendering")
    from models.dynamics_coarse3d import CoarseDynamics3D
    from scripts.train_dynamics import (
        _find_packed,
        _load_cams_packed,
        _render_latent,
        load_vae,
        predict_chunk_inf,
    )

    device = torch.device("cuda:0")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    vae_path = (args.vae_checkpoint or Path(saved["vae_ckpt"])).resolve()
    val_dir = (args.val_latent_dir or Path(saved["val_latent_dir"])).resolve()
    packed_dir = args.packed_dir.resolve() if args.packed_dir else None
    if packed_dir is None:
        choices = [Path(item).resolve() for item in saved.get("packed_dirs", [])]
        choices = [path for path in choices if path.name == "val" or "val" in path.parts]
        packed_dir = choices[0] if choices else None
    if not vae_path.is_file() or not val_dir.is_dir() or packed_dir is None or not packed_dir.is_dir():
        raise FileNotFoundError("VAE checkpoint, validation latents, or packed validation data is missing")

    latent_path = select_episode(val_dir.glob("episode_*_latents.h5"), args.episode_id)
    episode_id = episode_number(latent_path)
    packed_token = re.search(r"episode_(\d+)_latents\.h5$", latent_path.name).group(1)
    packed_path = _find_packed([str(packed_dir)], packed_token)
    if packed_path is None:
        raise FileNotFoundError(f"packed episode {episode_id} not found under {packed_dir}")

    with h5py.File(latent_path, "r") as handle:
        zgt = handle["z"][:, 0].astype(np.float32)
        actions = handle["action"][:].astype(np.float32)
        latent_channels = int(zgt.shape[1])
        action_dim = int(actions.shape[1])
    n_history = int(saved["n_history"])
    n_future = int(saved["n_future"])
    frame_count = min(args.frames, len(zgt) - n_history)
    if frame_count <= 0:
        raise ValueError(f"episode {episode_id} is too short for {n_history} history frames")

    model = CoarseDynamics3D(
        latent_channels=latent_channels,
        action_dim=action_dim,
        n_history=n_history,
        n_future=n_future,
        width=int(saved["dim"]),
        action_emb_dim=128,
        n_levels=int(saved["n_levels"]),
        film=bool(saved.get("film", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    vae, seed_head, bg_bank, feat_cache, render_config = load_vae(str(vae_path), device)
    if seed_head is None:
        raise RuntimeError("the VAE checkpoint does not contain a gaussian render head")
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    action_mean = np.asarray(checkpoint["action_mean"], np.float32)
    action_std = np.asarray(checkpoint["action_std"], np.float32)
    actions = torch.from_numpy((actions - action_mean) / action_std).float().to(device)
    zgt_t = torch.from_numpy(zgt).float().to(device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    z_history = zgt_t[:n_history].unsqueeze(0)
    predictions = []
    while len(predictions) < frame_count:
        future_start = n_history + len(predictions)
        future_actions = actions[future_start : future_start + n_future].unsqueeze(0)
        if future_actions.shape[1] < n_future:
            future_actions = torch.cat(
                [future_actions, future_actions[:, -1:].expand(-1, n_future - future_actions.shape[1], -1)],
                dim=1,
            )
        predicted = predict_chunk_inf(
            model,
            z_history,
            actions[future_start - n_history : future_start].unsqueeze(0),
            future_actions,
            device,
            hist_noise=args.infer_noise,
        )
        keep = min(n_future, frame_count - len(predictions))
        predictions.extend(predicted[:, index] for index in range(keep))
        z_history = torch.cat([z_history, predicted[:, :keep]], dim=1)[:, -n_history:]

    writer = None
    decoded_frames = 0
    with h5py.File(packed_path, "r") as packed:
        for step, prediction in enumerate(predictions):
            frame_index = n_history + step
            cameras = _load_cams_packed(packed, frame_index, render_config["downscale"], device)
            if not cameras:
                raise RuntimeError(f"no valid cameras at frame {frame_index}")
            rendered_gt = _render_latent(vae, seed_head, feat_cache, bg_bank, zgt_t[frame_index], cameras)
            rendered_pred = _render_latent(vae, seed_head, feat_cache, bg_bank, prediction[0], cameras)
            rows = []
            for label, images in (
                ("Ground truth", [(camera["rgb"].cpu().numpy() * 255).astype(np.uint8) for camera in cameras]),
                ("VAE reconstruction", rendered_gt),
                ("Predicted future", rendered_pred),
            ):
                labeled = []
                for camera_index, image in enumerate(images):
                    tile = image.copy()
                    cv2.putText(tile, f"cam{camera_index}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                    labeled.append(tile)
                row = np.concatenate(labeled, axis=1)
                cv2.putText(row, label, (8, row.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
                rows.append(row)
            grid = np.concatenate(rows, axis=0)
            banner = np.zeros((36, grid.shape[1], 3), dtype=np.uint8)
            cv2.putText(
                banner,
                f"Held-out episode {episode_id} | open-loop t={frame_index} | history={n_history}",
                (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            frame = np.concatenate([banner, grid], axis=0)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError("OpenCV could not initialize the MP4 writer")
            writer.write(frame[..., ::-1])
            decoded_frames += 1
    if writer is not None:
        writer.release()

    capture = cv2.VideoCapture(str(output))
    verified = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame is None or frame.size == 0:
            raise RuntimeError("video contains an unreadable frame")
        verified += 1
    capture.release()
    if verified != decoded_frames:
        raise RuntimeError(f"video verification read {verified} of {decoded_frames} frames")

    manifest = {
        "status": "complete",
        "video": str(output),
        "episode": episode_id,
        "frames": verified,
        "fps": args.fps,
        "history_frames": n_history,
        "inference_noise": args.infer_noise,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(device),
        "wall_seconds": time.time() - started,
        "video_bytes": output.stat().st_size,
        "video_sha256": sha256(output),
        "dynamics_checkpoint": str(checkpoint_path),
        "dynamics_epoch": int(checkpoint["epoch"]),
        "vae_checkpoint": str(vae_path),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
