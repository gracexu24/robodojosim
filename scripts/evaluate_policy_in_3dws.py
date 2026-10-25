#!/usr/bin/env python3
"""Roll a trained ACT or Diffusion Policy checkpoint through frozen 3DWS.

This is a learned-model evaluation.  It renders policy observations from the
predicted Gaussian scene and feeds policy commands back to the dynamics model;
it does not execute Isaac Sim physics and must not be reported as task success.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


CAMERA_NAMES = ("head", "left wrist", "right wrist")
HEADER = struct.Struct("!Q")


def euler_xyz_to_wxyz(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(euler, dtype=np.float64)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def wxyz_to_euler_xyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sin_pitch = float(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = max(math.sqrt(w * w + x * x + y * y + z * z), 1e-12)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def action14_to_state16(action: np.ndarray) -> np.ndarray:
    """Convert xyz + static-XYZ Euler + gripper into xyz + wxyz + gripper."""
    source = np.asarray(action, dtype=np.float64)
    if source.shape != (14,):
        raise ValueError(f"expected a 14D action, got {source.shape}")
    parts = []
    for offset in (0, 7):
        quaternion = euler_xyz_to_wxyz(source[offset + 3 : offset + 6])
        parts.extend((source[offset : offset + 3], quaternion, source[offset + 6 : offset + 7]))
    return np.concatenate(parts).astype(np.float32)


def state16_to_action14(state: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Convert xyz + wxyz + gripper into continuous static-XYZ Euler targets."""
    source = np.asarray(state, dtype=np.float64)
    if source.shape != (16,):
        raise ValueError(f"expected a 16D state, got {source.shape}")
    previous = None if reference is None else np.asarray(reference, dtype=np.float64)
    output = np.empty(14, dtype=np.float64)
    for arm, (source_offset, target_offset) in enumerate(((0, 0), (8, 7))):
        output[target_offset : target_offset + 3] = source[source_offset : source_offset + 3]
        quaternion = source[source_offset + 3 : source_offset + 7]
        quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
        euler = wxyz_to_euler_xyz(quaternion)
        if previous is not None:
            old = previous[target_offset + 3 : target_offset + 6]
            euler = old + (euler - old + np.pi) % (2 * np.pi) - np.pi
        output[target_offset + 3 : target_offset + 6] = euler
        output[target_offset + 6] = source[source_offset + 7]
    return output


def pose_matrix(state: np.ndarray, arm: int) -> np.ndarray:
    offset = 8 * arm
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = wxyz_to_matrix(state[offset + 3 : offset + 7])
    transform[:3, 3] = state[offset : offset + 3]
    return transform


def camera_rig(initial_state: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray):
    """Anchor the head camera and rigidly attach wrist cameras to command poses."""
    relatives = []
    for arm, camera_index in enumerate((1, 2)):
        relatives.append(np.linalg.inv(pose_matrix(initial_state, arm)) @ extrinsics[camera_index])
    return {
        "K": np.asarray(intrinsics, dtype=np.float32),
        "head": np.asarray(extrinsics[0], dtype=np.float64),
        "wrist_relatives": relatives,
    }


def cameras_for_state(rig: dict, state: np.ndarray) -> list[dict]:
    extrinsics = [rig["head"]]
    for arm in range(2):
        extrinsics.append(pose_matrix(state, arm) @ rig["wrist_relatives"][arm])
    cameras = []
    for index, cam2world in enumerate(extrinsics):
        K = np.asarray(rig["K"][index], dtype=np.float32)
        cameras.append(
            {
                "viewmat": np.linalg.inv(cam2world).astype(np.float32),
                "K": K,
                "W": int(round(2 * K[0, 2])),
                "H": int(round(2 * K[1, 2])),
            }
        )
    return cameras


def _recv_exact(connection: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("policy worker disconnected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class PolicyClient:
    def __init__(self, connection: socket.socket):
        self.connection = connection

    def request(self, value: dict) -> dict:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        self.connection.sendall(HEADER.pack(len(payload)) + payload)
        size = HEADER.unpack(_recv_exact(self.connection, HEADER.size))[0]
        response = pickle.loads(_recv_exact(self.connection, size))
        if not response.get("ok"):
            raise RuntimeError(response.get("traceback") or response.get("error"))
        return response

    def close(self) -> None:
        try:
            self.request({"command": "close"})
        finally:
            self.connection.close()


def start_policy_worker(args, output_dir: Path):
    socket_path = Path(
        f"/tmp/robodojosim-policy-{os.getpid()}-{time.time_ns()}.sock"
    )
    args.worker_socket = socket_path
    log_handle = (output_dir / "policy-worker.log").open("w")
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.policy_gpu),
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(args.repo_root / "src"),
            "HF_HOME": str(args.cache_root / "huggingface"),
            "TORCH_HOME": str(args.cache_root / "torch"),
            "XDG_CACHE_HOME": str(args.cache_root / "xdg"),
        }
    )
    command = [
        str(args.policy_python),
        str(args.repo_root / "scripts" / "serve_lerobot_policy_worker.py"),
        "--checkpoint",
        str(args.policy_checkpoint),
        "--socket",
        str(socket_path),
    ]
    if args.policy_extra_site_packages is not None:
        command.extend(
            ["--extra-site-packages", str(args.policy_extra_site_packages)]
        )
    process = subprocess.Popen(
        command,
        cwd=args.repo_root,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + args.worker_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_handle.flush()
            raise RuntimeError(
                f"policy worker exited with {process.returncode}; see {output_dir / 'policy-worker.log'}"
            )
        if socket_path.exists():
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                connection.connect(str(socket_path))
                return process, log_handle, PolicyClient(connection)
            except ConnectionRefusedError:
                connection.close()
        time.sleep(0.1)
    process.terminate()
    raise TimeoutError(f"policy worker did not become ready within {args.worker_timeout}s")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(path: Path) -> Path:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def select_episode_ids(ids: list[int], count: int) -> list[int]:
    if count <= 0 or count >= len(ids):
        return ids
    indices = np.linspace(0, len(ids) - 1, count).round().astype(int)
    return [ids[index] for index in indices]


def make_mosaic(images: list[np.ndarray], metadata: list[str]) -> np.ndarray:
    import cv2

    height, width = images[0].shape[:2]
    resized = [cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) for image in images]
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    y = 48
    for line in metadata:
        cv2.putText(panel, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (25, 25, 25), 2, cv2.LINE_AA)
        y += 38
    for image, name in zip(resized, CAMERA_NAMES):
        cv2.putText(image, name, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, name, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 1, cv2.LINE_AA)
    top = np.concatenate([resized[0], resized[1]], axis=1)
    bottom = np.concatenate([resized[2], panel], axis=1)
    return np.concatenate([top, bottom], axis=0)


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    import cv2

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(path))
    readable = capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frames)
    capture.release()
    if not readable:
        raise RuntimeError(f"video verification failed: {path}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, required=True)
    result.add_argument("--threedws-root", type=Path, required=True)
    result.add_argument("--dynamics-checkpoint", type=Path, required=True)
    result.add_argument("--latent-dir", type=Path, required=True)
    result.add_argument("--packed-dir", type=Path, action="append", dest="packed_dirs", required=True)
    result.add_argument("--policy-checkpoint", type=Path, required=True)
    result.add_argument("--policy-python", type=Path, required=True)
    result.add_argument("--policy-extra-site-packages", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--cache-root", type=Path, default=Path("/data/gracexu/cache/3dws"))
    result.add_argument("--policy-gpu", type=int, default=1)
    result.add_argument("--episodes", type=int, default=10)
    result.add_argument("--steps", type=int, default=16)
    result.add_argument("--seed-frame", type=int, default=0)
    result.add_argument("--fps", type=float, default=10.0)
    result.add_argument("--seed", type=int, default=20261025)
    result.add_argument("--worker-timeout", type=float, default=180.0)
    result.add_argument("--bf16", action="store_true")
    return result


def main(argv=None) -> int:
    import cv2
    import torch

    from serve_3dws_bottle_viser import FrozenWorldModel, LatentEpisodes, clamp_action

    args = parser().parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    args.threedws_root = args.threedws_root.resolve()
    args.dynamics_checkpoint = args.dynamics_checkpoint.resolve()
    args.latent_dir = args.latent_dir.resolve()
    args.packed_dirs = [path.resolve() for path in args.packed_dirs]
    args.policy_checkpoint = args.policy_checkpoint.resolve()
    args.policy_python = args.policy_python.resolve()
    if args.policy_extra_site_packages is not None:
        args.policy_extra_site_packages = args.policy_extra_site_packages.resolve()
    args.cache_root = args.cache_root.resolve()
    output_dir = prepare_output(args.output_dir.resolve())
    started = time.time()
    config = json.loads((args.policy_checkpoint / "config.json").read_text())
    policy_family = str(config["type"])
    policy_id = args.policy_checkpoint.parents[2].name

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 3DWS policy evaluation")

    episodes = LatentEpisodes(args.latent_dir, args.packed_dirs, "cuda")
    selected_ids = select_episode_ids(episodes.ids, args.episodes)
    world = FrozenWorldModel(args.threedws_root, args.dynamics_checkpoint, "cuda", args.bf16)
    process = log_handle = client = None
    all_rows = []
    video_paths = []
    images = []
    try:
        process, log_handle, client = start_policy_worker(args, output_dir)
        execution = None
        for episode_id in selected_ids:
            client.request({"command": "reset"})
            latent, recorded_action, intrinsics, extrinsics = episodes.load(episode_id)
            if world.model is None:
                world.build(latent[0], recorded_action.shape[1])
            start = min(max(0, args.seed_frame), len(latent) - world.n_history - 1)
            stop = start + world.n_history
            latent_history = latent[start:stop][None].clone()
            action_history = torch.stack(
                [world.normalize_action(item) for item in recorded_action[start:stop]], dim=0
            )[None]
            state = action14_to_state16(recorded_action[stop - 1])
            reference_action = recorded_action[stop - 1].copy()
            rig = camera_rig(state, intrinsics[stop - 1], extrinsics[stop - 1])
            frames = []
            for step in range(args.steps):
                decode_started = time.perf_counter()
                gaussians = world.decode(latent_history[0, -1])
                if gaussians is None:
                    raise RuntimeError(f"decoder produced no Gaussians for episode {episode_id} step {step}")
                cameras = cameras_for_state(rig, state)
                renders = [world.render(gaussians, camera, camera["W"], camera["H"]) for camera in cameras]
                decode_render_ms = 1000.0 * (time.perf_counter() - decode_started)
                policy_started = time.perf_counter()
                response = client.request(
                    {"command": "act", "state": state, "images": renders}
                )
                policy_ms = 1000.0 * (time.perf_counter() - policy_started)
                execution = response["execution"]
                raw_state = np.asarray(response["action"], dtype=np.float32)
                raw_action = state16_to_action14(raw_state, reference_action)
                action = clamp_action(raw_action, episodes.action_min, episodes.action_max)
                clamped = int(np.count_nonzero(np.abs(action - raw_action) > 1e-7))
                state = action14_to_state16(action)
                reference_action = action.copy()
                next_action = world.normalize_action(action)[None, None]
                dynamics_started = time.perf_counter()
                predicted = world.step(latent_history, action_history, next_action)
                dynamics_ms = 1000.0 * (time.perf_counter() - dynamics_started)
                latent_history = torch.cat([latent_history[:, 1:], predicted], dim=1)
                action_history = torch.cat([action_history[:, 1:], next_action], dim=1)
                gaussian_count = int(gaussians["xyz"].shape[0])
                row = {
                    "policy": policy_family,
                    "episode": episode_id,
                    "seed_frame": start,
                    "step": step,
                    "gaussians": gaussian_count,
                    "clamped_action_values": clamped,
                    "policy_ms": policy_ms,
                    "dynamics_ms": dynamics_ms,
                    "decode_render_ms": decode_render_ms,
                    **{f"action_{index}": float(value) for index, value in enumerate(action)},
                }
                if not all(math.isfinite(float(value)) for value in row.values() if isinstance(value, (int, float))):
                    raise RuntimeError(f"non-finite rollout metric: {row}")
                all_rows.append(row)
                metadata = [
                    f"{policy_family.upper()} in learned 3DWS",
                    f"held-out episode {episode_id}",
                    f"predicted step {step + 1}/{args.steps}",
                    f"Gaussians: {gaussian_count:,}",
                    f"policy: {policy_ms:.0f} ms",
                    f"dynamics: {dynamics_ms:.0f} ms",
                    f"decode + 3 renders: {decode_render_ms:.0f} ms",
                    f"clamped values: {clamped}",
                    "Model rollout; not Isaac task success",
                ]
                frames.append(make_mosaic(renders, metadata))
            episode_dir = output_dir / f"episode_{episode_id:06d}"
            episode_dir.mkdir()
            video_path = episode_dir / f"{policy_family}_3dws_rollout.mp4"
            write_video(video_path, frames, args.fps)
            video_paths.append(video_path)
            for label, index in (("early", 0), ("mid", len(frames) // 2), ("final", len(frames) - 1)):
                path = episode_dir / f"{label}.png"
                cv2.imwrite(str(path), cv2.cvtColor(frames[index], cv2.COLOR_RGB2BGR))
                if cv2.imread(str(path)) is None:
                    raise RuntimeError(f"could not verify {path}")
                images.append(path)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        if process is not None:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
        if log_handle is not None:
            log_handle.close()
        socket_path = getattr(args, "worker_socket", None)
        if socket_path is not None and socket_path.exists():
            socket_path.unlink()

    fieldnames = list(all_rows[0])
    with (output_dir / "rollout_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "status": "complete",
        "evaluation_kind": "closed_loop_policy_in_learned_3dws",
        "not_task_success_rate": True,
        "state_assumption": "the prior clamped Cartesian command is used as the next 16D policy state",
        "camera_assumption": "head camera fixed; wrist cameras follow a rigid seed-frame EE-to-camera transform",
        "policy_id": policy_id,
        "policy_family": policy_family,
        "policy_execution": execution,
        "policy_checkpoint": str(args.policy_checkpoint),
        "policy_model_sha256": sha256(args.policy_checkpoint / "model.safetensors"),
        "dynamics_checkpoint": str(args.dynamics_checkpoint),
        "dynamics_checkpoint_sha256": sha256(args.dynamics_checkpoint),
        "world_gpu": torch.cuda.get_device_name(0),
        "policy_gpu_physical_index": args.policy_gpu,
        "selected_episodes": selected_ids,
        "steps_per_episode": args.steps,
        "world_model_trained_ar_horizon": 16,
        "videos": [str(path.relative_to(output_dir)) for path in video_paths],
        "representative_images": [str(path.relative_to(output_dir)) for path in images],
        "total_steps": len(all_rows),
        "total_clamped_action_values": sum(row["clamped_action_values"] for row in all_rows),
        "mean_policy_ms": float(np.mean([row["policy_ms"] for row in all_rows])),
        "mean_dynamics_ms": float(np.mean([row["dynamics_ms"] for row in all_rows])),
        "mean_decode_render_ms": float(np.mean([row["decode_render_ms"] for row in all_rows])),
        "wall_seconds": time.time() - started,
    }
    (output_dir / "evaluation_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
