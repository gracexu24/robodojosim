#!/usr/bin/env python3
"""Serve one LeRobot policy checkpoint over a private Unix-domain socket.

The worker intentionally runs in the LeRobot environment while the caller runs
the 3DWS decoder and dynamics model in their separate environment.  The socket
protocol is local-only and exists for the lifetime of one evaluation process.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import sys
import traceback
from pathlib import Path

import numpy as np

from robodojosim.inference import CAMERAS, LeRobotPolicyRuntime


HEADER = struct.Struct("!Q")


def _recv_exact(connection: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("policy client disconnected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive(connection: socket.socket):
    size = HEADER.unpack(_recv_exact(connection, HEADER.size))[0]
    return pickle.loads(_recv_exact(connection, size))


def send(connection: socket.socket, value) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    connection.sendall(HEADER.pack(len(payload)))
    connection.sendall(payload)


def observation(payload: dict) -> dict:
    state = np.asarray(payload["state"], dtype=np.float32)
    if state.shape != (16,):
        raise ValueError(f"expected a 16D state, got {state.shape}")
    images = payload["images"]
    if len(images) != len(CAMERAS):
        raise ValueError(f"expected {len(CAMERAS)} images, got {len(images)}")
    return {
        "state": {
            "left_ee_pose": state[0:7],
            "left_ee_joint_state": state[7:8],
            "right_ee_pose": state[8:15],
            "right_ee_joint_state": state[15:16],
        },
        "vision": {
            camera: {"color": np.asarray(image, dtype=np.uint8)}
            for camera, image in zip(CAMERAS, images)
        },
    }


def flatten_action(action: dict) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(action["left_ee_pose"], dtype=np.float32),
            np.asarray(action["left_ee_joint_state"], dtype=np.float32),
            np.asarray(action["right_ee_pose"], dtype=np.float32),
            np.asarray(action["right_ee_joint_state"], dtype=np.float32),
        ]
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--socket", type=Path, required=True)
    result.add_argument("--device", default="cuda")
    result.add_argument(
        "--extra-site-packages",
        type=Path,
        help="append a LeRobot environment's site-packages after the GPU environment",
    )
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.extra_site_packages is not None:
        sys.path.append(str(args.extra_site_packages.resolve()))
    runtime = LeRobotPolicyRuntime(args.checkpoint, device=args.device)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket))
    server.listen(1)
    print(f"policy worker ready: {args.socket}", flush=True)
    connection, _ = server.accept()
    with connection:
        while True:
            try:
                request = receive(connection)
                command = request.get("command")
                if command == "close":
                    send(connection, {"ok": True})
                    break
                if command == "reset":
                    runtime.reset()
                    send(connection, {"ok": True})
                    continue
                if command != "act":
                    raise ValueError(f"unknown policy command: {command!r}")
                runtime.update(observation(request))
                action = flatten_action(runtime.get_action())
                send(
                    connection,
                    {
                        "ok": True,
                        "action": action,
                        "execution": runtime._policy_execution,
                    },
                )
            except EOFError:
                break
            except Exception as exc:
                send(
                    connection,
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                )
    server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
