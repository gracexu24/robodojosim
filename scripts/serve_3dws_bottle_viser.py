#!/usr/bin/env python3
"""Interactive Viser teleoperator for a frozen two-bottle 3DWS world model.

The viewer is intentionally model-only: it edits the 14D absolute Cartesian
action target and autoregressively predicts the next latent scene.  It never
connects to Isaac Sim, robot middleware, or physical hardware.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import transforms3d


ACTION_NAMES = (
    "left_x",
    "left_y",
    "left_z",
    "left_roll",
    "left_pitch",
    "left_yaw",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_roll",
    "right_pitch",
    "right_yaw",
    "right_gripper",
)

KEYBOARD_DELTAS = {
    "KeyA": (0, -1.0),
    "KeyD": (0, 1.0),
    "KeyQ": (1, 1.0),
    "KeyE": (1, -1.0),
    "KeyW": (2, 1.0),
    "KeyS": (2, -1.0),
    "KeyJ": (7, -1.0),
    "KeyL": (7, 1.0),
    "KeyU": (8, 1.0),
    "KeyO": (8, -1.0),
    "KeyI": (9, 1.0),
    "KeyK": (9, -1.0),
}

POINT_CLOUD_MODE = "Point cloud (interactive 3D)"
GAUSSIAN_MODE = "Exact Gaussian render (fixed camera)"


def apply_keyboard_action(
    action: np.ndarray,
    keys: set[str],
    step: float,
    lower: np.ndarray,
    upper: np.ndarray,
    clamp: bool = True,
) -> np.ndarray:
    """Apply simultaneous held-key Cartesian deltas to a 14D action."""
    out = np.asarray(action, dtype=np.float64).copy()
    for key in keys:
        update = KEYBOARD_DELTAS.get(key)
        if update is not None:
            index, direction = update
            out[index] += float(step) * direction
    return clamp_action(out, lower, upper) if clamp else out


KEYBOARD_CLIENT_SCRIPT = r"""
<script id="three-dws-keyboard-client">
(() => {
  const controlled = new Set([
    "KeyA", "KeyD", "KeyQ", "KeyE", "KeyW", "KeyS",
    "KeyJ", "KeyL", "KeyU", "KeyO", "KeyI", "KeyK",
    "KeyG", "KeyH", "KeyR", "Space", "Period"
  ]);
  const movement = new Set([
    "KeyA", "KeyD", "KeyQ", "KeyE", "KeyW", "KeyS",
    "KeyJ", "KeyL", "KeyU", "KeyO", "KeyI", "KeyK"
  ]);
  const held = new Set();
  let socket = null;
  let reconnectTimer = null;
  const send = (payload) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(payload));
    }
  };
  const sendHeld = () => send({type: "held", keys: Array.from(held)});
  const connect = () => {
    socket = new WebSocket(`ws://${location.hostname}:__KEYBOARD_PORT__`);
    socket.onopen = sendHeld;
    socket.onclose = () => {
      clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connect, 1000);
    };
  };
  const isForm = (target) => target && (
    target.tagName === "INPUT" || target.tagName === "TEXTAREA" ||
    target.tagName === "SELECT" || target.isContentEditable
  );
  document.addEventListener("keydown", (event) => {
    if (!controlled.has(event.code) || isForm(event.target)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    if (movement.has(event.code)) {
      if (!held.has(event.code)) {
        held.add(event.code);
        send({type: "tap", key: event.code});
        sendHeld();
      }
    } else if (!event.repeat) {
      send({type: "tap", key: event.code});
    }
  }, true);
  document.addEventListener("keyup", (event) => {
    if (!controlled.has(event.code) || isForm(event.target)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    if (held.delete(event.code)) sendHeld();
  }, true);
  window.addEventListener("blur", () => {
    held.clear();
    sendHeld();
  });
  connect();
})();
</script>
"""


def install_keyboard_client(viser_module, build_dir: Path, keyboard_port: int) -> None:
    """Serve a viewer-specific Viser client with a small keyboard bridge."""
    source_root = Path(viser_module.__file__).resolve().parent / "client" / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    for source in source_root.iterdir():
        target = build_dir / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        elif source.name != "index.html":
            shutil.copy2(source, target)
    html = (source_root / "index.html").read_text()
    script = KEYBOARD_CLIENT_SCRIPT.replace("__KEYBOARD_PORT__", str(keyboard_port))
    if "</body>" not in html:
        raise RuntimeError("Viser client index is missing </body>")
    (build_dir / "index.html").write_text(html.replace("</body>", script + "\n</body>"))
    # ViserServer resolves its HTTP root from viser._viser.__file__ at runtime.
    fake_module_root = build_dir.parent.parent
    import viser._viser as viser_server_module

    viser_server_module.__file__ = str(fake_module_root / "_viser.py")


class KeyboardBridge:
    """Receive key state from the browser on a dedicated loopback websocket."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.lock = threading.Lock()
        self.held: set[str] = set()
        self.taps: list[str] = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        from websockets.sync.server import serve

        def handler(websocket) -> None:
            try:
                for raw in websocket:
                    message = json.loads(raw)
                    with self.lock:
                        if message.get("type") == "held":
                            self.held = {
                                key for key in message.get("keys", []) if key in KEYBOARD_DELTAS
                            }
                        elif message.get("type") == "tap":
                            self.taps.append(str(message.get("key", "")))
            finally:
                with self.lock:
                    self.held.clear()

        with serve(handler, self.host, self.port) as server:
            print(f"[keyboard] websocket ws://{self.host}:{self.port}", flush=True)
            server.serve_forever()

    def snapshot(self) -> tuple[set[str], list[str]]:
        with self.lock:
            held = set(self.held)
            taps = list(self.taps)
            self.taps.clear()
        return held, taps


def action_to_poses(action: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Convert the 14D [left pose+grip, right pose+grip] action to two poses."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (14,):
        raise ValueError(f"expected a 14D action, got {action.shape}")
    out = []
    for arm in range(2):
        base = 7 * arm
        pos = action[base : base + 3].copy()
        wxyz = np.asarray(
            transforms3d.euler.euler2quat(*action[base + 3 : base + 6], axes="sxyz"),
            dtype=np.float64,
        )
        out.append((pos, wxyz, float(action[base + 6])))
    return out


def update_action_pose(
    action: np.ndarray,
    arm: int,
    position: np.ndarray,
    wxyz: np.ndarray,
    gripper: float,
) -> np.ndarray:
    """Return a copy of ``action`` with one arm replaced by a Viser gizmo pose."""
    out = np.asarray(action, dtype=np.float64).copy()
    base = 7 * arm
    out[base : base + 3] = np.asarray(position, dtype=np.float64)
    out[base + 3 : base + 6] = transforms3d.euler.quat2euler(
        np.asarray(wxyz, dtype=np.float64), axes="sxyz"
    )
    out[base + 6] = float(gripper)
    return out


def clamp_action(action: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Clamp a 14D target to the action range observed in the training corpus."""
    action = np.asarray(action, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if action.shape != (14,) or lower.shape != (14,) or upper.shape != (14,):
        raise ValueError("action and limits must all be 14D")
    return np.clip(action, lower, upper)


def evenly_sample(n: int, maximum: int) -> np.ndarray:
    """Deterministically choose at most ``maximum`` indices from ``range(n)``."""
    if n <= 0 or maximum <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n <= maximum:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, maximum, dtype=np.int64)


def bbox_segments(bbox: np.ndarray) -> np.ndarray:
    """Return twelve line segments for a [xmin,xmax,ymin,ymax,zmin,zmax] box."""
    x0, x1, y0, y1, z0, z1 = map(float, bbox)
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ],
        dtype=np.int64,
    )
    return np.stack([corners[edges[:, 0]], corners[edges[:, 1]]], axis=1)


class LatentEpisodes:
    def __init__(self, latent_dir: Path, packed_dirs: list[Path], device: str):
        self.files = sorted(
            glob.glob(str(latent_dir / "episode_*_latents.h5")),
            key=lambda p: int(re.search(r"episode_(\d+)_", p).group(1)),
        )
        if not self.files:
            raise FileNotFoundError(f"no latent episodes under {latent_dir}")
        self.ids = [int(re.search(r"episode_(\d+)_", p).group(1)) for p in self.files]
        self.by_id = dict(zip(self.ids, self.files))
        self.packed_dirs = [str(path) for path in packed_dirs]
        self.device = device
        self.lengths: dict[int, int] = {}
        mins = np.full(14, np.inf, dtype=np.float64)
        maxs = np.full(14, -np.inf, dtype=np.float64)
        with h5py.File(self.files[0], "r") as handle:
            self.bbox = np.asarray(handle.attrs["bbox"], dtype=np.float32)
        for path, episode_id in zip(self.files, self.ids):
            with h5py.File(path, "r") as handle:
                self.lengths[episode_id] = int(handle["z"].shape[0])
                action = handle["action"][:]
                mins = np.minimum(mins, action.min(axis=0))
                maxs = np.maximum(maxs, action.max(axis=0))
        self.action_min = mins
        self.action_max = maxs

    def load(self, episode_id: int):
        with h5py.File(self.by_id[episode_id], "r") as handle:
            latent = torch.from_numpy(handle["z"][:, 0].astype(np.float32)).to(self.device)
            action = handle["action"][:].astype(np.float64)
            intrinsics = handle["K"][:]
            extrinsics = handle["E"][:]
        return latent, action, intrinsics, extrinsics


class FrozenWorldModel:
    """Best-checkpoint VAE decoder, gaussian head, and action-conditioned dynamics."""

    def __init__(self, threedws_root: Path, checkpoint: Path, device: str, bf16: bool):
        sys.path.insert(0, str(threedws_root))
        from models.gsplat_render_loss import render_gaussians
        from models.seed_upsample import get_seed_indices_from_occupancy
        from scripts.train_dynamics import CoarseDynamics3D, load_vae, predict_chunk_inf

        self.render_gaussians = render_gaussians
        self.get_seed_indices_from_occupancy = get_seed_indices_from_occupancy
        self.CoarseDynamics3D = CoarseDynamics3D
        self.predict_chunk_inf = predict_chunk_inf
        self.device = device
        self.bf16 = bf16
        self.checkpoint_path = checkpoint
        self.checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)
        self.args = self.checkpoint["args"]
        self.vae, self.seed_head, self.bg_bank, self.feat_cache, _ = load_vae(
            self.args["vae_ckpt"], device
        )
        self.action_mean = np.asarray(self.checkpoint["action_mean"], dtype=np.float64)
        self.action_std = np.maximum(
            np.asarray(self.checkpoint["action_std"], dtype=np.float64), 1e-9
        )
        self.n_history = int(self.args["n_history"])
        self.hist_noise = float(self.args.get("infer_hist_noise", 0.1))
        split = int(self.vae._ck_args.get("latent_split", 0) or 0)
        self.groups = None
        self.latent_split = split
        self.model = None
        self.bg = None
        if self.bg_bank is not None:
            with torch.no_grad():
                bg = self.bg_bank.gaussians()
                count = bg["xyz"].shape[0]
                self.bg = {
                    "xyz": bg["xyz"].float(),
                    "scale": bg["scale"].float(),
                    "opacity": bg["opacity"].float(),
                    "rgb01": (bg["rgb"].float() + 1.0) / 2.0,
                    "quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
                    .expand(count, 4)
                    .contiguous(),
                    "sh1": torch.zeros(count, 3, 3, device=device),
                }

    def build(self, example: torch.Tensor, action_dim: int) -> None:
        channels = int(example.shape[0])
        self.groups = (
            [channels - self.latent_split, self.latent_split]
            if self.latent_split > 0
            else None
        )
        model = self.CoarseDynamics3D(
            latent_channels=channels,
            action_dim=action_dim,
            n_history=self.n_history,
            n_future=1,
            width=int(self.args["dim"]),
            action_emb_dim=128,
            n_levels=int(self.args.get("n_levels", 2)),
            film=bool(self.args.get("film", False)),
        ).to(self.device)
        state = {
            key.replace("_orig_mod.", ""): value
            for key, value in self.checkpoint["model_state_dict"].items()
        }
        model.load_state_dict(state)
        self.model = model.eval()

    def normalize_action(self, action: np.ndarray) -> torch.Tensor:
        normalized = (
            (np.asarray(action, dtype=np.float64) - self.action_mean) / self.action_std
        ).astype(np.float32)
        return torch.from_numpy(normalized).to(self.device)

    @torch.no_grad()
    def step(
        self,
        latent_history: torch.Tensor,
        action_history: torch.Tensor,
        next_action: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.predict_chunk_inf(
                self.model,
                latent_history,
                action_history,
                next_action,
                self.device,
                hist_noise=self.hist_noise,
            ).float()

    @torch.no_grad()
    def decode(self, latent: torch.Tensor):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            output = self.vae.decode(latent.unsqueeze(0), occ_mask_0=None, use_gt_mask=False)
        features = self.feat_cache.get("feat_2")
        if features is None:
            return None
        seeds = self.get_seed_indices_from_occupancy(output["scene_occ_2"] > 0.0)
        if seeds.shape[0] == 0:
            return None
        bmin = self.vae.bounds_min_world
        nx, ny, nz = self.vae.grid_shapes[-1]
        voxel_size = (self.vae.bounds_max_world - bmin) / torch.tensor(
            [nx, ny, nz], device=bmin.device, dtype=bmin.dtype
        )
        decoded = self.seed_head(features.float(), seeds, bmin, voxel_size)[1]
        count = decoded["xyz"].reshape(-1, 3).shape[0]
        return {
            "xyz": decoded["xyz"].reshape(-1, 3).float(),
            "scale": decoded["scale"].reshape(-1, 3).float(),
            "opacity": decoded["opacity"].reshape(-1).float(),
            "rgb01": ((decoded["rgb"].reshape(-1, 3).float().clamp(-1, 1) + 1.0) / 2.0),
            "quat": decoded.get(
                "quat",
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
                .expand(count, 4)
                .contiguous(),
            ).reshape(-1, 4).float(),
            "sh1": decoded.get(
                "sh1", torch.zeros(count, 3, 3, device=self.device)
            ).reshape(-1, 3, 3).float(),
        }

    @torch.no_grad()
    def render(self, foreground, camera, width: int, height: int) -> np.ndarray:
        parts = [part for part in (foreground, self.bg) if part is not None]
        combined = {
            key: torch.cat([part[key] for part in parts])
            for key in ("xyz", "scale", "opacity", "rgb01", "quat", "sh1")
        }
        viewmat = torch.as_tensor(
            camera["viewmat"], device=self.device, dtype=torch.float32
        )[None]
        intrinsics = torch.as_tensor(
            np.ascontiguousarray(camera["K"]), device=self.device, dtype=torch.float32
        )[None]
        rendered, _ = self.render_gaussians(
            combined["xyz"],
            combined["scale"],
            combined["opacity"],
            combined["rgb01"],
            viewmat,
            intrinsics,
            width,
            height,
            near_plane=0.05,
            far_plane=10.0,
            quats=combined["quat"],
            sh1=combined["sh1"],
        )
        return (rendered[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


class BottleViewer:
    def __init__(self, args):
        import viser
        from scripts.train_dynamics import _find_packed, _load_cams_packed

        self.args = args
        self.viser = viser
        self.find_packed = _find_packed
        self.load_cams_packed = _load_cams_packed
        self.lock = threading.RLock()
        self.syncing_controls = False
        self.pending_steps = 0
        self.pending_reset = True
        self.running = True
        self.last_step_seconds = 0.0
        self.last_keyboard_step = time.monotonic()
        self.world = FrozenWorldModel(
            args.threedws_root, args.dynamics_checkpoint, "cuda", args.bf16
        )
        self.episodes = LatentEpisodes(args.latent_dir, args.packed_dirs, "cuda")
        install_keyboard_client(viser, args.client_build_dir, args.keyboard_port)
        self.keyboard = KeyboardBridge(args.host, args.keyboard_port)
        self.server = viser.ViserServer(host=args.host, port=args.port, label="3DWS bottle world model")
        self._build_gui()
        self._build_scene()
        self.reset()
        self.pending_reset = False

    def _build_gui(self) -> None:
        gui = self.server.gui
        gui.add_markdown(
            "## Two-bottle 3DWS teleop\n"
            "**Latent imagination only.** This predicts future 3D scenes; it does not command Isaac Sim or a robot. "
            "Click the 3D canvas, then use the keyboard controls below."
        )
        gui.add_markdown(
            "**Left arm:** `A/D` x−/x+, `Q/E` y+/y−, `W/S` z+/z−, `G` gripper  \n"
            "**Right arm:** `J/L` x−/x+, `U/O` y+/y−, `I/K` z+/z−, `H` gripper  \n"
            "`R` reset · `Space` play/pause recorded sequence · `.` one prediction"
        )
        with gui.add_folder("Seed / rollout", expand_by_default=True):
            initial_episode = str(
                self.args.episode
                if self.args.episode in self.episodes.ids
                else self.episodes.ids[0]
            )
            self.episode_widget = gui.add_dropdown(
                "held-out episode",
                options=[str(x) for x in self.episodes.ids],
                initial_value=initial_episode,
            )
            self.frame_widget = gui.add_number(
                "seed start frame", initial_value=self.args.frame, min=0, step=1
            )
            self.reset_button = gui.add_button("Reset from held-out history", color="red")
            self.action_source = gui.add_dropdown(
                "action source",
                options=("Keyboard teleoperation", "Recorded validation actions"),
                initial_value="Keyboard teleoperation",
            )
            self.clamp_widget = gui.add_checkbox("Clamp to held-out action range", True)
            self.step_button = gui.add_button("Predict one future", color="blue")
            self.play_recorded_button = gui.add_button(
                "Play recorded held-out sequence", color="green"
            )
            self.pause_button = gui.add_button("Pause", color="gray")
            self.play_widget = gui.add_checkbox("Autoplay predictions", False)
            self.hz_widget = gui.add_slider("autoplay Hz", 1.0, 10.0, 1.0, 5.0)
        with gui.add_folder("Action target", expand_by_default=True):
            self.step_size_widget = gui.add_slider(
                "keyboard Cartesian step (m)", 0.001, 0.02, 0.001, 0.005
            )
            self.keyboard_state = gui.add_text("held keys", "none", disabled=True)
            self.action_text = gui.add_text("14D target", "", disabled=True)
        with gui.add_folder("Rendering", expand_by_default=False):
            self.display_mode_widget = gui.add_dropdown(
                "display mode",
                options=(POINT_CLOUD_MODE, GAUSSIAN_MODE),
                initial_value=POINT_CLOUD_MODE,
            )
            self.reset_camera_button = gui.add_button("Frame predicted world")
            self.show_bg_widget = gui.add_checkbox("Show background points", True)
            self.point_size_widget = gui.add_slider(
                "foreground point size", 0.0005, 0.01, 0.0005, 0.0025
            )
            self.bg_size_widget = gui.add_slider(
                "background point size", 0.0005, 0.01, 0.0005, 0.002
            )
            self.camera_widget = gui.add_dropdown(
                "exact render camera", options=("cam0", "cam1", "cam2", "cam3"), initial_value="cam0"
            )
            self.render_image = gui.add_image(
                np.zeros((240, 320, 3), dtype=np.uint8), label="Exact gaussian render", format="jpeg"
            )
        self.status_widget = gui.add_text("status", "loading", disabled=True)

        @self.reset_button.on_click
        def _(_event):
            with self.lock:
                self.pending_reset = True

        @self.step_button.on_click
        def _(_event):
            with self.lock:
                self.pending_steps += 1

        @self.play_recorded_button.on_click
        def _(_event):
            with self.lock:
                self.action_source.value = "Recorded validation actions"
                self.play_widget.value = True

        @self.pause_button.on_click
        def _(_event):
            with self.lock:
                self.play_widget.value = False

        @self.display_mode_widget.on_update
        def _(_event):
            with self.lock:
                self._apply_display_mode()

        @self.show_bg_widget.on_update
        def _(_event):
            self._apply_display_mode()

        @self.point_size_widget.on_update
        def _(_event):
            self.foreground_handle.point_size = float(self.point_size_widget.value)

        @self.bg_size_widget.on_update
        def _(_event):
            self.background_handle.point_size = float(self.bg_size_widget.value)

        @self.camera_widget.on_update
        def _(_event):
            with self.lock:
                if getattr(self, "last_gaussians", None) is not None:
                    self._update_exact_render(self.last_gaussians)

        @self.reset_camera_button.on_click
        def _(_event):
            for client in self.server.get_clients().values():
                self._set_client_camera(client)

    def _build_scene(self) -> None:
        self.grid_handle = self.server.scene.add_grid(
            "/table_grid",
            width=1.0,
            height=1.0,
            cell_size=0.05,
            section_size=0.25,
            plane="xy",
            position=(0.0, 0.0, float(self.episodes.bbox[4])),
        )
        self.bounds_handle = self.server.scene.add_line_segments(
            "/training_bounds",
            points=bbox_segments(self.episodes.bbox),
            colors=np.array((30, 200, 240), dtype=np.uint8),
            line_width=2.0,
        )
        self.foreground_handle = self.server.scene.add_point_cloud(
            "/predicted_world/foreground",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=np.zeros((1, 3), dtype=np.uint8),
            point_size=float(self.point_size_widget.value),
            point_shape="circle",
            precision="float32",
        )
        self.background_handle = self.server.scene.add_point_cloud(
            "/predicted_world/background",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=np.zeros((1, 3), dtype=np.uint8),
            point_size=float(self.bg_size_widget.value),
            point_shape="circle",
            precision="float32",
        )
        if self.world.bg is not None:
            xyz = self.world.bg["xyz"].detach().cpu().numpy()
            rgb = (self.world.bg["rgb01"].clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
            indices = evenly_sample(len(xyz), self.args.max_background_points)
            self.background_handle.points = xyz[indices]
            self.background_handle.colors = rgb[indices]
        self.action_frames = []
        for arm, label in enumerate(("left", "right")):
            handle = self.server.scene.add_frame(
                f"/action_targets/{label}",
                axes_length=0.07,
                axes_radius=0.0025,
            )
            self.action_frames.append(handle)

        self.scene_center = np.array(
            [
                np.mean(self.episodes.bbox[0:2]),
                np.mean(self.episodes.bbox[2:4]),
                np.mean(self.episodes.bbox[4:6]),
            ]
        )

        @self.server.on_client_connect
        def _(client):
            self._set_client_camera(client)

    def _set_client_camera(self, client) -> None:
        # Set position before look-at, and publish the camera properties atomically.
        # Otherwise Viser can preserve the old view direction and frame an empty area.
        with client.atomic():
            client.camera.position = self.scene_center + np.array([0.7, -0.9, 0.55])
            client.camera.look_at = self.scene_center
            client.camera.up_direction = np.array([0.0, 0.0, 1.0])

    def _sync_action_frames(self, action: np.ndarray) -> None:
        self.syncing_controls = True
        try:
            for arm, (position, wxyz, gripper) in enumerate(action_to_poses(action)):
                self.action_frames[arm].position = position
                self.action_frames[arm].wxyz = wxyz
        finally:
            self.syncing_controls = False
        self._update_action_text()

    def _update_action_text(self) -> None:
        values = ", ".join(f"{value:.3f}" for value in self.desired_action)
        self.action_text.value = f"[{values}]"

    def _fixed_cameras(self):
        packed_path = self.find_packed(self.episodes.packed_dirs, self.episode_id)
        frame = self.seed_frame + self.world.n_history - 1
        if packed_path is not None:
            with h5py.File(packed_path, "r") as handle:
                cameras = self.load_cams_packed(
                    handle, min(frame, handle["render/camera_0/viewmat"].shape[0] - 1), self.args.cam_downscale, "cuda"
                )
            if cameras:
                return [
                    {
                        "viewmat": camera["viewmat"].cpu().numpy(),
                        "K": camera["K"].cpu().numpy(),
                        "W": int(camera["W"]),
                        "H": int(camera["H"]),
                    }
                    for camera in cameras
                ]
        frame = min(frame, len(self.intrinsics) - 1)
        cameras = []
        for camera_index in range(self.intrinsics.shape[1]):
            intrinsics = self.intrinsics[frame, camera_index].astype(np.float32).copy()
            width = int(round(2 * intrinsics[0, 2]))
            height = int(round(2 * intrinsics[1, 2]))
            if self.args.cam_downscale > 1:
                intrinsics[:2] /= self.args.cam_downscale
                width //= self.args.cam_downscale
                height //= self.args.cam_downscale
            cameras.append(
                {
                    "viewmat": np.linalg.inv(self.extrinsics[frame, camera_index]).astype(np.float32),
                    "K": intrinsics,
                    "W": width,
                    "H": height,
                }
            )
        return cameras

    def reset(self) -> None:
        self.episode_id = int(self.episode_widget.value)
        latent, action, intrinsics, extrinsics = self.episodes.load(self.episode_id)
        maximum_start = max(0, len(latent) - self.world.n_history - 1)
        self.seed_frame = min(max(0, int(self.frame_widget.value)), maximum_start)
        self.frame_widget.value = self.seed_frame
        if self.world.model is None:
            self.world.build(latent[0], action.shape[1])
        start = self.seed_frame
        stop = start + self.world.n_history
        self.latent_all = latent
        self.recorded_action = action
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.latent_history = latent[start:stop][None].clone()
        self.action_history = torch.stack(
            [self.world.normalize_action(item) for item in action[start:stop]], dim=0
        )[None]
        self.predicted_steps = 0
        self.desired_action = action[stop].copy()
        self.cameras = self._fixed_cameras()
        self._sync_action_frames(self.desired_action)
        self._update_scene(self.latent_history[0, -1], label="held-out seed")

    def _next_action(self) -> np.ndarray | None:
        if self.action_source.value == "Recorded validation actions":
            frame = self.seed_frame + self.world.n_history + self.predicted_steps
            if frame >= len(self.recorded_action):
                self.play_widget.value = False
                return None
            action = self.recorded_action[frame].copy()
            self.desired_action = action.copy()
            self._sync_action_frames(action)
        else:
            action = self.desired_action.copy()
        if self.clamp_widget.value:
            action = clamp_action(action, self.episodes.action_min, self.episodes.action_max)
        return action

    def predict_once(self) -> bool:
        action = self._next_action()
        if action is None:
            return False
        next_action = self.world.normalize_action(action)[None, None]
        start_time = time.perf_counter()
        predicted = self.world.step(self.latent_history, self.action_history, next_action)
        self.latent_history = torch.cat([self.latent_history[:, 1:], predicted], dim=1)
        self.action_history = torch.cat([self.action_history[:, 1:], next_action], dim=1)
        self.predicted_steps += 1
        self.last_step_seconds = time.perf_counter() - start_time
        self._update_scene(predicted[0, 0], label="predicted future")
        return True

    def _update_scene(self, latent: torch.Tensor, label: str) -> None:
        start_time = time.perf_counter()
        gaussians = self.world.decode(latent)
        if gaussians is None:
            self.foreground_handle.visible = False
            self.status_widget.value = "decoder produced no foreground points"
            return
        self.last_gaussians = gaussians
        xyz = gaussians["xyz"].detach().cpu().numpy()
        rgb = (gaussians["rgb01"].clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
        indices = evenly_sample(len(xyz), self.args.max_foreground_points)
        self.foreground_handle.points = xyz[indices]
        self.foreground_handle.colors = rgb[indices]
        self.foreground_handle.visible = self.display_mode_widget.value == POINT_CLOUD_MODE
        self._update_exact_render(gaussians)
        self._apply_display_mode()
        total_seconds = time.perf_counter() - start_time
        dataset_frame = self.seed_frame + self.world.n_history - 1 + self.predicted_steps
        self.status_widget.value = (
            f"{label} | episode {self.episode_id}, dataset frame {dataset_frame}, "
            f"future step {self.predicted_steps} | {len(xyz):,} foreground gaussians "
            f"({len(indices):,} shown) | dynamics {1000 * self.last_step_seconds:.0f} ms, "
            f"decode+publish {1000 * total_seconds:.0f} ms"
        )

    def _update_exact_render(self, gaussians) -> None:
        index = min(int(self.camera_widget.value[3:]), len(self.cameras) - 1)
        camera = self.cameras[index]
        image = self.world.render(gaussians, camera, camera["W"], camera["H"])
        self.last_exact_image = image
        self.render_image.image = image
        if self.display_mode_widget.value == GAUSSIAN_MODE:
            self.server.scene.set_background_image(image, format="jpeg", jpeg_quality=90)

    def _apply_display_mode(self) -> None:
        """Switch the main viewport between 3D points and the exact camera render."""
        exact = self.display_mode_widget.value == GAUSSIAN_MODE
        self.foreground_handle.visible = not exact and getattr(self, "last_gaussians", None) is not None
        self.background_handle.visible = not exact and bool(self.show_bg_widget.value)
        self.grid_handle.visible = not exact
        self.bounds_handle.visible = not exact
        for handle in self.action_frames:
            handle.visible = not exact
        if exact and getattr(self, "last_exact_image", None) is not None:
            self.server.scene.set_background_image(
                self.last_exact_image, format="jpeg", jpeg_quality=90
            )
        elif not exact:
            self.server.scene.set_background_image(None)

    def _apply_keyboard_tap(self, key: str) -> bool:
        """Apply one browser key event; return whether a prediction is requested."""
        if key in KEYBOARD_DELTAS:
            self.play_widget.value = False
            self.action_source.value = "Keyboard teleoperation"
            self.desired_action = apply_keyboard_action(
                self.desired_action,
                {key},
                float(self.step_size_widget.value),
                self.episodes.action_min,
                self.episodes.action_max,
                bool(self.clamp_widget.value),
            )
            self._sync_action_frames(self.desired_action)
            self.last_keyboard_step = time.monotonic()
            return True
        if key == "KeyG":
            self.desired_action[6] = 0.0 if self.desired_action[6] > 0.5 else 1.0
            self.action_source.value = "Keyboard teleoperation"
            self._sync_action_frames(self.desired_action)
            return True
        if key == "KeyH":
            self.desired_action[13] = 0.0 if self.desired_action[13] > 0.5 else 1.0
            self.action_source.value = "Keyboard teleoperation"
            self._sync_action_frames(self.desired_action)
            return True
        if key == "KeyR":
            self.play_widget.value = False
            self.pending_reset = True
            return False
        if key == "Space":
            if not self.play_widget.value:
                self.action_source.value = "Recorded validation actions"
            self.play_widget.value = not self.play_widget.value
            return False
        return key == "Period"

    def _process_keyboard(self) -> bool:
        held, taps = self.keyboard.snapshot()
        self.keyboard_state.value = ", ".join(sorted(key[3:] for key in held)) or "none"
        should_step = False
        for key in taps:
            should_step = self._apply_keyboard_tap(key) or should_step
        now = time.monotonic()
        if held and now - self.last_keyboard_step >= 1.0 / float(self.hz_widget.value):
            self.play_widget.value = False
            self.action_source.value = "Keyboard teleoperation"
            self.desired_action = apply_keyboard_action(
                self.desired_action,
                held,
                float(self.step_size_widget.value),
                self.episodes.action_min,
                self.episodes.action_max,
                bool(self.clamp_widget.value),
            )
            self._sync_action_frames(self.desired_action)
            self.last_keyboard_step = now
            should_step = True
        return should_step

    def smoke_test(self) -> dict:
        self.predict_once()
        image = np.asarray(self.render_image.image)
        return {
            "episode": self.episode_id,
            "seed_frame": self.seed_frame,
            "predicted_steps": self.predicted_steps,
            "foreground_points": int(len(self.foreground_handle.points)),
            "render_shape": list(image.shape),
            "render_finite": bool(np.isfinite(image).all()),
            "checkpoint": str(self.world.checkpoint_path),
            "checkpoint_epoch": int(self.world.checkpoint.get("epoch", -1)),
        }

    def run(self) -> None:
        print(
            f"[viser] model-only teleop at http://{self.args.host}:{self.args.port}; "
            f"checkpoint epoch {self.world.checkpoint.get('epoch')}",
            flush=True,
        )
        while self.running:
            should_step = False
            with self.lock:
                should_step = self._process_keyboard()
                if self.pending_reset:
                    self.pending_reset = False
                    self.reset()
                if self.pending_steps > 0:
                    self.pending_steps -= 1
                    should_step = True
                elif self.play_widget.value and not should_step:
                    should_step = True
            if should_step:
                with self.lock:
                    self.predict_once()
                time.sleep(max(0.0, 1.0 / float(self.hz_widget.value)))
            else:
                time.sleep(0.05)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threedws-root", type=Path, required=True)
    parser.add_argument("--dynamics-dir", type=Path, required=True)
    parser.add_argument("--dynamics-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--packed-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--keyboard-port", type=int, default=8098)
    parser.add_argument(
        "--client-build-dir",
        type=Path,
        default=Path("/data/gracexu/cache/3dws/viser-bottle-client/client/build"),
    )
    parser.add_argument("--episode", type=int, default=106)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--max-foreground-points", type=int, default=120_000)
    parser.add_argument("--max-background-points", type=int, default=120_000)
    parser.add_argument("--cam-downscale", type=int, default=2)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(args.threedws_root))
    viewer = BottleViewer(args)
    if args.smoke_test:
        print(json.dumps(viewer.smoke_test(), indent=2, sort_keys=True), flush=True)
        return 0
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
