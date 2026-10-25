# Two-bottle 3DWS Viser viewer

This is an interactive, GPU-backed viewer for the frozen two-bottle 3DWS VAE
and action-conditioned dynamics model. It is a **model-only teleoperator**: it
changes 14D Cartesian action targets and predicts future latent scenes. It does
not send commands to Isaac Sim, ROS, or physical robot hardware.

## Connect to the running viewer

On the Mac, keep this tunnel open:

```bash
ssh -N -L 8097:127.0.0.1:8097 -L 8098:127.0.0.1:8098 gracexu@128.59.17.246
```

Then open <http://127.0.0.1:8097/>.

The server is bound to remote loopback for privacy. Its PID and log are under
the most recent timestamped directory in:

```text
/data/gracexu/robodojo-work/3dws-bottles/viewer/
```

## Controls

- Click the 3D canvas, then hold `A/D`, `Q/E`, `W/S` to move the left target
  in x, y, z. Hold `J/L`, `U/O`, `I/K` for the right target.
- Press `G` or `H` to toggle the left or right gripper. Rotation stays at the
  seed/recorded target while keyboard teleoperation changes Cartesian position.
- `R` resets, `Space` plays/pauses the recorded sequence, and `.` predicts one
  future step. A short key press always produces one step; holding repeats at
  the selected Hz.
- `Predict one future` advances the frozen dynamics model one 10 Hz step.
- Keyboard control defaults to `Stable one-step preview`: every changed target
  is evaluated from the same eight real held-out history frames with a fixed
  future-noise seed. This isolates the left/right command response and avoids
  accumulating open-loop drift while teleoperating.
- `Autoregressive imagination` remains available for continuous free-running
  rollouts. The model was trained through 16 consecutive predicted steps, so
  the viewer visibly warns when that horizon is exceeded.
- `Autoplay predictions` repeatedly advances with the current target.
- `Recorded validation actions` replays the held-out episode's real action
  sequence through the world model. `Play recorded held-out sequence` starts it
  directly and `Pause` stops it.
- `Reset before teleop` stops playback, switches to keyboard control, reloads
  eight real held-out history frames, and restores the matching action target.
  Use it whenever switching from a recorded rollout to manual commands so the
  latent history and command history begin from the same state.
- Magenta and cyan 3D markers show the left and right absolute Cartesian
  targets. The `workspace limits` field reports when a key is pushing a target
  against the held-out training-data envelope.
- `Frame predicted world` restores the useful camera view.
- The Rendering section provides four modes:
  - `Point cloud (interactive 3D)` is the lightest interactive view.
  - `Native 3D Gaussian splats` exposes Viser's orbitable, zoomable splat
    renderer. It uses the model's Gaussian positions, covariances, RGB colors,
    and opacity, but not its view-dependent spherical-harmonic color term.
  - `Exact Gaussian render (fixed camera)` is the pixel-accurate gsplat output
    from one recorded camera.
  - `Exact Gaussian render (track orbit camera)` rerenders with the interactive
    Viser camera as it moves. It retains the exact model renderer, but updates
    at the selected tracked-render FPS rather than every browser animation
    frame.

Action targets are clamped to the held-out data envelope by default. The action
order is:

```text
left xyz, left roll/pitch/yaw, left gripper,
right xyz, right roll/pitch/yaw, right gripper
```

## Launch manually

```bash
cd /home/gracexu/robodojo-work/robodojosim
GPU=0 scripts/run_3dws_bottle_viser.sh \
  --host 127.0.0.1 --port 8097 --episode 106 --frame 0
```

The launcher uses `PYTHONNOUSERSITE=1`, the
`/data/gracexu/conda-envs/3dws` Python environment, and Viser plus runtime
caches under `/data/gracexu/cache/3dws`.

## Validation

The utility tests cover action/pose conversion, action clamping, point
subsampling, bounds geometry, workspace warnings, Gaussian covariance, and
Viser-camera conversion. The full GPU smoke test loads
`checkpoints/best.pth` (epoch 738), seeds held-out episode 106, predicts one
future latent, decodes it, publishes native splats, and checks a finite 320x240
exact Gaussian render. The live browser test confirms reset-before-teleop,
workspace warnings, camera-tracked exact rendering, and an orbitable splat scene.

## Evaluate ACT and Diffusion Policy inside 3DWS

The registered two-bottle ACT and Diffusion Policy checkpoints can be rolled
out through the frozen world model with:

```bash
cd /home/gracexu/robodojo-work/robodojosim
EPISODES=3 STEPS=16 scripts/evaluate_trained_policies_in_3dws.sh
```

Each policy runs sequentially.  GPU 0 hosts the VAE/dynamics model and GPU 1
hosts the policy by default.  The evaluator records a three-camera MP4,
representative PNGs, per-step timings/actions, worker logs, checkpoint hashes,
and an evaluation manifest for every held-out seed.

This is a **closed-loop learned-world-model evaluation**, not an Isaac Sim task
success test.  The prior clamped Cartesian command is used as the next policy
state; the head camera is fixed and each wrist camera follows the rigid
end-effector-to-camera transform measured at the held-out seed.  Runs default
to 16 predicted steps because that is the dynamics model's trained
autoregressive horizon.

The policy worker uses the CUDA-enabled `RoboDojo` Python and appends the
LeRobot 0.4.4 environment's site-packages after it.  This preserves the CUDA
PyTorch build while reusing the saved checkpoint processors and policy code.
