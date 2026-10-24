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

The server is bound to remote loopback for privacy. Its PID and log are under:

```text
/data/gracexu/robodojo-work/3dws-bottles/viewer/viser-keyboard-render-toggle-20261024/
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
- `Autoplay predictions` repeatedly advances with the current target.
- `Recorded validation actions` replays the held-out episode's real action
  sequence through the world model. `Play recorded held-out sequence` starts it
  directly and `Pause` stops it.
- `Reset from held-out history` loads eight real latent history frames and
  restores the action target.
- `Frame predicted world` restores the useful camera view.
- The Rendering section can switch the main viewport between the interactive
  3D point representation and the model's exact fixed-camera Gaussian render.

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
subsampling, and bounds geometry. The full GPU smoke test loads
`checkpoints/best.pth` (epoch 738), seeds held-out episode 106, predicts one
future latent, decodes it, and checks a finite 320x240 Gaussian render. The live
browser test confirmed a one-step prediction and visible 3D scene.
