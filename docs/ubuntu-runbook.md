# Ubuntu GPU runbook

## 1. Prerequisites

Use the OS, NVIDIA driver, CUDA-capable GPU, and disk space required by the current RoboDojo documentation. The
upstream installer currently creates a Python 3.11 `RoboDojo` conda environment and installs Isaac Sim 5.1,
IsaacLab, cuRobo, and XPolicyLab. Accept the NVIDIA/Isaac licenses when prompted.

Keep the repositories in this shape:

```text
/workspace/RoboDojo/
├── XPolicyLab/
├── Assets/
└── ...
/workspace/robodojosim/
```

Complete the official RoboDojo asset/config validation before adding this controller. A missing robot, bottle,
dustbin, or layout asset cannot be diagnosed from the controller layer. Production collection uses the explicit
tabletop-dustbin training variant; set `ROBODOJOSIM_CENTRALIZE_DUSTBIN=0` only when reproducing the public layout.

## 2. Install the adapter

```bash
cd /workspace/robodojosim
bash scripts/install_xpolicylab_adapter.sh \
  /workspace/RoboDojo/XPolicyLab RoboDojo RoboDojo
```

The installer refuses to overwrite an existing `policy/bottle_scripted` directory. Remove or rename an old copy
deliberately before reinstalling.

## 3. Smoke test one seed

```bash
mkdir -p /workspace/datasets/bottle-smoke
ROBODOJOSIM_EVAL_NUM=1 bash scripts/run_collection.sh \
  /workspace/RoboDojo/XPolicyLab \
  0 RoboDojo RoboDojo /workspace/datasets/bottle-smoke
```

In another shell, watch GPU memory with `nvidia-smi`. The policy server is only a lightweight heartbeat; almost all
GPU use should belong to Isaac Sim.

Expected completion output contains a line like:

```text
[bottle_scripted] episode=0 steps=... success=True reason=task_success
```

Then validate:

```bash
conda run -n RoboDojo robodojosim validate \
  /workspace/datasets/bottle-smoke/episode_000000.hdf5
```

Do not treat `success=False` files as demonstrations. Retain them for debugging or move them out of the training
split based on `manifest.json`.

To watch the same run in the Isaac Sim GUI, first patch the pinned RoboDojo launcher and then run from a terminal on
the Ubuntu desktop (not from a display-less SSH session):

```bash
cd /workspace/robodojosim
bash scripts/enable_gui_mode.sh /workspace/RoboDojo
export ROBODOJO_HEADLESS=0
export OMNI_KIT_ACCEPT_EULA=YES
export ROBODOJOSIM_EVAL_NUM=1
# On multi-GPU workstations, select the GPU driving the desktop display.
# Check it with: nvidia-smi --query-gpu=index,display_active --format=csv,noheader
export ROBODOJOSIM_ENV_GPU_ID=1
bash scripts/run_collection.sh \
  /workspace/RoboDojo/XPolicyLab \
  0 RoboDojoPolicy RoboDojo /workspace/datasets/bottle-gui
```

`ROBODOJOSIM_ENV_GPU_ID` defaults to `0` for headless runs. A visible GUI needs the renderer GPU to match the GPU
with `Display Active: Enabled`; otherwise Kit can create an X11 window but repeatedly report that its backbuffers are
not initialized. `ROBODOJOSIM_POLICY_GPU_ID` independently selects the policy process GPU and also defaults to `0`.

For state-only reward and controller diagnostics, install and enable the optional camera-free path:

```bash
bash scripts/enable_fast_calibration.sh /workspace/RoboDojo
export ROBODOJOSIM_CALIBRATION_FAST=1
```

This saves disk and video-encoding work but does not materially accelerate the IK/physics loop on the tested system.
Unset the variable before collecting any LeRobot-bound data because it intentionally omits RGB observations.

## 4. Calibrate

Edit a copy of `configs/bottle_task.json` and point the run at it:

```bash
export ROBODOJOSIM_CONFIG=/workspace/configs/bottle-calibrated.json
```

Follow [calibration.md](calibration.md). First prove direct left-arm pickups; then enable/tune the right-to-left
right-arm direct drop. Test at least ten deterministic seeds before a long run.

## 5. Run quota-driven campaigns

```bash
conda run -n RoboDojo robodojosim campaign \
  --profile policy \
  --xpolicylab /workspace/RoboDojo/XPolicyLab \
  --output /workspace/datasets/bottle-policy-50

conda run -n RoboDojo robodojosim campaign \
  --profile world_model \
  --xpolicylab /workspace/RoboDojo/XPolicyLab \
  --output /workspace/datasets/bottle-world-6h
```

Keep these in separate directories. The first command stops at 50 successful simple-task episodes. The second stops
at six hours of complete randomized interaction frames; its episodes include pushing, airborne holding, waypoint
carrying, and set-down/regrasp sequences.
That estimate changes after calibration, so use `robodojosim report` rather than an episode count for the world-model
target.

RoboDojo selects its deterministic layout IDs internally (25 for this task in the current release). Each campaign
pass assigns a new output range and trajectory variant. An interrupted episode leaves a hidden partial file; the next
attempt rewrites that partial path and commits it only after clean close.

## 6. Quality gate

Before training:

- Validate every HDF5 file with `robodojosim validate /path/to/dataset`.
- Filter manifest entries to `success=true`.
- Check action/state lengths match and no values are NaN/Inf.
- Review videos or sampled RGB sequences for left/right grasp and tabletop-bin drop quality.
- Split by seed, not by individual frames.
- Keep the calibrated config and upstream commit hashes with the dataset.
- Confirm `splits.json` keeps every repeat of a layout in one split.
- For the six-hour run, measure JPEG storage after the first pass before committing disk capacity.

## Common failures

| Symptom | Likely cause | First adjustment |
|---|---|---|
| IK does not move an arm | pose/orientation unreachable | reduce approach height or calibrate quaternion |
| Fingers close above bottle | grasp target too high | lower `grasp_clearance` |
| Bottle collides during transit | lift too low | raise `lift_height` |
| Right bottle cannot reach bin | public far-left layout is active | enable the default tabletop-dustbin transform |
| Bottle hits bin rim | drop target too low | raise `drop_clearance` |
| Objects placed but task not complete | arms not home or gripper not open | inspect final EE/gripper state and reward checks |
| `ModuleNotFoundError: robodojosim` | overlay not installed in eval env | rerun installer with the RoboDojo env name |
