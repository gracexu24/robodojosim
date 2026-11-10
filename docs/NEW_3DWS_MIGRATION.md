# RoboDojo ALOHA tactile + imported 3DWS migration

This branch connects the copied ALOHA/tactile Isaac Sim setup to the code
imported from `3d-world-simulator-main.zip`.  The original ARX simulation,
packed datasets, checkpoints, and old 3DWS tree are not overwritten.

## Repository map

- `/home/gracexu/robodojo-work/RoboDojo-arx-aloha-fixed-cameras`
  contains the copied Isaac Sim task.  `env_cfg/arx_x5_aloha_fixed_cameras.yml`
  selects two X5A arms with ALOHA grippers, four 12x32 VT-Refine pads, and the
  overhead/left-fixed/right-fixed cameras.  Reset writes the exact home joint
  state to PhysX; the calibrated USD's joint1=0 home and the existing +90-degree
  root transforms point both arms forward into the workspace.  The tactile
  backend registers spawned meshes once and snapshots object kinematics once
  per observation, avoiding recursive PhysX access across four pads.
- `/home/gracexu/robodojo-work/robodojosim` is the integration/recording repo.
  `EpisodeRecorder` writes the four spatial tactile grids.  The world-model
  exporter writes packed `tactile/xyz` and normalized `tactile/sig` arrays.
  The scripts below are the stable operator interface.
- `/data/gracexu/robodojo-work/3dws` is the imported upstream code.  It now has
  the `robodojo_bottles` task, staged-ID-safe dataset lookup, exact Isaac metric
  depth fallback, tactile-aware latent export, loss CSVs, and the sidecar
  preprocessor.
- `/data/gracexu/robodojo-work/3dws-before-zip-9152640-20261109` is the preserved
  old 3DWS code.  It is reference-only.

## Safe data flow

1. Record raw episodes in the copied simulation.  The raw HDF5 includes state,
   Cartesian action, three cameras, teacher metadata, and four tactile pads.
2. Export/stage packed episodes with `robodojosim`.  Existing packed HDF5 files
   are never rewritten by preprocessing.
3. Run `scripts/prepare_3dws_bottle_data.sh 3DWS_ROOT STAGE_ROOT`.  It poses the
   actual X5A-ALOHA URDF with each 14-D recorded joint state and atomically adds:
   `robot_label_cache`, projected `gripper_supervision` masks, and
   `tactile_targets`.  Simulator metric depth is consumed directly, so no fake
   FoundationStereo disparity is produced.  Complete files are skipped.
4. Run `train_3dws_bottle_vae.sh`, then `export_3dws_bottle_latents.sh`, then
   `train_3dws_bottle_dynamics.sh`.  Each resumes from `latest.pth` when present.
   The imported trainers are single-process; wrappers reject comma-separated
   GPU lists instead of silently launching unsupported DDP.

Legacy pre-import wrappers are retained verbatim under
`scripts/legacy-import-v1/` for audit and rollback.

## Validation performed

- Python compilation and shell syntax validation across all three repos.
- One real staged episode passed packed point-cloud, exact-depth, render,
  projected gripper-mask, and three-resolution robot-label loading.
- A fixed-camera Isaac episode completed both pick-and-place operations in 268
  control actions with `success=True` and `termination_reason=task_success`.
  Both final grippers were open and both bottles rested inside the bin.  Its
  107-frame/10 Hz raw episode contains metric depth for all three cameras and
  all four `(12,32)` tactile streams.  The run also confirmed that the former
  recursive PhysX mutex abort does not recur.
