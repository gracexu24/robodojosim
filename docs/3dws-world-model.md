# 3DWS world-model data contract

This integration was derived from the code and schema in the supplied `3dws` repository. Text inside that repository
is treated as reference material, not as instructions to execute. RoboDojoSim remains the source of truth for data
collection, and the exporter writes the `packed_v2` contract consumed by `3dws/datasets/aloha_packed.py`.

## What changed

World-model runs now temporarily enable all three Isaac Sim cameras' metric `distance_to_image_plane` output,
intrinsic matrices, and camera-to-world matrices. Normal policy runs remain RGB-only. The original RoboDojo YAML files
are backed up, and the collection wrapper restores them on exit, including after a failed run.

The recorder now stores synchronized RGB and metric depth; camera intrinsics and per-frame extrinsics; explicit
timestamps; six arm joints plus gripper state per arm; lossless Cartesian commands; and both bottle poses and
mesh-local bounds as privileged teacher metadata. Wrist-camera extrinsics cannot be episode constants.

World-model storage defaults to 10 Hz while closed-loop control stays at RoboDojo's 25 Hz. This matches the supplied
3DWS tasks and cuts a six-hour depth corpus from 540,000 to 216,000 frames. True simulator timestamps and the source
frequency are preserved.

Depth is stored as gzip-compressed float16 meters, the same precision used by `packed_v2`. Even so, depth dominates
storage: 216,000 frames × three 480×640 cameras is about 398 GB before compression, plus RGB and a second packed
copy during export. Check real compression on a 20-minute pilot and reserve several hundred GB before starting six
hours. Export to another volume or remove the packed copy only after it has been validated and backed up.

`robodojosim export-3dws` creates one HDF5 file per episode with:

- `img/camera_C/depth`, `rgb_in_ir`, bit-packed `valid_mask`, and `obj_mask`;
- `cam/camera_C/K_ir` and OpenCV-axis `E_cam2world`;
- `render/camera_C/jpg`, `K`, `viewmat`, object mask, and valid mask;
- `action/cartesian_action (T,14)` in per-arm `xyz, roll, pitch, yaw, gripper` order;
- `robot/joint_pos` and `robot/full_joint_pos` when joint streams are present;
- stage, timestamp, source-frame index, camera names, action layout, mask semantics, and conversion provenance.

Isaac provides aligned RGB and metric depth directly, so FoundationStereo and RGB-to-IR reprojection are bypassed.
`rgb_in_ir` is the aligned simulated RGB image and its transform is identity by construction.

## Mismatches and required processing

| Area | Existing RoboDojo data | 3DWS expectation | Resolution |
|---|---|---|---|
| Geometry | Historical episodes contain RGB only | Metric depth per camera | Cannot be recovered exactly. Re-record with the new world-model capture configuration. |
| Camera calibration | Historical episodes omit calibration | `K_ir` and OpenCV camera-to-world `E_cam2world` per frame | Re-record. Export converts Isaac/USD axes (`+x right,+y up,-z forward`) to OpenCV (`+x right,+y down,+z forward`). |
| Stereo | No left/right IR stereo pair | FoundationStereo disparity in the ALOHA preprocessing path | Not needed for simulation. Direct metric depth replaces disparity processing. |
| RGB/IR | One aligned simulated camera | RGB resampled into IR frame | No resampling is needed; the exporter records identity alignment. |
| Object masks | Perfect bottle poses, but no SAM3 masks | Per-pixel object label map | Exporter lifts depth to world coordinates and labels points inside each bottle's transformed local mesh bound. This is a box-derived mask, not a pixel-perfect silhouette. SAM3 or Isaac instance masks may replace it offline. |
| Action representation | 16 values: two `xyz+quaternion+gripper` commands | 14 values: two `xyz+rpy+gripper` commands | Exporter converts quaternion to static-XYZ Euler radians and unwraps angles through time. Original lossless actions remain in source HDF5. |
| Action rate | Existing collection is 25 Hz | Supplied 3DWS tasks and teleop use 10 Hz | New world-model recording is 10 Hz; exporter also performs timestamp-nearest resampling. History/horizon settings represent frames at the exported rate. |
| Robot | Dual ARX X5, six joints plus gripper per arm | ALOHA/VX300S state and optional fitted robot geometry | Numeric joint state maps to 14 dimensions, but kinematics and link geometry do not. Never run VX300S FK/rig scripts on ARX data. |
| Robot masks | No ARX Gaussian rig or gripper-mask cache | Required only by robot/class-seed recipes | Use the plain point-cloud VAE path. A robot-scene recipe needs a separate ARX rig, link labels, FK, and mask baker; that tooling is absent from 3DWS. |
| Tactile | No tactile pads | Optional ALOHA tactile streams | Omit tactile-specific recipes. Base VAE/dynamics loaders do not require tactile data. |
| Base poses | Fixed ARX bases are not in observations | Used by ALOHA robot/tactile helpers | Not needed for the plain path. Add calibrated ARX base transforms before robot-aware losses. |
| Failure/misc data | 1/3 success, 1/3 failure, 1/3 miscellaneous | 3DWS accepts episode sequences | Preserved as separate datasets and split metadata. Task success is metadata, not an export filter. |
| LeRobot | Policy-facing Parquet/MP4 | 3DWS reads packed HDF5 | LeRobot cannot be passed directly to 3DWS. Use source HDF5 plus this exporter. |

The 3DWS dynamics loader discovers action dimension from latent files, but its interactive teleop hard-codes
task-specific workspaces and controls. Training accepts the exported action. Interactive rollouts require a RoboDojo
task entry with this dataset's action normalization, ARX workspace, camera views, and control increments.

## Collect and export

World-model campaigns automatically enable and restore depth/calibration capture:

```bash
conda run -n RoboDojoPolicy robodojosim balanced-world-model \
  --xpolicylab /home/gracexu/robodojo-work/RoboDojo/XPolicyLab \
  --output /home/gracexu/robodojo-work/datasets/bottle-world-balanced-6h \
  --target-hours 6
```

Inspect a source episode before conversion:

```bash
robodojosim inspect-3dws /path/to/episode_000000.hdf5
```

After exporting a pilot episode, run numerical calibration gates and create a Mac-compatible RGB/depth/mask video:

```bash
robodojosim review-3dws \
  --episode /path/to/3dws-pilot/train/episode_0.hdf5 \
  --output /path/to/3dws-pilot/review.mp4
```

The command writes a sibling `review.quality.json`. A full collection should not start unless `passed` is true and
the colored bottle-mask overlays visually follow the brown and yellow bottles in all camera views.

Export each success/failure/miscellaneous directory independently. These bounds match the current controller
workspace, but must be visually verified against a short capture before a long training run.

```bash
robodojosim export-3dws \
  --dataset /path/to/bottle-world-balanced-6h/success \
  --output /path/to/bottle-world-balanced-6h/3dws-success \
  --bbox -0.85 0.75 -0.55 0.40 0.30 1.35 \
  --target-fps 10

robodojosim validate-3dws /path/to/bottle-world-balanced-6h/3dws-success
```

If an episode intentionally has no object annotations, `--geometry-only` creates a geometry pack with
`has_obj_mask=false`. Do not use it with object-weighted losses.

## Required quality gates

1. Every camera has finite positive depth, nonsingular intrinsics, and finite camera-to-world poses.
2. Wrist-camera point clouds follow the wrists rather than remaining fixed.
3. Fused points align across all cameras after the Isaac-to-OpenCV axis conversion.
4. The world box retains the table, both bottles, grippers, and bin interactions while excluding distant geometry.
5. Box-derived labels cover bottle points without labeling substantial table/bin regions.
6. Exported timestamps are monotonic and near 0.1 seconds apart.
7. Train/validation/test remain grouped by layout seed; the exporter honors `splits.json`.

Historical RGB-only episodes remain useful for policy work or 2D video models, but are invalid input for this 3D
point-cloud model. The inspector intentionally rejects them.

## Bottle-specific training integration

The supplied 3DWS launchers are task- and machine-specific. Do not point the cloth, rope, or Push-T launchers at
this corpus: they reference ALOHA/VX300S geometry, FoundationStereo sidecars, different camera IDs, and different
workspace bounds. RoboDojoSim provides a plain bottle recipe that consumes only the validated `packed_v2` contract.

First stage every packed category and side order. Labels must be unique. The command creates only lightweight
symlinks and assigns one global episode ID space, so roots that each contain `episode_0.hdf5` cannot collide:

```bash
robodojosim stage-3dws-training \
  --root brown_success=/data/brown-left/3dws-success \
  --root brown_failure=/data/brown-left/3dws-failure \
  --root brown_misc=/data/brown-left/3dws-miscellaneous \
  --root yellow_success=/data/yellow-left/3dws-success \
  --root yellow_failure=/data/yellow-left/3dws-failure \
  --root yellow_misc=/data/yellow-left/3dws-miscellaneous \
  --output /data/3dws-bottles/stage
```

Run a smoke VAE before a full run. The caller must already be in the supplied 3DWS Python environment:

```bash
ROBODOJOSIM_3DWS_SMOKE=1 bash scripts/train_3dws_bottle_vae.sh \
  /workspace/3dws /data/3dws-bottles/stage \
  /data/3dws-bottles/models/vae-smoke 0

bash scripts/train_3dws_bottle_vae.sh \
  /workspace/3dws /data/3dws-bottles/stage \
  /data/3dws-bottles/models/vae 0,1
```

The baseline deliberately uses a 32x20x20 latent grid and one latent encoding per frame. Measure a representative
latent episode on disk before increasing `ROBODOJOSIM_3DWS_NX/NY/NZ` or `ROBODOJOSIM_3DWS_ENCODINGS`; latent caches,
not checkpoints, dominate storage.

Export each original root separately, then stage its latents under the same global IDs as the packed files:

```bash
ROBODOJOSIM_BIN=/path/to/robodojosim bash scripts/export_3dws_bottle_latents.sh \
  /workspace/3dws /data/3dws-bottles/stage \
  /data/3dws-bottles/models/vae/checkpoints/best.pth \
  /data/3dws-bottles/raw-latents /data/3dws-bottles/stage/latents 0
```

Finally smoke-test and train action-conditioned dynamics:

```bash
ROBODOJOSIM_3DWS_SMOKE=1 bash scripts/train_3dws_bottle_dynamics.sh \
  /workspace/3dws /data/3dws-bottles/stage /data/3dws-bottles/stage/latents \
  /data/3dws-bottles/models/vae/checkpoints/best.pth \
  /data/3dws-bottles/models/dynamics-smoke 0

bash scripts/train_3dws_bottle_dynamics.sh \
  /workspace/3dws /data/3dws-bottles/stage /data/3dws-bottles/stage/latents \
  /data/3dws-bottles/models/vae/checkpoints/best.pth \
  /data/3dws-bottles/models/dynamics 0,1
```

The checked-in baseline is summarized in `configs/training_3dws_bottles.json`. It uses 8 history frames and an
autoregressive 1-to-15-step curriculum at 10 Hz. W&B is off unless `ROBODOJOSIM_3DWS_WANDB=1`. Both launchers save
their exact command and copy the staging manifest into the model directory. Existing `latest.pth` checkpoints are
resumed rather than overwritten.
