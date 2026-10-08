# Dataset contract

The collector first commits each episode as an atomic XPolicyLab-compatible HDF5 file. Completed campaigns then
export eligible episodes as LeRobotDataset v3.0 under the campaign's `lerobot/` directory. HDF5 is retained as the
lossless recovery/source layer; LeRobot is the training-facing Parquet/MP4 representation.

The recorder follows XPolicyLab trajectory format v1.0. Observation keys are singular per frame; stored time-series
keys are plural (`left_ee_pose` becomes `left_ee_poses`). Every state/action dataset has the same leading length `T`.

Episode attributes include:

- `complete`: only true after a clean atomic close.
- `episode_id` and `seed`.
- `length`.
- `success` and `termination_reason`.
- Optional `score`.

RGB images are stored under `vision/<camera>/colors`. Manual/raw mode uses `(T,H,W,3)` `uint8` arrays with gzip.
Campaign mode uses a variable-length sequence of JPEG byte arrays carrying XPolicyLab's `XPL-RGB1` marker, so the
standard XPolicyLab decoder returns correct RGB. Camera shape/intrinsic/extrinsic values are episode constants and
written once.

## Teacher separation

`/teacher` holds controller diagnostics and privileged simulator object poses. A learner must not concatenate this
group into its policy observation. Recommended training inputs are:

- `vision/*/colors`
- `state/*`
- `instruction`

Recommended targets are `action/*`.

## Filtering

Use `manifest.json` as the episode index. Train only on entries where `success` is true unless deliberately building
a failure-aware dataset. Preserve failed files in a separate debug split because they are useful for controller
improvement.

## Validation

```bash
robodojosim validate /path/to/dataset
```

The validator accepts a file or a complete dataset directory. It checks required groups and metadata, atomic-completion
status, every recorded time-axis length, positive frequency, NaN/Inf in numeric streams, and sampled JPEG
decodability. Dataset-level quality review still needs physics success, visual inspection, action/state distribution
plots by `teacher/skill`, and train/validation seed separation.

Create leakage-safe splits grouped by layout ID:

```bash
robodojosim split --dataset /path/to/dataset --profile policy
```

Every trajectory variant of one initial layout stays in the same split.
