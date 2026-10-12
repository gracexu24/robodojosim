# Training, inference, evaluation, and publishing

The LeRobot tree is the training-facing copy of a completed policy campaign. Keep source HDF5, LeRobot data, model
checkpoints, and evaluation videos on lab storage; Git tracks code, configuration, and selected small reports.

## Environment isolation

Do not add LeRobot training dependencies to the Isaac Sim `RoboDojo` environment. Create a policy environment:

```bash
bash scripts/setup_training_env.sh RoboDojoLeRobotTrain
```

This installs the repository plus pinned LeRobot 0.4.4, including ACT and Diffusion Policy dependencies, without `sudo`.
Confirm the lab NVIDIA driver is compatible with the PyTorch wheel before accepting dependency changes.

## Validate and train

```bash
conda run -n RoboDojoLeRobotTrain robodojosim inspect-lerobot \
  --dataset /workspace/datasets/bottle-policy-50/lerobot

conda run -n RoboDojoLeRobotTrain robodojosim train \
  --policy act \
  --dataset /workspace/datasets/bottle-policy-50/lerobot \
  --output /workspace/models/bottle-act \
  --gpu 1
```

Use `--policy diffusion` for Diffusion Policy. Add `--dry-run` to validate the dataset and print the exact command
without starting training. Each run writes `robodojosim_training.json`; completed checkpoints live under
`checkpoints/last/pretrained_model`. Resume with `--resume-from .../train_config.json`.

The checked-in defaults use short executed chunks so both policies observe and replan during contact-rich movement.
`configs/training_act.json` and `configs/training_diffusion.json` document those horizons.

## Install the learned-policy adapter and evaluate

```bash
bash scripts/install_xpolicylab_adapter.sh \
  /workspace/RoboDojo/XPolicyLab RoboDojo RoboDojoLeRobotTrain

conda run -n RoboDojo robodojosim evaluate-policy \
  --xpolicylab /workspace/RoboDojo/XPolicyLab \
  --checkpoint /workspace/models/bottle-act/checkpoints/last/pretrained_model \
  --output /workspace/evaluations/bottle-act \
  --episodes 10 \
  --policy-gpu 1 \
  --env-gpu 0 \
  --gui
```

The policy process loads LeRobot in the policy environment. RoboDojo stays in its Isaac Sim environment. XPolicyLab
passes camera/state observations to the policy and Cartesian end-effector actions back to the simulator. The adapter
normalizes quaternions, clamps the calibrated workspace and grippers, and rejects non-finite actions.
`--policy-env` accepts either a Conda environment name or an absolute virtual-environment path containing
`bin/python`, which is useful when reusing the lab machine's existing CUDA-enabled PyTorch installation.

Each episode produces atomic JSON metrics. The command also writes `metrics.json`, `episodes.csv`, and `summary.md`.
Run `robodojosim summarize-evaluation --output <directory>` to rebuild those reports. Commit a small `summary.md` and
plots to Git when useful; store MP4 files as release assets or with Git LFS.

## Hugging Face storage

Authentication is external to this repository. Log in with `hf auth login` or set `HF_TOKEN`; never commit a token.
Uploads are private by default and validate the local artifact before network access:

```bash
robodojosim push-dataset \
  --dataset /workspace/datasets/bottle-policy-50/lerobot \
  --repo-id YOUR_USER/robodojo-bottle-policy

robodojosim push-model \
  --checkpoint /workspace/models/bottle-act/checkpoints/last/pretrained_model \
  --repo-id YOUR_USER/robodojo-bottle-act
```

Add `--dry-run` first. Add `--public` only when public access is intentional. Datasets and models use separate Hub
repository types. Git ignores local datasets, checkpoints, Parquet, HDF5, safetensors, and MP4 files.

## GitHub automation boundary

GitHub-hosted CI runs portable unit tests only. Isaac Sim requires the lab GPU, licensed assets, and its local
installation. `.github/workflows/gpu-evaluation.yml` is a manual workflow for a restricted self-hosted runner carrying
the `robodojo-gpu` label. Configure the `ROBODOJO_XPOLICYLAB` repository variable before using it. The workflow adds
`summary.md` to the GitHub job summary and uploads metrics and videos as an artifact. Do not put SSH or Hugging Face
credentials in repository files.
