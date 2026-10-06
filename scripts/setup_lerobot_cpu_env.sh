#!/bin/bash
set -euo pipefail

ENV_NAME="${1:-RoboDojoLeRobot}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_EXE="${CONDA_EXE:-$(command -v conda)}"

if ! "${CONDA_EXE}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    "${CONDA_EXE}" create -y -n "${ENV_NAME}" python=3.11 pip
fi

# Install Torch from the CPU-only index first. LeRobot's normal resolver then
# sees a compatible pair and cannot pull CUDA runtime wheels.
"${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.1 torchvision==0.22.1
"${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install \
    lerobot==0.4.4 'h5py>=3.10'
# LeRobot's broad TorchCodec constraint currently resolves to 0.10, which is
# for Torch 2.10. Pin the official Torch 2.7-compatible CPU wheel and provide
# its shared FFmpeg runtime through Conda (not the operating system).
"${CONDA_EXE}" install -y -n "${ENV_NAME}" -c conda-forge 'ffmpeg<8'
"${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install --force-reinstall --no-deps \
    --index-url https://download.pytorch.org/whl/cpu torchcodec==0.5.0
"${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install -e "${PROJECT_ROOT}" --no-deps

"${CONDA_EXE}" run -n "${ENV_NAME}" python -c \
    'import lerobot, torch, torchcodec; assert torch.version.cuda is None; print(f"LeRobot {lerobot.__version__}, Torch {torch.__version__}, TorchCodec {torchcodec.__version__}, CUDA runtime: {torch.version.cuda}")'
