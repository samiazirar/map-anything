#!/usr/bin/env bash
set -euo pipefail

# Ensure the workspace root is used as the working directory
cd /workspace

export DEBIAN_FRONTEND=noninteractive

APT_PACKAGES=(
  libavcodec-dev
  libavdevice-dev
  libavformat-dev
  libavfilter-dev
  libswscale-dev
  libswresample-dev
  libavutil-dev
)

# Install multimedia dependencies required by several demos and video utilities
apt-get update
apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"
rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling inside the container
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA 12.8 build) matching the base image toolchain
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
python -m pip install --upgrade --no-cache-dir \
  torch torchvision torchaudio --index-url "${PYTORCH_INDEX_URL}"

# Install MapAnything with all optional extras for a feature-complete developer environment
python -m pip install --no-cache-dir -e .[all]

# Set up linting hooks
if command -v pre-commit >/dev/null 2>&1; then
  pre-commit install --install-hooks || pre-commit install
fi

# Surface the repository to the Python path for interactive shells spawned outside pip
PYTHONPATH_LINE='export PYTHONPATH="/workspace:${PYTHONPATH:-}"'
if ! grep -Fxq "${PYTHONPATH_LINE}" /root/.bashrc 2>/dev/null; then
  echo "${PYTHONPATH_LINE}" >> /root/.bashrc
fi

cat <<'INFO'
=====================================================
Dev container post-create steps completed successfully
- CUDA dev image ready for custom operator builds
- PyTorch + MapAnything (with extras) installed
- pre-commit hooks configured
=====================================================
INFO
