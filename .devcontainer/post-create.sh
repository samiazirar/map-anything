#!/usr/bin/env bash
set -euo pipefail

# Ensure the workspace root is used as the working directory
cd /workspace

export DEBIAN_FRONTEND=noninteractive
# Prefer pre-built wheels to avoid unnecessary source builds during dependency resolution
export PIP_PREFER_BINARY=1

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

# Configure SSH for GitHub access (using SSH agent forwarding from host)
# Disable strict host key checking for GitHub to avoid known_hosts issues
export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"


# Install MapAnything with all optional extras for a feature-complete developer environment
python -m pip install .[all]


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
