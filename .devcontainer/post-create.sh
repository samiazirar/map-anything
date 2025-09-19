#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends libavcodec-dev libavdevice-dev libavformat-dev libavfilter-dev libswscale-dev libswresample-dev libavutil-dev
rm -rf /var/lib/apt/lists/*

cd "${REPO_ROOT}"

python -m pip install --upgrade pip
python -m pip install --no-cache-dir -e ".[all]"
python -m pip install --no-cache-dir av

DUSTER_DIR="${REPO_ROOT}/duster"
if [ ! -d "${DUSTER_DIR}" ]; then
  git clone --recursive https://github.com/ethz-vlg/duster.git "${DUSTER_DIR}"
fi

python -m pip install --no-cache-dir -r "${DUSTER_DIR}/requirements.txt"

CHECKPOINT_DIR="${DUSTER_DIR}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
if [ ! -f "${CHECKPOINT_PATH}" ]; then
  wget -O "${CHECKPOINT_PATH}" https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
fi
md5sum "${CHECKPOINT_PATH}"

if ! grep -Fq "duster" /root/.bashrc; then
  printf 'export PYTHONPATH="%s:$PYTHONPATH"\n' "${DUSTER_DIR}" >> /root/.bashrc
fi

if ! grep -Fq "${REPO_ROOT}" /root/.bashrc; then
  printf 'export PYTHONPATH="%s:$PYTHONPATH"\n' "${REPO_ROOT}" >> /root/.bashrc
fi
