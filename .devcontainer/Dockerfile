# syntax=docker/dockerfile:1

# CUDA \u226512.8 is required for building custom ops for recent GPUs (e.g., Blackwell sm_100)
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Ensure any CUDA extensions compile for a wide range of NVIDIA GPUs
ENV TORCH_CUDA_ARCH_LIST="80;86;89;90;100;120"

# Base OS dependencies - Updated for Ubuntu 24.04 compatibility
RUN apt-get update \
     && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        git curl wget ca-certificates build-essential \
        libglx-mesa0 libxkbcommon-x11-0 x11-apps ffmpeg \
        ninja-build cmake pkg-config openssh-client \
        libssl-dev libffi-dev \
     && rm -rf /var/lib/apt/lists/*

# Make python point to python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Configure pip to work with Ubuntu 24.04's externally-managed environment
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Helpful utilities
RUN pip install -U "huggingface_hub[hf_xet]"

# Prefer SSH for GitHub remotes to leverage mounted host SSH keys
RUN git config --system url."ssh://git@github.com/".insteadOf https://github.com/

# Pre-populate GitHub host key to avoid prompts in fresh containers
RUN mkdir -p /etc/ssh \
 && ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /etc/ssh/ssh_known_hosts 2>/dev/null || true

# Avoid git safety prompts when working in bind-mounted workspace
RUN git config --global --add safe.directory /workspace

WORKDIR /workspace
EXPOSE 7860
CMD ["bash"]
