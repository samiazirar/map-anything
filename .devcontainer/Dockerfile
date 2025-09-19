# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Ensure CUDA extensions build for recent architectures including Blackwell
ENV TORCH_CUDA_ARCH_LIST="80;86;89;90;100"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git curl wget ca-certificates build-essential \
    libgl1-mesa-glx libxkbcommon-x11-0 x11-apps ffmpeg \
    ninja-build cmake pkg-config openssh-client \
    libboost-all-dev libeigen3-dev libgoogle-glog-dev libgflags-dev libmetis-dev \
    libblas-dev liblapack-dev libopenexr-dev libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && python -m pip install --upgrade pip

# Useful tooling for working with HuggingFace Hub/Xet cache backed repos
RUN pip install --no-cache-dir "huggingface_hub[hf_xet]"

# Prefer SSH for GitHub to leverage mounted keys, and trust GitHub host key out of the box
RUN git config --system url."ssh://git@github.com/".insteadOf https://github.com/ \
    && mkdir -p /etc/ssh \
    && ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /etc/ssh/ssh_known_hosts 2>/dev/null || true

RUN git config --global --add safe.directory /workspace

WORKDIR /workspace
EXPOSE 7860
CMD ["bash"]
