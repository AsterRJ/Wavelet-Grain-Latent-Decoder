FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    openssh-client \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# JAX CUDA 12 wheel. This does not install the full CUDA toolkit.
# It uses the host NVIDIA driver exposed through Docker's --gpus=all.
RUN python -m pip install --upgrade "jax[cuda12]" pillow

WORKDIR /workspace
