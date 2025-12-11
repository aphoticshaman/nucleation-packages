# =============================================================================
# LFBM-TRAINER: Ultimate Fine-Tuning Docker Image
# =============================================================================
# Image: aphoticshaman/lfbm-trainer:latest
# Purpose: QLoRA/DoRA fine-tuning for Qwen-72B with all dependencies resolved
# Target: RunPod H200 (141GB HBM3e) or equivalent
#
# CUDA: 12.6  |  PyTorch: 2.5.1  |  Python: 3.11
# =============================================================================

FROM nvcr.io/nvidia/pytorch:24.10-py3

LABEL maintainer="aphoticshaman"
LABEL version="1.0.0"
LABEL description="Ultimate fine-tuning image with DoRA, NEFTune, rsLoRA, LoRA+"

# -----------------------------------------------------------------------------
# Environment Configuration
# -----------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# CUDA optimization flags
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Flash Attention environment
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV MAX_JOBS=16

# Hugging Face cache configuration
ENV HF_HOME=/workspace/cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/workspace/cache/huggingface/datasets

# Weights & Biases
ENV WANDB_PROJECT=lfbm-elle
ENV WANDB_DIR=/workspace/logs/wandb

# -----------------------------------------------------------------------------
# System Dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    vim \
    htop \
    nvtop \
    tmux \
    screen \
    build-essential \
    ninja-build \
    cmake \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# -----------------------------------------------------------------------------
# Python Package Installation (Kaggle-Compatible Versions)
# -----------------------------------------------------------------------------
# CRITICAL: These versions are verified compatible with Kaggle as of 2025-12-10
# Python: 3.11, CUDA: 12.4, PyTorch: 2.5.1
# -----------------------------------------------------------------------------

# Core ML/DL packages
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
    # Transformers ecosystem (exact Kaggle versions)
    transformers==4.47.0 \
    datasets==3.2.0 \
    tokenizers==0.21.0 \
    accelerate==1.2.1 \
    safetensors==0.4.5 \
    huggingface-hub==0.27.0 \
    # PEFT with LoRA+ and rsLoRA support
    peft==0.14.0 \
    # Quantization
    bitsandbytes==0.45.0 \
    auto-gptq==0.7.1 \
    # Optimization
    scipy==1.14.1 \
    einops==0.8.0 \
    packaging==24.2 \
    # Logging and tracking
    wandb==0.19.1 \
    tensorboard==2.18.0 \
    # Math utilities
    sympy==1.13.3 \
    mpmath==1.3.0

# Flash Attention 2 (build from source for CUDA 12.6)
RUN pip install flash-attn==2.7.2.post1 --no-build-isolation

# DeepSpeed with all optimizations
RUN pip install deepspeed==0.16.2

# Axolotl (latest for advanced training features)
RUN pip install axolotl==0.6.0

# vLLM for inference (compatible with Qwen)
RUN pip install vllm==0.6.6.post1

# Additional utilities
RUN pip install \
    trl==0.13.0 \
    sentencepiece==0.2.0 \
    protobuf==5.29.2 \
    tiktoken==0.8.0 \
    jsonlines==4.0.0 \
    pyyaml==6.0.2 \
    rich==13.9.4 \
    tqdm==4.67.1

# -----------------------------------------------------------------------------
# Build Kaggle-Compatible Wheels
# -----------------------------------------------------------------------------
WORKDIR /wheels

# Create wheel build script
RUN echo '#!/bin/bash\n\
set -e\n\
WHEEL_DIR=/wheels/kaggle_compatible\n\
mkdir -p $WHEEL_DIR\n\
\n\
# Packages to build as wheels\n\
PACKAGES=(\n\
    "flash-attn==2.7.2.post1"\n\
    "bitsandbytes==0.45.0"\n\
    "vllm==0.6.6.post1"\n\
    "deepspeed==0.16.2"\n\
    "auto-gptq==0.7.1"\n\
)\n\
\n\
for pkg in "${PACKAGES[@]}"; do\n\
    echo "Building wheel for: $pkg"\n\
    pip wheel --no-deps -w $WHEEL_DIR "$pkg" || echo "Warning: $pkg wheel failed"\n\
done\n\
\n\
echo "Wheels built in $WHEEL_DIR"\n\
ls -la $WHEEL_DIR\n\
' > /usr/local/bin/build_kaggle_wheels.sh && chmod +x /usr/local/bin/build_kaggle_wheels.sh

# Pre-build wheels during image creation
RUN mkdir -p /wheels/kaggle_compatible && \
    pip wheel --no-deps -w /wheels/kaggle_compatible \
    flash-attn==2.7.2.post1 \
    bitsandbytes==0.45.0 \
    deepspeed==0.16.2 || true

# -----------------------------------------------------------------------------
# Workspace Configuration
# -----------------------------------------------------------------------------
WORKDIR /workspace

# Create directory structure
RUN mkdir -p \
    /workspace/data \
    /workspace/models \
    /workspace/outputs \
    /workspace/logs \
    /workspace/cache/huggingface \
    /workspace/scripts \
    /workspace/configs

# Copy default training configurations
COPY configs/ /workspace/configs/
COPY scripts/ /workspace/scripts/

# -----------------------------------------------------------------------------
# Health Check and Entry
# -----------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command
CMD ["/bin/bash"]

# -----------------------------------------------------------------------------
# Usage:
# -----------------------------------------------------------------------------
# Build:
#   docker build -t aphoticshaman/lfbm-trainer:latest .
#
# Run locally:
#   docker run --gpus all -it --shm-size=64g \
#     -v $(pwd)/data:/workspace/data \
#     -v $(pwd)/outputs:/workspace/outputs \
#     aphoticshaman/lfbm-trainer:latest
#
# Push to Docker Hub:
#   docker push aphoticshaman/lfbm-trainer:latest
# -----------------------------------------------------------------------------
