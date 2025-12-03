# LatticeForge Cognitive LLM - GPU Inference Server
# Deploy to RunPod Serverless, Modal, or any GPU instance
#
# Build: docker build -t latticeforge-llm .
# Run:   docker run --gpus all -p 8000:8000 latticeforge-llm

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy inference code
COPY . /app/research/inference/

# Note: cognitive module should be copied from build context
# When building from repo root, use: docker build -f packages/research/inference/Dockerfile .
# Or copy cognitive folder into inference/ before building

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Environment
ENV PYTHONPATH=/app
ENV ADAPTER_REPO=aphoticshaman/latticeforge-unified
ENV BASE_MODEL=microsoft/Phi-3-mini-4k-instruct
ENV USE_GPU=1
ENV HF_HOME=/app/.cache/huggingface

# Pre-download model on build (optional - makes container larger but faster cold start)
# RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct'); AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "research.inference.server:app", "--host", "0.0.0.0", "--port", "8000"]
