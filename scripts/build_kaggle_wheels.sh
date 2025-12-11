#!/bin/bash
# Build COMPLETE offline wheels for Kaggle AIMO3
# Run on RunPod, upload result to Kaggle as single dataset
set -e

WHEEL_DIR="/workspace/kaggle_aimo3_wheels"
rm -rf $WHEEL_DIR
mkdir -p $WHEEL_DIR

echo "=============================================="
echo "Kaggle AIMO3 Complete Offline Wheels Builder"
echo "=============================================="

# vLLM 0.6.2 exact requirements (from wheel metadata)
VLLM_DEPS=(
    "vllm==0.6.2"
    "psutil"
    "sentencepiece"
    "numpy<2.0.0"
    "requests"
    "tqdm"
    "py-cpuinfo"
    "transformers>=4.45.0"
    "tokenizers>=0.19.1"
    "protobuf"
    "aiohttp"
    "openai>=1.40.0"
    "pydantic>=2.9"
    "pillow"
    "prometheus-client>=0.18.0"
    "prometheus-fastapi-instrumentator>=7.0.0"
    "tiktoken>=0.6.0"
    "lm-format-enforcer==0.10.6"
    "outlines>=0.0.43,<0.1"
    "typing-extensions>=4.10"
    "filelock>=3.10.4"
    "partial-json-parser"
    "pyzmq"
    "msgspec"
    "gguf==0.10.0"
    "importlib-metadata"
    "mistral-common>=1.4.3"
    "pyyaml"
    "einops"
    "ray>=2.9"
    "nvidia-ml-py"
    "xformers==0.0.27.post2"
    "fastapi>=0.114.1"
    "six>=1.16.0"
    "setuptools>=74.1.1"
    "compressed-tensors"
)

# uvicorn with standard extras
UVICORN_DEPS=(
    "uvicorn"
    "httptools"
    "uvloop"
    "watchfiles"
    "websockets"
    "python-multipart"
)

# FastAPI/Starlette chain
FASTAPI_DEPS=(
    "fastapi"
    "starlette"
    "anyio"
    "h11"
    "click"
    "annotated-doc"
    "annotated-types"
    "pydantic-core"
    "typing-inspection"
    "idna"
    "sniffio"
)

# Outlines deps chain
OUTLINES_DEPS=(
    "outlines-core"
    "interegular"
    "cloudpickle"
    "diskcache"
    "genson"
    "jinja2"
    "markupsafe"
    "jsonpath-ng"
    "jsonschema"
    "jsonschema-specifications"
    "referencing"
    "rpds-py"
    "attrs"
    "ply"
)

# OpenAI client deps
OPENAI_DEPS=(
    "httpx"
    "httpcore"
    "certifi"
    "distro"
    "jiter"
)

# aiohttp deps
AIOHTTP_DEPS=(
    "aiohttp"
    "aiohappyeyeballs"
    "aiosignal"
    "frozenlist"
    "multidict"
    "yarl"
    "propcache"
)

# mistral-common deps
MISTRAL_DEPS=(
    "mistral-common"
    "pydantic-extra-types"
    "pycountry"
)

# Other ML deps you use
OTHER_DEPS=(
    "accelerate"
    "peft"
    "bitsandbytes"
    "safetensors"
    "huggingface-hub"
    "regex"
    "fsspec"
    "packaging"
    "urllib3"
    "charset-normalizer"
)

echo ""
echo "Downloading all packages..."
echo ""

# Combine all deps
ALL_DEPS=("${VLLM_DEPS[@]}" "${UVICORN_DEPS[@]}" "${FASTAPI_DEPS[@]}" "${OUTLINES_DEPS[@]}" "${OPENAI_DEPS[@]}" "${AIOHTTP_DEPS[@]}" "${MISTRAL_DEPS[@]}" "${OTHER_DEPS[@]}")

# Download for Python 3.11, Linux x86_64
pip download \
    -d $WHEEL_DIR \
    --only-binary=:all: \
    --platform manylinux2014_x86_64 \
    --platform manylinux_2_17_x86_64 \
    --platform manylinux_2_28_x86_64 \
    --platform manylinux1_x86_64 \
    --python-version 311 \
    "${ALL_DEPS[@]}" 2>&1 | tee /tmp/pip_download.log

# Also get any-platform wheels
pip download \
    -d $WHEEL_DIR \
    --only-binary=:all: \
    --platform any \
    --python-version 311 \
    "${ALL_DEPS[@]}" 2>&1 | tee -a /tmp/pip_download.log

# Count results
WHEEL_COUNT=$(ls $WHEEL_DIR/*.whl 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh $WHEEL_DIR | cut -f1)

echo ""
echo "=============================================="
echo "DOWNLOAD COMPLETE"
echo "=============================================="
echo "Wheels: $WHEEL_COUNT"
echo "Size: $TOTAL_SIZE"
echo "Location: $WHEEL_DIR"
echo ""
echo "Check for missing deps:"
grep -i "no matching\|not found\|error" /tmp/pip_download.log | head -10 || echo "None found!"
echo ""
echo "=============================================="
echo "UPLOAD TO KAGGLE:"
echo "=============================================="
echo ""
echo "1. Initialize dataset:"
echo "   kaggle datasets init -p $WHEEL_DIR"
echo ""
echo "2. Edit $WHEEL_DIR/dataset-metadata.json:"
echo '   {"title": "aimo3-offline-wheels", "id": "YOUR_USER/aimo3-offline-wheels", ...}'
echo ""
echo "3. Upload:"
echo "   kaggle datasets create -p $WHEEL_DIR"
echo ""
echo "4. In notebook, use ONLY this one dataset:"
echo '   WHEEL_DIR = "/kaggle/input/aimo3-offline-wheels"'
echo ""
