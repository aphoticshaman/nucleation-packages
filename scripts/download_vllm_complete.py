#!/usr/bin/env python3
"""
Download complete vLLM wheels for Kaggle offline environment.
Run on RunPod, then upload the resulting folder to Kaggle.
"""
import subprocess
import sys
import os
import shutil

WHEEL_DIR = "/workspace/vllm_complete_wheels"

# All packages vLLM needs (captured from dependency resolution)
PACKAGES = [
    # Core vLLM
    "vllm==0.6.2",

    # vLLM direct deps
    "xformers==0.0.27.post2",
    "ray==2.9.3",
    "msgspec",
    "pyzmq",
    "nvidia-ml-py",
    "compressed-tensors",

    # FastAPI stack (prometheus-fastapi-instrumentator deps)
    "prometheus-fastapi-instrumentator",
    "prometheus-client",
    "fastapi",
    "starlette",
    "uvicorn",
    "pydantic>=2.0",
    "pydantic-core",
    "annotated-doc",
    "anyio",
    "click",
    "h11",
    "typing-extensions",
    "typing-inspection",
    "annotated-types",
    "idna",

    # Outlines (vLLM structured output)
    "outlines>=0.0.43,<0.1",
    "outlines-core",
    "lm-format-enforcer",
    "partial-json-parser",
    "interegular",
    "cloudpickle",
    "diskcache",
    "genson",
    "jinja2",
    "jsonpath-ng",
    "jsonschema",
    "jsonschema-specifications",
    "referencing",
    "rpds-py",
    "attrs",
    "markupsafe",
    "ply",
    "pillow",
    "pyyaml",
    "packaging",

    # OpenAI client (vLLM uses this)
    "openai>=1.40.0",
    "httpx",
    "httpcore",
    "certifi",
    "sniffio",
    "distro",
    "jiter",

    # Other vLLM deps
    "tiktoken",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "sentencepiece",
    "protobuf",
    "regex",
    "filelock",
    "fsspec",
    "tqdm",
    "requests",
    "urllib3",
    "charset-normalizer",

    # CUDA deps (may already be in Kaggle)
    "triton",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvtx-cu12",
    "nvidia-nvjitlink-cu12",

    # ML core
    "torch>=2.4.0",
    "transformers>=4.40.0",
    "accelerate",
    "numpy",
    "scipy",
    "einops",
]

def main():
    print("=" * 60)
    print("vLLM Complete Wheels Downloader for Kaggle")
    print("=" * 60)

    # Clean and create dir
    if os.path.exists(WHEEL_DIR):
        shutil.rmtree(WHEEL_DIR)
    os.makedirs(WHEEL_DIR)

    print(f"\nDownloading to: {WHEEL_DIR}")
    print(f"Packages: {len(PACKAGES)}")
    print("-" * 60)

    # Download all wheels
    cmd = [
        sys.executable, "-m", "pip", "download",
        "-d", WHEEL_DIR,
        "--only-binary=:all:",
        "--platform", "manylinux2014_x86_64",
        "--platform", "manylinux_2_17_x86_64",
        "--platform", "manylinux_2_28_x86_64",
        "--python-version", "311",
    ] + PACKAGES

    print("Running pip download (this may take a few minutes)...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Some packages failed:\n{result.stderr[:1000]}")
        print("\nTrying packages individually...")

        for pkg in PACKAGES:
            pkg_cmd = [
                sys.executable, "-m", "pip", "download",
                "-d", WHEEL_DIR,
                "--only-binary=:all:",
                pkg
            ]
            subprocess.run(pkg_cmd, capture_output=True)

    # Count wheels
    wheels = [f for f in os.listdir(WHEEL_DIR) if f.endswith('.whl')]
    total_size = sum(os.path.getsize(os.path.join(WHEEL_DIR, f)) for f in wheels)

    print("-" * 60)
    print(f"Downloaded: {len(wheels)} wheels")
    print(f"Total size: {total_size / 1e6:.1f} MB")
    print(f"Location: {WHEEL_DIR}")
    print("-" * 60)

    # Create upload instructions
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print(f"""
1. Create Kaggle dataset:
   kaggle datasets init -p {WHEEL_DIR}

2. Edit {WHEEL_DIR}/dataset-metadata.json:
   - Set "title": "vllm-complete-wheels"
   - Set "id": "YOUR_USERNAME/vllm-complete-wheels"

3. Upload to Kaggle:
   kaggle datasets create -p {WHEEL_DIR}

4. In your notebook, add as Input:
   /kaggle/input/vllm-complete-wheels
""")

if __name__ == "__main__":
    main()
