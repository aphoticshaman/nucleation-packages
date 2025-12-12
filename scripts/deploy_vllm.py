#!/usr/bin/env python3
"""
Start vLLM server for Elle-72B-Ultimate on RunPod.

Usage:
    python deploy_vllm.py --model /workspace/elle-merged

Or with HuggingFace model:
    python deploy_vllm.py --model aphoticshaman/Elle-72B-Ultimate

This script configures and launches vLLM optimized for 4x H200 GPUs.
"""

import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Deploy Elle with vLLM")
    parser.add_argument("--model", default="/workspace/elle-merged", help="Model path or HF repo")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size (GPUs)")
    parser.add_argument("--max-model-len", type=int, default=32768, help="Max context length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95, help="GPU memory usage")
    args = parser.parse_args()

    # Build vLLM command
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tp),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--enforce-eager",  # More stable for large models
        "--disable-log-requests",  # Reduce log spam
    ]

    print("=" * 60)
    print("  Elle-72B-Ultimate vLLM Deployment")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Tensor Parallel: {args.tp} GPUs")
    print(f"Max Context: {args.max_model_len} tokens")
    print(f"Endpoint: http://{args.host}:{args.port}")
    print("=" * 60)
    print("\nStarting vLLM server...")
    print(f"Command: {' '.join(cmd)}\n")

    # Run vLLM
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except subprocess.CalledProcessError as e:
        print(f"Error: vLLM exited with code {e.returncode}")
        raise


if __name__ == "__main__":
    main()
