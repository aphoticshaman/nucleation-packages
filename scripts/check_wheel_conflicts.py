#!/usr/bin/env python3
"""
Check wheel conflicts against Kaggle's known packages.
Run this BEFORE uploading to Kaggle to catch all version mismatches at once.

Usage:
    python check_wheel_conflicts.py /path/to/wheels/
"""
import os
import sys
import re
from pathlib import Path
from packaging import version
from typing import Dict, List, Tuple, Optional

# Known Kaggle package versions (as of Dec 2024)
# These are packages Kaggle has pre-installed that we should NOT override
KAGGLE_PACKAGES = {
    # Core ML - NEVER override these
    'torch': '2.5.1',
    'transformers': '4.47.0',
    'tokenizers': '0.21.0',
    'numpy': '1.26.4',
    'scipy': '1.14.1',

    # CUDA/Triton - Kaggle has its own
    'triton': '3.1.0',
    'nvidia-cublas-cu12': '12.4.5.8',
    'nvidia-cuda-cupti-cu12': '12.4.127',
    'nvidia-cuda-nvrtc-cu12': '12.4.127',
    'nvidia-cuda-runtime-cu12': '12.4.127',
    'nvidia-cudnn-cu12': '9.1.0.70',
    'nvidia-cufft-cu12': '11.2.1.3',
    'nvidia-curand-cu12': '10.3.5.147',
    'nvidia-cusolver-cu12': '11.6.1.9',
    'nvidia-cusparse-cu12': '12.3.1.170',
    'nvidia-nccl-cu12': '2.21.5',
    'nvidia-nvjitlink-cu12': '12.4.127',
    'nvidia-nvtx-cu12': '12.4.127',

    # Common deps Kaggle has
    'pillow': '11.0.0',
    'pandas': '2.2.3',
    'polars': '1.16.0',
    'requests': '2.32.3',
    'urllib3': '2.2.3',
    'certifi': '2024.8.30',
    'charset-normalizer': '3.4.0',
    'idna': '3.10',
    'tqdm': '4.67.1',
    'pyyaml': '6.0.2',
    'packaging': '24.2',
    'filelock': '3.16.1',
    'fsspec': '2024.10.0',
    'regex': '2024.11.6',
    'safetensors': '0.4.5',
    'huggingface-hub': '0.26.5',
    'accelerate': '1.2.1',
    'datasets': '3.2.0',
    'evaluate': '0.4.3',
    'jinja2': '3.1.4',
    'markupsafe': '3.0.2',
    'typing-extensions': '4.12.2',
    'protobuf': '5.29.1',
    'sentencepiece': '0.2.0',
}

# Packages that MUST be skipped (causes immediate crashes)
CRITICAL_SKIP = {
    'torch', 'transformers', 'tokenizers', 'numpy', 'triton',
}

# Packages that are risky to override
RISKY_PACKAGES = {
    'scipy', 'pillow', 'pandas', 'polars', 'protobuf',
    'huggingface-hub', 'safetensors', 'accelerate',
}


def parse_wheel_name(wheel_path: str) -> Tuple[str, str]:
    """Extract package name and version from wheel filename."""
    name = os.path.basename(wheel_path)
    # Wheel format: {distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl
    match = re.match(r'^([A-Za-z0-9_]+(?:[._-][A-Za-z0-9_]+)*)-([0-9][^-]*)-', name)
    if match:
        pkg_name = match.group(1).lower().replace('_', '-')
        pkg_version = match.group(2)
        return pkg_name, pkg_version
    return None, None


def check_version_conflict(pkg: str, wheel_ver: str, kaggle_ver: str) -> Optional[str]:
    """Check if wheel version conflicts with Kaggle's version."""
    try:
        wv = version.parse(wheel_ver)
        kv = version.parse(kaggle_ver)

        if wv.major != kv.major:
            return f"MAJOR version mismatch: wheel={wheel_ver} vs kaggle={kaggle_ver}"
        if wv.minor != kv.minor:
            return f"MINOR version mismatch: wheel={wheel_ver} vs kaggle={kaggle_ver}"
        if wv != kv:
            return f"PATCH version mismatch: wheel={wheel_ver} vs kaggle={kaggle_ver}"
    except:
        if wheel_ver != kaggle_ver:
            return f"Version mismatch: wheel={wheel_ver} vs kaggle={kaggle_ver}"
    return None


def scan_wheels(wheel_dir: str) -> Dict[str, str]:
    """Scan directory for wheels and return {package: version} dict."""
    wheels = {}
    wheel_dir = Path(wheel_dir)

    for whl in wheel_dir.rglob('*.whl'):
        pkg, ver = parse_wheel_name(str(whl))
        if pkg and ver:
            wheels[pkg] = ver

    return wheels


def check_conflicts(wheel_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Check all wheels for conflicts.
    Returns: (critical_conflicts, risky_conflicts, warnings)
    """
    critical = []
    risky = []
    warnings = []

    wheels = scan_wheels(wheel_dir)

    print(f"\nScanned {len(wheels)} wheels in {wheel_dir}\n")
    print("=" * 70)

    for pkg, wheel_ver in sorted(wheels.items()):
        # Check against Kaggle's known packages
        if pkg in KAGGLE_PACKAGES:
            kaggle_ver = KAGGLE_PACKAGES[pkg]
            conflict = check_version_conflict(pkg, wheel_ver, kaggle_ver)

            if conflict:
                if pkg in CRITICAL_SKIP:
                    critical.append(f"CRITICAL - {pkg}: {conflict}")
                elif pkg in RISKY_PACKAGES:
                    risky.append(f"RISKY - {pkg}: {conflict}")
                else:
                    warnings.append(f"WARNING - {pkg}: {conflict}")

        # Check if it's a nvidia/triton package we should skip
        if pkg.startswith('nvidia-') or pkg == 'triton':
            if pkg not in KAGGLE_PACKAGES:
                risky.append(f"RISKY - {pkg}: nvidia package not in Kaggle baseline (may conflict)")

    return critical, risky, warnings


def generate_skip_list(wheel_dir: str) -> List[str]:
    """Generate recommended SKIP_PACKAGES list."""
    wheels = scan_wheels(wheel_dir)
    skip = set()

    for pkg in wheels:
        # Always skip critical packages
        if pkg in CRITICAL_SKIP:
            skip.add(pkg.split('-')[0])  # 'nvidia-xxx' -> 'nvidia'

        # Skip nvidia packages
        if pkg.startswith('nvidia'):
            skip.add('nvidia')

        # Check for conflicts with Kaggle
        if pkg in KAGGLE_PACKAGES:
            wheel_ver = wheels[pkg]
            kaggle_ver = KAGGLE_PACKAGES[pkg]
            conflict = check_version_conflict(pkg, wheel_ver, kaggle_ver)
            if conflict:
                skip.add(pkg.split('-')[0])

    return sorted(skip)


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_wheel_conflicts.py /path/to/wheels/")
        print("\nThis script checks wheels against Kaggle's known packages")
        print("to detect version conflicts BEFORE uploading.")
        sys.exit(1)

    wheel_dir = sys.argv[1]

    if not os.path.exists(wheel_dir):
        print(f"Error: Directory not found: {wheel_dir}")
        sys.exit(1)

    print("=" * 70)
    print("WHEEL CONFLICT CHECKER FOR KAGGLE")
    print("=" * 70)

    critical, risky, warnings = check_conflicts(wheel_dir)

    # Report critical issues
    if critical:
        print("\n🚨 CRITICAL CONFLICTS (MUST SKIP):")
        print("-" * 50)
        for c in critical:
            print(f"  {c}")

    # Report risky packages
    if risky:
        print("\n⚠️  RISKY PACKAGES (RECOMMEND SKIP):")
        print("-" * 50)
        for r in risky:
            print(f"  {r}")

    # Report warnings
    if warnings:
        print("\n📝 WARNINGS (May cause issues):")
        print("-" * 50)
        for w in warnings:
            print(f"  {w}")

    # Generate skip list
    skip_list = generate_skip_list(wheel_dir)

    print("\n" + "=" * 70)
    print("RECOMMENDED SKIP_PACKAGES for notebook:")
    print("=" * 70)
    print(f"\nSKIP_PACKAGES = {set(skip_list)}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_issues = len(critical) + len(risky) + len(warnings)
    print(f"  Critical conflicts: {len(critical)}")
    print(f"  Risky packages:     {len(risky)}")
    print(f"  Warnings:           {len(warnings)}")
    print(f"  Total issues:       {total_issues}")

    if critical:
        print("\n❌ BLOCKING: Fix critical conflicts before uploading!")
        print("   Add these to SKIP_PACKAGES in your notebook.")
        sys.exit(1)
    elif risky:
        print("\n⚠️  WARNING: Review risky packages before uploading.")
        sys.exit(0)
    else:
        print("\n✅ All clear! No obvious conflicts detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
