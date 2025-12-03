#!/usr/bin/env python3
"""
Merge all training data files into a single combined dataset.
Run this to create one file for upload to Pod 2.
"""
import json
import os
from pathlib import Path

def main():
    script_dir = Path(__file__).parent

    # Training data files to merge
    files_to_merge = [
        "global_intel_training.json",
        "specialized_intel_training.json",
        "deep_intel_training.json",
        "market_intel_training.json",
        "geopolitical_risk_training.json",
        "cascade_history_training.json",
        "energy_materials_training.json",
        "freedom_coercion_training.json",
        "tech_materials_systems_training.json",
        "meta_compression_training.json",
    ]

    all_examples = []
    stats = {}

    print("=" * 70)
    print("MERGING TRAINING DATA FILES")
    print("=" * 70)

    for filename in files_to_merge:
        filepath = script_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
                count = len(data)
                all_examples.extend(data)
                stats[filename] = count
                print(f"  + {filename}: {count:,} examples")
        else:
            print(f"  - {filename}: NOT FOUND")

    print(f"\n{'=' * 70}")
    print(f"TOTAL EXAMPLES: {len(all_examples):,}")
    print(f"{'=' * 70}")

    # Save combined
    output_path = script_dir / "combined_intel_training.json"
    with open(output_path, 'w') as f:
        json.dump(all_examples, f, indent=2)

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved to: {output_path}")
    print(f"File size: {file_size:.2f} MB")

    print(f"\n{'=' * 70}")
    print("UPLOAD TO POD 2:")
    print("=" * 70)
    print(f"1. Download: combined_intel_training.json ({file_size:.2f} MB)")
    print("2. Upload to /workspace/ on Pod 2")
    print("3. Update train_combined.py to use combined_intel_training.json")
    print("   OR run with: python train_combined.py --intel combined_intel_training.json")

    return all_examples

if __name__ == "__main__":
    main()
