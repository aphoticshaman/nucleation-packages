#!/usr/bin/env python3
"""
Prepare interleaved training dataset for Elle.
Spreads PROMETHEUS/NSM framework examples throughout code training data.
"""

import json
import random
from datasets import load_dataset

# Config
CODE_DATASET = "nickrosh/Evol-Instruct-Code-80k-v1"
CODE_SPLIT = "train[:30000]"
FRAMEWORK_FILE = "/workspace/datasets/prometheus_nsm_training.jsonl"
OUTPUT_FILE = "/workspace/datasets/elle_interleaved_training.jsonl"
REPEAT_FRAMEWORK = 3  # Repeat framework examples 3x throughout

def load_framework_examples():
    """Load PROMETHEUS/NSM/XYZA/SDPM framework examples."""
    examples = []
    with open(FRAMEWORK_FILE, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    print(f"Loaded {len(examples)} framework examples")
    return examples

def load_code_examples():
    """Load code training examples from HuggingFace."""
    ds = load_dataset(CODE_DATASET, split=CODE_SPLIT)
    examples = []
    for row in ds:
        examples.append({
            "instruction": row.get("instruction", ""),
            "input": row.get("input", ""),
            "output": row.get("output", "")
        })
    print(f"Loaded {len(examples)} code examples")
    return examples

def interleave_datasets(code_examples, framework_examples, repeat=3):
    """
    Interleave framework examples throughout code examples.
    Framework examples are repeated 'repeat' times and spread evenly.
    """
    # Repeat framework examples
    repeated_framework = framework_examples * repeat
    random.shuffle(repeated_framework)

    n_code = len(code_examples)
    n_framework = len(repeated_framework)

    # Calculate spacing
    # Insert one framework example every N code examples
    spacing = n_code // (n_framework + 1)

    print(f"Interleaving {n_framework} framework examples into {n_code} code examples")
    print(f"Spacing: 1 framework example every ~{spacing} code examples")

    result = []
    framework_idx = 0

    for i, code_ex in enumerate(code_examples):
        result.append(code_ex)

        # Insert framework example at regular intervals
        if (i + 1) % spacing == 0 and framework_idx < n_framework:
            result.append(repeated_framework[framework_idx])
            framework_idx += 1

    # Add any remaining framework examples
    while framework_idx < n_framework:
        result.append(repeated_framework[framework_idx])
        framework_idx += 1

    print(f"Final dataset: {len(result)} examples")
    return result

def save_dataset(examples, output_file):
    """Save interleaved dataset as JSONL."""
    with open(output_file, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    print(f"Saved to {output_file}")

def main():
    random.seed(42)

    print("=" * 60)
    print("Preparing Interleaved Training Dataset")
    print("=" * 60)

    # Load datasets
    framework_examples = load_framework_examples()
    code_examples = load_code_examples()

    # Interleave
    interleaved = interleave_datasets(code_examples, framework_examples, repeat=REPEAT_FRAMEWORK)

    # Save
    save_dataset(interleaved, OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("Dataset ready for training!")
    print(f"Total examples: {len(interleaved)}")
    print(f"  - Code: {len(code_examples)}")
    print(f"  - Framework (×{REPEAT_FRAMEWORK}): {len(framework_examples) * REPEAT_FRAMEWORK}")
    print("=" * 60)

if __name__ == "__main__":
    main()
