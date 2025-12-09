#!/usr/bin/env python3
"""
Upload training data to HuggingFace Hub for Axolotl training.

Usage:
    export HF_TOKEN=your_token
    python upload_to_hf.py training_data.jsonl your-username/latticeforge-briefing-data

Then use the dataset in RunPod Axolotl config.
"""

import json
import sys
import os

try:
    from huggingface_hub import HfApi, create_repo
    from datasets import Dataset
except ImportError:
    print("pip install huggingface_hub datasets")
    sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print("Usage: python upload_to_hf.py <input.jsonl> <repo-id>")
        print("Example: python upload_to_hf.py training_data.jsonl myuser/latticeforge-briefing-data")
        sys.exit(1)

    input_file = sys.argv[1]
    repo_id = sys.argv[2]

    token = os.environ.get('HF_TOKEN')
    if not token:
        print("ERROR: Set HF_TOKEN environment variable")
        print("Get your token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    # Load data
    print(f"Loading {input_file}...")
    examples = []
    with open(input_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples")

    # Convert to HuggingFace Dataset format
    # Using 'sharegpt' format which Axolotl understands
    dataset_records = []
    for ex in examples:
        conversations = ex.get('conversations', [])
        if conversations:
            dataset_records.append({"conversations": conversations})

    if not dataset_records:
        print("ERROR: No valid examples found. Make sure data has 'conversations' field.")
        print("Run prepare_data.py first to convert your data.")
        sys.exit(1)

    dataset = Dataset.from_list(dataset_records)
    print(f"Created dataset with {len(dataset)} examples")

    # Create repo if needed
    api = HfApi(token=token)
    try:
        create_repo(repo_id, repo_type="dataset", private=True, token=token)
        print(f"Created private dataset repo: {repo_id}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Repo {repo_id} already exists, will update")
        else:
            raise

    # Push to hub
    print(f"Uploading to {repo_id}...")
    dataset.push_to_hub(repo_id, token=token, private=True)
    print(f"\n✅ Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")
    print(f"\nUse in Axolotl config:")
    print(f'  "datasets": [{{"path": "{repo_id}", "type": "sharegpt"}}]')


if __name__ == '__main__':
    main()
