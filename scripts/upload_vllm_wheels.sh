#!/bin/bash
# Download and upload vLLM wheels to Kaggle for AIMO3 offline submission
# Run this on RunPod or any machine with Kaggle CLI configured

set -e

DEST_DIR="${1:-/workspace/vllm_kaggle_upload}"
KAGGLE_ID="${2:-aphoticshaman/vllm-offline-wheels}"

echo "=== vLLM Wheels for Kaggle AIMO3 ==="
echo "Destination: $DEST_DIR"
echo "Kaggle Dataset: $KAGGLE_ID"

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

echo ""
echo "Downloading vLLM 0.6.2 wheel..."
pip download vllm==0.6.2 --dest . --no-deps

echo ""
echo "Downloading dependencies..."
pip download xformers==0.0.27.post2 --dest . --no-deps
pip download ray==2.9.3 --dest . --no-deps
pip download msgspec --dest . --no-deps
pip download pyzmq --dest . --no-deps
pip download nvidia-ml-py --dest . --no-deps
pip download compressed-tensors==0.5.0 --dest . --no-deps

echo ""
echo "Creating dataset metadata..."
cat > dataset-metadata.json << EOF
{
  "title": "vLLM Offline Wheels",
  "id": "$KAGGLE_ID",
  "licenses": [{"name": "Apache 2.0"}]
}
EOF

echo ""
echo "Files ready:"
ls -lh "$DEST_DIR"

echo ""
echo "Total size:"
du -sh "$DEST_DIR"

echo ""
echo "=== To upload to Kaggle, run: ==="
echo "cd $DEST_DIR && kaggle datasets create -p . --dir-mode zip"
echo ""
echo "Or to update existing dataset:"
echo "cd $DEST_DIR && kaggle datasets version -p . -m 'Update vLLM wheels' --dir-mode zip"
