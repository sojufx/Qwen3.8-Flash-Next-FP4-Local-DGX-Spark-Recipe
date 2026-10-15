#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE='eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595'
mkdir -p "$HOME/.cache/huggingface"
docker pull "$IMAGE"
docker run --rm --pull never \
    --user "$(id -u):$(id -g)" \
    -e HF_HOME=/hf -e HF_TOKEN -e PYTHONUNBUFFERED=1 \
    -v "$HOME/.cache/huggingface:/hf" \
    --entrypoint python3 "$IMAGE" -c '
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="nvidia/Qwen3.8-Flash-Next-FP4-Local",
    revision="b9ce18a6887e18c089fdea8509a33e63429a3300",
    max_workers=4,
), flush=True)
'
echo 'Pinned image and model downloaded. No production service was changed.'
