#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE='eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595'
NAME='vllm-nvidia-fp4-local'
REVISION='b9ce18a6887e18c089fdea8509a33e63429a3300'
SNAPSHOT="hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/$REVISION"
CACHE="$HOME/.cache/nvidia-fp4-local"

test -f "$HOME/.cache/huggingface/$SNAPSHOT/config.json"
docker image inspect "$IMAGE" >/dev/null
if docker container inspect "$NAME" >/dev/null 2>&1; then
    echo "Container $NAME already exists; inspect it before retrying." >&2
    exit 1
fi
if ss -ltn 'sport = :8001' | grep -q LISTEN; then
    echo 'Port 8001 is occupied; stop the previous backend first.' >&2
    exit 1
fi
mkdir -p "$CACHE"/{vllm,flashinfer,triton,b12x}

docker run -d --pull never --name "$NAME" --init --restart unless-stopped \
    --gpus all --ipc host --network host --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$CACHE/vllm:/root/.cache/vllm" \
    -v "$CACHE/flashinfer:/root/.cache/flashinfer" \
    -v "$CACHE/triton:/root/.triton" \
    -v "$CACHE/b12x:/root/.cache/b12x" \
    -e HF_HUB_OFFLINE=1 \
    -e VLLM_PLE_TABLE_MEMORY=disk \
    -e B12X_DISK_BACKEND=io_uring \
    -e CUTE_DSL_ARCH=sm_121a \
    -e SAFETENSORS_FAST_GPU=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
    -e VLLM_USE_AOT_COMPILE=1 \
    -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e B12X_POLICY_MODE=auto \
    -e VLLM_MXFP8_LM_HEAD=1 \
    -e VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 \
    -e VLLM_B12X_MOE_FP4_FORCE_A16=1 \
    -e B12X_W4A16_A4_PREFILL=1 \
    --entrypoint /usr/local/bin/vllm "$IMAGE" \
    serve "/root/.cache/huggingface/$SNAPSHOT" \
    --host 127.0.0.1 --port 8001 --served-model-name ornith \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 \
    --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \
    --dtype bfloat16 --kv-cache-dtype fp8 --quantization modelopt_mixed \
    --block-size 16 --load-format b12x \
    --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 4096 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"standard"}' \
    --gdn-decode-kernel b12x --linear-backend b12x --moe-backend b12x \
    --no-enable-flashinfer-autotune --mm-encoder-tp-mode data \
    --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --compilation-config '{"pass_config":{"fuse_act_quant":true}}' \
    --gpu-memory-utilization 0.8
