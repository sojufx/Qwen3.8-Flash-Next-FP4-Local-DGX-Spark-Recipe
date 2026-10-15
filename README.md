# Qwen3.8 Flash-Next FP4-Local on One DGX Spark

A reproducible Sojufx serving recipe for NVIDIA's **Qwen3.8-Flash-Next-FP4-Local** checkpoint on a single DGX Spark / GB10. Native 262,144-token context, FP8 KV, exact MTP K=3, four scheduler slots, vision, tools, and a localhost-only OpenAI-compatible backend.

Measured on our Spark: **66.2 tok/s single-stream counting decode**, **166.4 tok/s combined at C4**, and approximately **41-54 tok/s single-stream across our four-prompt wall-clock suite**. These are different measurements, not interchangeable performance claims. See [benchmarks](benchmarks/RESULTS.md).

This repository contains launch and verification scripts, not model weights or a new quantization. NVIDIA supplies the FP4-Local checkpoint; Qwen supplies the base model. This is separate from our older TokenV3 recipe: **no TokenV3 overlay, permissive alpha, or reduced draft vocabulary is applied here**.

## Tested Configuration

| Component | Setting |
|---|---|
| Model | `nvidia/Qwen3.8-Flash-Next-FP4-Local` |
| Model revision | `b9ce18a6887e18c089fdea8509a33e63429a3300` |
| Runtime | Pinned `eugr/spark-vllm-b12x` image digest in `start.sh` |
| Reported vLLM version | `0.1.dev21830+g8244d1c44.d20261015` |
| Context limit | 262,144 tokens, no extension |
| KV cache | FP8 |
| MTP | K=3, `rejection_sample_method=standard` (exact) |
| Maximum sequences | 4 |
| Batched prefill | 4,096 tokens |
| GPU memory utilization | 0.80 |
| Prefix cache | Enabled, Mamba cache mode `align` |
| PLE table | Disk-backed, `io_uring` |
| Kernels | B12X linear, MoE and GDN |
| Backend | `127.0.0.1:8001`, alias `qwen38-flash-next-fp4-local` |
| Tool parser / reasoning parser | `qwen3_xml` / `qwen3` |
| Thinking | Off by default; can be requested explicitly |
| Container restart | `unless-stopped` |

The checkpoint has a new compressed NVFP4 scale format. Use the compatible pinned image, not an arbitrary stock vLLM container. The model card describes quantization-aware distillation and approximately 96 GiB of weights. Allow substantial additional SSD space for runtime images and caches. Read the upstream terms before downloading.

## Install

Requires a Linux DGX Spark, working NVIDIA driver, Docker with NVIDIA GPU support, internet access for the initial download, and permission to use Docker. The container uses host networking/IPC and relaxed seccomp/AppArmor for this tested runtime: treat Docker access as privileged and use a trusted host.

```bash
git clone https://github.com/sojufx/Qwen3.8-Flash-Next-FP4-Local-DGX-Spark-Recipe.git
cd Qwen3.8-Flash-Next-FP4-Local-DGX-Spark-Recipe
bash download.sh
```

After downloading, stop any existing GPU model server and free port 8001. The launcher deliberately refuses to replace an existing container or take an occupied port.

```bash
bash start.sh
docker logs -f vllm-nvidia-fp4-local
```

First startup took about 14 minutes on our host, including compilation and warm-up. Caches are retained in `~/.cache/nvidia-fp4-local/`; subsequent timing depends on cache reuse. Wait for serving readiness, not merely a running container.

```bash
python3 scripts/smoke.py
curl -fsS http://127.0.0.1:8001/v1/models
ss -ltn 'sport = :8001'
```

The socket must show **127.0.0.1:8001**, not `0.0.0.0`. There is intentionally no backend API key because access is local-only. LAN/public clients must use an authenticated gateway. Do not expose this raw port through a tunnel, router or reverse proxy without authentication.

## Persistent Operation

`start.sh` records `--restart unless-stopped` in the container. Enable the Docker daemon at boot if it is not already enabled:

```bash
sudo systemctl enable docker
docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' vllm-nvidia-fp4-local
```

Docker can restart this container after a reboot unless it was manually stopped. This policy is verified; a power-cycle recovery test has not been performed for this profile. Disable competing model auto-start services before enabling this one. Your gateway must also start at boot and tolerate backend warm-up.

To apply launcher changes, stop and remove **only this container**, then launch it again. Weights and host-mounted caches remain:

```bash
docker stop vllm-nvidia-fp4-local
docker rm vllm-nvidia-fp4-local
bash start.sh
```

Keep your previous model image, launcher, checkpoint and caches for rollback. Never leave two large model engines starting together.

## Stable Gateway Contract

Keep one public URL, API key and model alias across model swaps. Route your authenticated gateway to `http://127.0.0.1:8001`; the backend alias is `qwen38-flash-next-fp4-local`. No personal hostname, IP address or serving key is included here.

Our existing production gateway supplies the policy below. **The gateway implementation is not bundled. These are proxy-side settings, not settings installed by `start.sh`.** Deploy equivalent middleware or configure your existing gateway if you need them.

| Default when omitted by client | Thinking off | Thinking on |
|---|---:|---:|
| `temperature` | 0.7 | 1.0 |
| `top_p` | 0.8 | 0.95 |
| `top_k` | 20 | 20 |
| `presence_penalty` | 1.5 | 0 |

- For a nonempty `tools` list or a `response_format`, default `presence_penalty` to **0**, not 1.5.
- Preserve explicit client sampling values, except our gateway raises a requested temperature below 0.3 to 0.3. This is our deployment policy, not a universal requirement.
- Default/cap total completion tokens at **32,768**, handling both `max_tokens` and `max_completion_tokens`.
- For reasoning requests, pass a **thinking token budget of 8,192** to a supporting backend. Verify enforcement when changing runtime. Reasoning shares the total completion budget; it is not an additional 8K allowance.
- Thinking is off by default; our guard maps ordinary explicit thinking requests to medium. An explicit disable remains off.
- Authenticate all gateway routes, including model discovery; preserve SSE, tool-call fields, reasoning fields and upstream errors.

These policies reduce some repetition and bound long generations; they do not guarantee agent completion or fix broken tool loops. The direct backend has none of these proxy caps or sampling overrides.

## Reproduce The Measurements

Run on the Spark against loopback so network/proxy overhead does not change the comparison. The benchmark's explicit temperature 0 bypasses the gateway temperature floor.

```bash
python3 scripts/benchmark_flash_next.py \
  --api http://127.0.0.1:8001/v1/chat/completions --model qwen38-flash-next-fp4-local \
  --all-prompts --concurrency 1,4 --repeat 2 --max-tokens 256 --temperature 0

python3 scripts/turn_ttft.py --sessions 3
```

The four-prompt script is retained unchanged for comparability. `agent_tool` requests JSON but does not execute real tools. `long_context_code` is a short production-review note with about 257 prompt tokens, **not a long-context benchmark**. Some answers hit the output cap. The suite measures throughput, not task correctness. It retains historical example service settings as prompt text, not as recommendations.

For the separate post-first-token counting decode test, use [MiaAI Lab's sparkDash](https://github.com/MiaAI-Lab/sparkDash): Structured, 400 output tokens, temperature 0, top_p 1, thinking off, C1/C2/C4. Our run used sparkDash 1.8.6. Point it at loopback on the Spark, or use an authenticated gateway with its policy differences disclosed. Do not open port 8001 to run a dashboard remotely.

For an existing sparkDash checkout with its dependencies installed, the included runner calls its benchmark directly (requires a compatible Node runtime):

```bash
SPARKDASH_DIR=/path/to/sparkDash node scripts/run-sparkdash.mjs
```

Run this on the Spark. Results are written to the ignored `results-local/` directory. The measured checkout was version 1.8.6; future sparkDash changes may alter its API or protocol.

## Validation And Limits

Production canaries passed for model discovery, text, SSE, one function call with parsed JSON arguments, a mock tool-error follow-up without retry, JSON output, a synthetic red image, and explicit thinking with a correct arithmetic answer. These are narrow checks, **not a quality evaluation**. Video and deep-context retrieval were not evaluated in this run.

The earlier K=4 startup reported about 1.09M KV-token capacity. That does not establish four simultaneous full-context users, and should not be presented as a K=3 capacity measurement. `max_num_seqs=4` is scheduler capacity, not a tested four-user 262K service guarantee.

Follow-up TTFT was measured at K=4: 4.16 s cold, 0.22 s on turns 2 and 3, medians of three sessions. It has not yet been remeasured at K=3. We did not disable speculative block dropping: the compatible runtime already produced fast follow-ups. Do not blindly carry the older recipe's block-drop patch into this image.

## Credits And Licenses

- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): base model.
- [NVIDIA FP4-Local model card](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local): checkpoint, compatible-runtime requirements and B12X reference launch configuration. This recipe adapts that configuration with a secure bind, pinned assets, K=3 and four scheduler slots.
- [eugr/spark-vllm-b12x](https://hub.docker.com/r/eugr/spark-vllm-b12x): compatible runtime container and its upstream contributors.
- [vLLM](https://github.com/vllm-project/vllm): serving engine.
- [MiaAI Lab / sparkDash](https://github.com/MiaAI-Lab/sparkDash): the separately reported counting decode benchmark; earlier community work informed our operational testing.

Original recipe code is MIT licensed. Model weights, runtime components and third-party benchmark software retain their own licenses. NVIDIA's model agreement and the upstream Qwen license apply to the weights; this repository does not relicense them. No NVIDIA, Qwen or MiaAI endorsement is implied.
