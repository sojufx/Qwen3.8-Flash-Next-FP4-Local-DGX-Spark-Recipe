#!/usr/bin/env python3
import argparse
import json
import statistics
import threading
import time
import urllib.request


PROMPTS = {
    "generic": "Write a compact Python function that validates an email address, then explain the edge cases in plain English.",
    "code_edit": """You are a senior Python engineer. Fix this bug with a minimal patch.

Bug report:
The trading bot sometimes opens a new BUY even after the daily risk lock is active. The intended behavior is: if risk_locked is true, no new BUY orders should be created, but existing SELL/close actions are allowed.

Current code:
```python
def should_place_order(signal, position, risk_locked):
    if signal == "SELL":
        return position.qty > 0
    if signal == "BUY" and position.qty == 0:
        return True
    if risk_locked:
        return False
    return False
```

Return:
1. The corrected function only.
2. A short root-cause explanation.
3. Three focused tests.
""",
    "agent_tool": """You are an autonomous coding agent working in a repo. Be concrete and structured.

Task:
A user reports: "Hermes says the response was truncated after 3 continuation attempts." Diagnose the most likely configuration cause in an OpenAI-compatible local vLLM setup and propose exact client-side settings to prevent it.

Return JSON with keys: root_cause, config_changes, verification_steps, risks.
""",
    "long_context_code": """You are reviewing a compact design note for a local LLM production service.

Context:
- Hardware: single DGX Spark / GB10 with 128 GB unified memory.
- Server: vLLM 0.27.2, OpenAI-compatible endpoint.
- Model: Ornith 1.5 35B A3B NVFP4.
- Production needs: 262k context, family users through Cloudflare, Hermes agent on LAN.
- Known constraints: high context reduces concurrency; thinking can slow agent jobs; prefix caching helps repeated agent prefixes; large prefill spikes can hang GB10 if batching is too large.
- Current flags: max_model_len 262144, max_num_seqs 16, max_num_batched_tokens 4096, kv_cache_dtype fp8_e4m3, prefix_match_unit 16, enable_prefix_caching, enable_thinking true, reasoning_effort medium.

Question:
Write a concise production-readiness review. Include what is safe, what is risky, what to benchmark, and what settings you would not change without an A/B test.
""",
}


def request_once(api, key, model, prompt, max_tokens, temperature, timeout, idx, results):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(api, data=json.dumps(payload).encode(), headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    elapsed = time.time() - t0
    usage = body.get("usage", {})
    output_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    if output_tokens is None:
        output_tokens = len(body["choices"][0]["message"].get("content", "").split())
    results[idx] = {
        "elapsed": elapsed,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "tok_s": output_tokens / elapsed if elapsed else 0.0,
        "finish_reason": body.get("choices", [{}])[0].get("finish_reason"),
    }


def run_concurrency(concurrency, args, prompt):
    results = {}
    threads = [
        threading.Thread(
            target=request_once,
            args=(args.api, args.key, args.model, prompt, args.max_tokens, args.temperature, args.timeout, i, results),
        )
        for i in range(concurrency)
    ]
    wall0 = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if len(results) != concurrency:
        raise RuntimeError("One or more benchmark requests failed; refusing a partial throughput result")
    wall = time.time() - wall0
    total_tokens = sum(result["output_tokens"] for result in results.values())
    per_stream = [results[i]["tok_s"] for i in sorted(results)]
    prompt_tokens = [results[i].get("prompt_tokens") for i in sorted(results)]
    finish = [results[i].get("finish_reason") for i in sorted(results)]
    return {
        "concurrency": concurrency,
        "aggregate_tok_s": total_tokens / wall if wall else 0.0,
        "wall_s": wall,
        "per_stream_tok_s": per_stream,
        "prompt_tokens": prompt_tokens,
        "tokens": [results[i]["output_tokens"] for i in sorted(results)],
        "finish": finish,
    }


def fetch_server_info(api, key):
    try:
        url = api.rstrip("/").replace("/v1/chat/completions", "/v1/models")
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
        item = body.get("data", [{}])[0]
        return {"served_model": item.get("id"), "root": item.get("root"), "max_model_len": item.get("max_model_len")}
    except Exception as exc:
        return {"server_info_error": str(exc)}


def print_row(row):
    per_stream = ",".join(f"{value:.1f}" for value in row["per_stream_tok_s"])
    print(
        "C{}: aggregate={:.1f} tok/s wall={:.2f}s per_stream=[{}] prompt_tokens={} output_tokens={} finish={}".format(
            row["concurrency"],
            row["aggregate_tok_s"],
            row["wall_s"],
            per_stream,
            row["prompt_tokens"],
            row["tokens"],
            row["finish"],
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Fixed prompt-suite benchmark for OpenAI-compatible vLLM servers.")
    parser.add_argument("--api", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="ornith")
    parser.add_argument("--key", default=None)
    parser.add_argument("--prompt-name", choices=sorted(PROMPTS), default="generic")
    parser.add_argument("--all-prompts", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--concurrency", default="1,2,3")
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()

    concurrencies = [int(item.strip()) for item in args.concurrency.split(",") if item.strip()]
    prompt_names = sorted(PROMPTS) if args.all_prompts else [args.prompt_name]

    print("server_info:", json.dumps(fetch_server_info(args.api, args.key), sort_keys=True))
    print("max_tokens:", args.max_tokens, "temperature:", args.temperature, "repeat:", args.repeat, "concurrency:", concurrencies)

    for prompt_name in prompt_names:
        prompt = PROMPTS[prompt_name]
        print("\n=== prompt:", prompt_name, "===")
        print("prompt_chars:", len(prompt))
        all_rows = []
        for repeat_index in range(1, args.repeat + 1):
            print(f"run {repeat_index}/{args.repeat}")
            for concurrency in concurrencies:
                row = run_concurrency(concurrency, args, prompt)
                all_rows.append(row)
                print_row(row)
        print("summary_median_by_concurrency")
        for concurrency in concurrencies:
            values = [row["aggregate_tok_s"] for row in all_rows if row["concurrency"] == concurrency]
            print(f"C{concurrency}: median={statistics.median(values):.1f} tok/s values={[round(value, 1) for value in values]}")


if __name__ == "__main__":
    main()
