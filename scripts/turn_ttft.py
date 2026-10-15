#!/usr/bin/env python3
"""Multi-turn TTFT probe: the metric MTP_DISABLE_BLOCK_DROP improves.
Runs N sessions of 3 turns over a ~4k-token context; reports median TTFT per turn.
Turn 1 is cold; turns 2-3 should hit the prefix cache."""
import argparse, json, random, statistics, string, time, urllib.request

def stream_ttft(api, model, messages, key=None):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": 64,
                       "temperature": 0, "stream": True,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    hdr = {"Content-Type": "application/json"}
    if key: hdr["Authorization"] = f"Bearer {key}"
    t0 = time.perf_counter(); ttft = None; text = []
    with urllib.request.urlopen(urllib.request.Request(api, body, hdr), timeout=300) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d = json.loads(line[5:]).get("choices", [{}])[0].get("delta", {})
            piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
            if piece and ttft is None: ttft = time.perf_counter() - t0
            text.append(d.get("content") or "")
    return ttft or float("nan"), "".join(text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8001/v1/chat/completions")
    ap.add_argument("--model", default="ornith")
    ap.add_argument("--key", default=None)
    ap.add_argument("--sessions", type=int, default=3)
    a = ap.parse_args()
    per_turn = {1: [], 2: [], 3: []}
    for s in range(a.sessions):
        salt = "".join(random.choices(string.ascii_lowercase, k=12))  # unique prefix per session
        code = "\n".join(f"def handler_{salt}_{i}(req):\n    return process(req, level={i})" for i in range(350))
        msgs = [{"role": "system", "content": f"Session {salt}. You are a code reviewer."},
                {"role": "user", "content": f"Here is a module:\n```python\n{code}\n```\nSummarise it in one line."}]
        for turn, follow in ((1, None), (2, "Which function has the highest level?"), (3, "Name one risk in two sentences.")):
            if follow: msgs.append({"role": "user", "content": follow})
            t, reply = stream_ttft(a.api, a.model, msgs, a.key)
            per_turn[turn].append(t)
            msgs.append({"role": "assistant", "content": reply or "ok"})
    for turn, v in per_turn.items():
        print(f"turn{turn}_ttft_median_s={statistics.median(v):.2f} values={[round(x,2) for x in v]}")

if __name__ == "__main__":
    main()
