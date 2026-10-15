#!/usr/bin/env python3
"""Readiness and text check. API_KEY is optional for a gateway."""
import argparse
import json
import os
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="qwen38-flash-next-fp4-local")
    args = parser.parse_args()
    headers = {"Content-Type": "application/json"}
    if os.environ.get("API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["API_KEY"]

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(args.base_url.rstrip("/") + path, data, headers)
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)

    models = request("/models")
    if args.model not in [item["id"] for item in models["data"]]:
        raise RuntimeError("Expected served model alias was not returned")
    result = request("/chat/completions", {
        "model": args.model, "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
        "max_tokens": 32, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    content = result["choices"][0]["message"].get("content") or ""
    if "ready" not in content.lower():
        raise RuntimeError("Readiness completion did not return the expected text")
    print("Model discovery and text completion PASS")


if __name__ == "__main__":
    main()
