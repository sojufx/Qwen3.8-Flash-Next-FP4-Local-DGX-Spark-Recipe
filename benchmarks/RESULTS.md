# Single DGX Spark Results

Recorded during the October 15-16, 2026 deployment session. One GB10, pinned model/runtime from `start.sh`, FP8 KV, 262,144 context limit, four scheduler slots, disk-backed PLE, GPU memory utilization 0.80. All benchmark requests go directly to loopback, thinking off. These are local speed results, not quality scores or cross-hardware promises.

## Four-Prompt Suite: K=3 Versus K=4

Two repeats per cell, reported median. 256-token cap, temperature 0, non-streaming. Completion tokens divided by request wall time at C1; total completion tokens divided by concurrent wall time at C4. Prefill is included. No failed requests in these runs.

| Prompt | K=4 C1 | K=3 C1 | K=4 C4 aggregate | K=3 C4 aggregate |
|---|---:|---:|---:|---:|
| Agent/tool JSON prompt | 40.1 | 40.8 | 87.7 | 94.5 |
| Code edit | 51.0 | 48.8 | 115.0 | 124.6 |
| Generic coding | 53.8 | 54.3 | 120.6 | 125.1 |
| Production review | 41.2 | 44.2 | 94.4 | 104.1 |

Units: tok/s. K=3 improved C4 by roughly 4-10% across these four cells, while code-edit C1 decreased. Two repeats are not sufficient to establish statistical significance. This was a sequential comparison, not an interleaved controlled experiment; live traffic and cache state can affect results.

The agent/tool prompt does not execute tools. Production review is a short note, not deep-context retrieval. The generic prompt produced different output lengths across runs; some outputs end naturally and others hit the cap. Even temperature 0 does not make different speculative/batching runs text-identical.

## sparkDash Structured Counting Decode: K=3

sparkDash 1.8.6 `DecodeBenchManager`, Structured, counting numbers, 400-token cap, temperature 0, top_p 1, thinking off, 32-token warm-up. One wave per concurrency, seven requests total. No failed requests; all streams reached the output cap.

| Concurrency | Aggregate decode tok/s | Mean stream decode tok/s | Mean TTFT |
|---|---:|---:|---:|
| 1 | 66.17 | 66.17 | 42.17 ms |
| 2 | 106.27 | 53.40 | 470.40 ms |
| 4 | 166.36 | 42.22 | 507.23 ms |

Decode uses the post-first-token window, not end-to-end wall time. Concurrent prompts include stream labels. Counting is highly predictable and favors speculative decoding. This is not a coding, tool-use, or quality benchmark. These values must not be compared as equivalent to the four-prompt suite above.

## Follow-Up TTFT: K=4, Not Yet Rerun At K=3

`turn_ttft.py`, three sessions, unique synthetic code prefix per session, three turns, 64-token output cap, temperature 0, thinking off. First nonempty streamed content/reasoning chunk is the timing boundary; these requests have thinking disabled. The script's approximate context-size description is not a tokenizer measurement.

| Turn | Median TTFT | Individual results |
|---|---:|---|
| Cold first turn | 4.16 s | 4.16, 4.14, 4.29 |
| Follow-up 1 | 0.22 s | 0.22, 0.22, 0.23 |
| Follow-up 2 | 0.22 s | 0.22, 0.22, 0.23 |

No block-drop override was applied. Do not label these K=3 results.
