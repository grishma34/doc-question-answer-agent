# Architecture

## Overview

Two pipelines share one local index:

```
INGEST (offline, run once per corpus change)
  .md/.txt files
      │  chunking.py — paragraph-aware, ~1200 chars, 200 overlap,
      │                stable ids "<doc>#<n>"
      ▼
  Bedrock Titan Embeddings V2 (512-dim, normalized)   embeddings.py
      ▼
  Local vector store: vectors.npy + chunks.json       vector_store.py
      (cosine similarity == dot product on normalized vectors)

QUERY (per question)
  question ──► LangGraph agent loop ──► answer + citations + trace
```

## The agent loop (agent.py)

A LangGraph `StateGraph` with two nodes:

```
START ─► agent ─► route ─┬─► tools ─► agent (repeat)
                         └─► END
```

- **State**: message history (via `add_messages`), `steps`, `tokens_used`,
  `started_at`, `stop_reason`.
- **agent node**: checks the three session budgets *before* invoking the
  model. Within budget → invoke Claude (Bedrock Converse API via
  `ChatBedrockConverse`) with the `search_documents` tool bound. Budget
  exceeded → invoke the model *without* tools plus an instruction to answer
  from the passages already retrieved, and set `stop_reason`.
- **tools node**: executes each tool call, logging call/result to the
  trace. Exceptions are caught and returned as a `ToolMessage` with
  `status="error"` and retry guidance — the model sees the failure and can
  correct itself (bad query, unknown tool, transient index error) instead
  of the run crashing.
- **route**: `tools` if the last AI message contains tool calls and no
  budget was hit; otherwise `END`. A forced final answer always ends the
  session even if the model tries to call a tool.

### Grounding

The system prompt requires: search before answering, answer only from
retrieved passages, cite `[source: <doc>#<chunk>]` for every claim, and
say so when the documents don't cover the question. The tool output labels
every passage with its source id so citation is mechanical, not recalled.

### Budgets (cost + latency control)

| Budget | Default | Enforced where |
|---|---|---|
| `max_steps` | 6 round trips | before each model call |
| `max_session_tokens` | 20,000 (in+out) | accumulated from `usage_metadata` |
| `max_session_seconds` | 60s wall clock | monotonic timer per session |
| `max_response_tokens` | 1,024 | `max_tokens` on every model call |

Worst-case per-question cost is therefore bounded at roughly
20K tokens ≈ $0.06 on Haiku 4.5 — the loop cannot run away.

## Tracing (trace.py)

One JSONL file per session (`traces/session-<id>.jsonl`). Events:
`session_start`, `llm_call`, `llm_response` (with token counts and tool
call names), `tool_call` (with args), `tool_result`, `tool_error`,
`limit_hit`, `session_end`. A wrong answer is debugged by replaying the
file: what was searched, what came back, what the model did with it.

## Why these choices

| Decision | Alternative | Why this one |
|---|---|---|
| Local numpy index | OpenSearch Serverless / Kendra | $0 vs ~$170+/month idle; corpus is demo-scale; keeps the $5 budget |
| Claude Haiku 4.5 | Sonnet/Opus tiers | 3–25× cheaper; grounded extraction over retrieved text doesn't need a frontier model; configurable via `DOCQA_MODEL_ID` |
| Titan Embeddings V2 @ 512 dims | 1024 dims / Cohere embed | half the storage, negligible quality loss at this scale |
| Custom tools node | `langgraph.prebuilt.ToolNode` | needed per-call trace logging and error-to-ToolMessage conversion with retry guidance |
| Char-based chunking | token-based | avoids a tokenizer dependency; ~1200 chars ≈ 300 tokens, well inside embedding limits |

## Failure modes and handling

- **Tool raises** → error ToolMessage, model retries or answers with a caveat.
- **Model loops on searches** → `max_steps` forces a final answer.
- **Runaway token spend** → `max_session_tokens` forces a final answer.
- **Slow Bedrock responses** → `max_session_seconds` forces a final answer.
- **Answer not in corpus** → prompt instructs an explicit "not covered" reply;
  eval case e20-style questions verify citation behavior on edge topics.
