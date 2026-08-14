# Build Plan

## Goal

A portfolio-quality document QA agent demonstrating four things:

1. A tool-calling agent loop (LangGraph) over a retrieval pipeline
   (chunk → embed → top-k vector search) with cited, grounded answers.
2. Hard per-session caps on steps, tokens, and time, plus full tool-call
   tracing for debuggability.
3. Tool errors fed back to the model for self-correction instead of
   crashing the run.
4. A 20-case eval set scoring retrieval relevance and grounding, usable as
   a regression test for prompt/retrieval changes.

Hard constraint: **≤ $5/month AWS cost**, achieved by using only on-demand
Bedrock calls (Claude Haiku 4.5 + Titan Embeddings V2) and a local numpy
vector index — no provisioned infrastructure.

## Milestones

| # | Milestone | Deliverables | Status |
|---|---|---|---|
| 1 | Retrieval pipeline | `chunking.py`, `embeddings.py`, `vector_store.py`, `ingest` CLI | ✅ Done |
| 2 | Agent loop | `agent.py` (LangGraph StateGraph), `tools.py`, system prompt with citation rules | ✅ Done |
| 3 | Guardrails | Step/token/time budgets, forced final answer, error-as-ToolMessage feedback | ✅ Done |
| 4 | Observability | `trace.py` JSONL session traces for every model/tool event | ✅ Done |
| 5 | Evaluation | `evals/eval_set.json` (20 cases), `evals/run_evals.py` (retrieval + grounding scores) | ✅ Done |
| 6 | Tests | Fake-LLM unit tests for loop, limits, errors; chunking/store/trace tests | ✅ Done |
| 7 | Docs + publish | readme, architecture/requirements/API/test docs, public GitHub repo | ✅ Done |
| 8 | Hosted demo | CloudFront + S3 frontend, API Gateway + Lambda backend with daily cap and budget alert (`infra/`) | ✅ Done |

## Key design decisions

- **Local vector store over a hosted one.** OpenSearch Serverless starts at
  ~$170/month idle; a numpy matrix on disk costs $0 and is plenty for a
  demo-scale corpus. Trade-off recorded in docs/ARCHITECTURE.md.
- **Claude Haiku 4.5 as default model.** Cheapest Claude tier on Bedrock
  ($1/$5 per MTok); strong enough for grounded extraction over retrieved
  passages. Model is configurable via `DOCQA_MODEL_ID`.
- **Custom tools node instead of LangGraph's prebuilt ToolNode.** Needed to
  log every call/result/error to the trace and to convert exceptions into
  error ToolMessages with retry guidance.
- **Deterministic, heuristic eval scoring** (source match + keyword match)
  rather than LLM-as-judge — free, reproducible, and sufficient to catch
  regressions; an LLM judge is listed as future work.

## Future work

- LLM-as-judge grounding score alongside the heuristic one.
- Hybrid retrieval (BM25 + vector) and a reranking step.
- PDF/HTML ingestion.
- GitHub Actions CI running `pytest` on every push.
