# Test Strategy

Two layers: fast deterministic unit tests (no AWS), and a paid-but-cheap
evaluation harness that measures end-to-end answer quality.

## Layer 1 — Unit tests (`pytest`, zero AWS cost)

Design principle: every Bedrock dependency is injectable, so the agent
loop is tested against a scripted fake LLM (`tests/conftest.py::
FakeToolCallingLLM`) and the vector store against hand-built vectors.

| Test file | What it proves |
|---|---|
| `test_chunking.py` | Size limits respected, stable ids, long-paragraph hard split, invalid params rejected, sample corpus ingests |
| `test_vector_store.py` | Top-k ordering by cosine similarity, k capped at store size, zero-vector rejection, save/load roundtrip preserves results |
| `test_agent.py` | Happy path (tool call → cited answer); tool exception becomes an error ToolMessage the model sees (run does not crash); unknown tool name handled; each of the three budgets (steps, tokens, time) forces a final answer and sets `stop_reason` |
| `test_trace.py` | JSONL records carry timestamp, session id, and event payload |

The budget tests are the contract for resume bullet #2; the error-feedback
tests are the contract for bullet #3. CI candidate: run `pytest` on every
push (needs no secrets).

## Layer 2 — Evaluation harness (`evals/run_evals.py`)

A fixed 20-case set (`eval_set.json`) over the sample corpus. Each case
declares the question, the document(s) containing the answer, and keywords
a correct grounded answer must include.

**Metrics**

- `retrieval_relevance` — fraction of cases where top-k search returned at
  least one chunk from an expected source document. Isolates the retrieval
  pipeline; runnable with `--retrieval-only` for near-zero cost.
- `grounding` — fraction of cases where the agent's final answer both
  cites a `[source: ...]` id from an expected document and contains at
  least one expected keyword. A full run costs roughly $0.02–0.05.

**Why heuristic scoring:** deterministic and free, so the same command run
before and after a change yields directly comparable numbers — a prompt,
chunk-size, or top-k change becomes a measurable regression instead of a
guess. Known limitation: keyword matching can't detect a fluent answer
that cites the right source but paraphrases incorrectly; an optional
LLM-as-judge pass is future work.

**Regression workflow**

1. `python evals/run_evals.py --retrieval-only` → baseline `results.json`
2. Make the change (prompt, chunking, top_k, model).
3. Re-run (full run if the change affects answering). Diff summaries; any
   metric drop blocks the change.

## What is deliberately not tested

- Live Bedrock integration in CI (needs credentials and spends money);
  covered instead by running `ingest` + one `ask` manually after changes
  to `embeddings.py` or the `ChatBedrockConverse` wiring.
- Concurrency/load — single-user CLI tool.
