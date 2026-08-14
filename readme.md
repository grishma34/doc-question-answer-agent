# Document Question and Answer Agent

A tool-calling agent that answers questions about your documents with cited
sources. Built with **Python, LangGraph, and Amazon Bedrock** (Claude Haiku
4.5 for generation, Titan Text Embeddings V2 for retrieval), designed to run
for **under $5/month** on AWS.

## What it does

- **Grounded answers, not invented ones** — a LangGraph agent loop sits on
  top of a retrieval pipeline that chunks documents, embeds them with
  Bedrock, and runs top-k cosine-similarity search. The agent must search
  before answering and cites every claim with a `[source: doc.md#chunk]` id.
- **Capped cost and latency** — every session enforces limits on agent
  steps (default 6), total tokens (20k), and wall-clock time (60s). When a
  budget is hit, the agent is forced to answer from what it has gathered.
- **Traceable** — every model call, tool call, tool result, error, and
  limit event is logged to a per-session JSONL file in `traces/`, so a
  wrong answer can be replayed step by step.
- **Self-correcting on tool errors** — a tool exception is returned to the
  model as an error message instead of crashing the run, letting the agent
  retry with corrected input.
- **Measurable** — a 20-case evaluation set scores retrieval relevance and
  answer grounding, turning any prompt/chunking change into a regression
  test instead of a guess.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# AWS credentials with Bedrock access (and model access enabled for
# Claude Haiku 4.5 + Titan Embeddings V2 in the Bedrock console)
export AWS_REGION=us-east-1

# 1. Build the index from the sample docs (~$0.0001)
PYTHONPATH=src python -m doc_qa_agent.cli ingest --docs data/sample_docs --index index

# 2. Ask a question
PYTHONPATH=src python -m doc_qa_agent.cli ask "What payment methods are accepted?" --index index
```

Example output:

```
=== Answer ===
External partner teams can pay by corporate credit card or by invoice with
net-30 terms; internal teams are charged via cost-center chargeback
[source: billing_faq.md#1].

=== Session ===
{ "steps": 2, "tokens_used": 3421, "stop_reason": "completed", ... }
Trace: traces/session-ab12cd34ef56.jsonl
```

## Running the evals

```bash
# Retrieval relevance only — embeddings cost, effectively free
PYTHONPATH=src python evals/run_evals.py --retrieval-only

# Full agent run — scores retrieval relevance AND grounding (~$0.02-0.05)
PYTHONPATH=src python evals/run_evals.py
```

## Running the tests (no AWS required)

```bash
pytest
```

The agent loop, limits, error feedback, chunking, vector store, and tracing
are all tested against a fake LLM — no Bedrock calls, no credentials needed.

## Cost model (the $5/month budget)

There is no provisioned infrastructure: no OpenSearch, no Kendra, no EC2,
no SageMaker. The vector index is a local numpy file; the **only** AWS
charges are on-demand Bedrock invocations.

| Item | Price | Typical monthly use | Cost |
|---|---|---|---|
| Claude Haiku 4.5 (Bedrock) | $1 / $5 per 1M in/out tokens | 500 questions × ~3K in + 300 out | ~$1.90 |
| Titan Embeddings V2 | $0.02 per 1M tokens | re-index + 500 query embeddings | < $0.01 |
| Full eval run (20 cases) | — | 4 runs | ~$0.20 |
| **Total** | | | **~$2.10** |

The per-session token cap (20k) bounds worst-case cost per question to
about $0.06 even if the agent loops to its step limit.

## Project structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md). Design docs live in
[docs/](docs/): [architecture](docs/ARCHITECTURE.md),
[requirements](docs/REQUIREMENTS.md), [API spec](docs/API_SPEC.md),
[test strategy](docs/TEST_STRATEGY.md).
