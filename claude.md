# Guidance for AI assistants working in this repo

## What this is

A document QA agent: LangGraph tool-calling loop over a local RAG pipeline,
with all model calls going through Amazon Bedrock (Claude Haiku 4.5 +
Titan Embeddings V2). Hard constraint: total AWS cost must stay under
$5/month, so no provisioned infrastructure — the vector index is a local
numpy file and the only paid calls are Bedrock invocations.

## Layout

- `src/doc_qa_agent/` — library code. `agent.py` is the core: a StateGraph
  with an `agent` node and a `tools` node, budget checks before each model
  call, and error-as-ToolMessage feedback.
- `evals/` — the 20-case regression eval (retrieval relevance + grounding).
- `tests/` — unit tests with a fake LLM; must pass with **no AWS access**.

## Conventions

- Run tests with `pytest` from the repo root. `tests/conftest.py` puts
  `src/` on the path.
- Run CLI/evals with `PYTHONPATH=src`.
- Never add a dependency on a hosted vector DB or any always-on AWS
  resource — that breaks the cost budget.
- Chunk ids are `<doc_name>#<n>` and are the citation format
  (`[source: doc.md#3]`). The eval grounding scorer parses that exact
  format — if you change it, update `evals/run_evals.py::score_grounding`
  and the system prompt together.
- Any change to prompts, chunking, or retrieval params must be validated by
  running `evals/run_evals.py --retrieval-only` (free) and, if it touches
  answering behavior, the full eval (~$0.05).
- Session budgets (`max_steps`, `max_session_tokens`,
  `max_session_seconds`) live in `config.py`; enforcement lives in
  `DocQAAgent._limit_hit`. Both `tests/test_agent.py` limit tests must keep
  passing.
