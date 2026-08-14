# Requirements

## Functional

| ID | Requirement | Implementation |
|---|---|---|
| F1 | Ingest a directory of .md/.txt documents into a searchable index | `cli.py ingest`, `chunking.py`, `embeddings.py`, `vector_store.py` |
| F2 | Chunk documents at paragraph boundaries with overlap and stable ids | `chunking.chunk_text` |
| F3 | Embed chunks and queries with Amazon Bedrock (Titan V2) | `embeddings.BedrockEmbedder` |
| F4 | Retrieve top-k chunks by cosine similarity | `vector_store.VectorStore.search` |
| F5 | Answer questions via a tool-calling agent that must search before answering | `agent.DocQAAgent`, `tools.make_search_tool` |
| F6 | Every claim in an answer cites a `[source: doc#chunk]` id from retrieved text | system prompt + labeled tool output |
| F7 | State clearly when the corpus does not contain the answer | system prompt |
| F8 | Enforce per-session caps on steps, tokens, and wall-clock time; on breach, force a final answer from gathered context | `DocQAAgent._limit_hit`, `_agent_node` |
| F9 | Log every model call, tool call, result, error, and limit event per session | `trace.TraceLogger` (JSONL) |
| F10 | Return tool errors to the model as error messages (never crash the session) | `DocQAAgent._tools_node` |
| F11 | Score the system on a fixed 20-case eval set: retrieval relevance + grounding | `evals/eval_set.json`, `evals/run_evals.py` |
| F12 | Support a retrieval-only eval mode with near-zero cost | `run_evals.py --retrieval-only` |

## Non-functional

| ID | Requirement | How it's met |
|---|---|---|
| N1 | Total AWS cost ≤ $5/month under typical demo usage | on-demand Bedrock only; local index; Haiku-tier model; per-session token cap bounds worst case (~$0.06/question); cost table in readme |
| N2 | No provisioned/always-on AWS resources | numpy index on local disk; nothing to deprovision |
| N3 | Unit tests runnable with no AWS credentials | fake LLM in `tests/conftest.py`; embedder is injectable |
| N4 | Deterministic, reproducible eval scoring | heuristic scorers (source match, keyword match), fixed eval set |
| N5 | Model and region configurable without code changes | `DOCQA_MODEL_ID`, `DOCQA_EMBED_MODEL_ID`, `AWS_REGION` env vars |
| N6 | A single question completes in ≤ 60s | `max_session_seconds` cap |
| N7 | Secrets never stored in the repo | standard AWS credential chain (env/profile); `.gitignore` blocks `.env` |

## Out of scope (v0.1)

- PDF/HTML ingestion, multi-user serving, conversation memory across
  sessions, hosted vector databases, LLM-as-judge evals (see PLAN.md
  future work).
