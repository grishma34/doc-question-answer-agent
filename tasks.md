# Tasks

## Done

- [x] Repo scaffolding and required doc structure
- [x] Paragraph-aware chunker with overlap and stable chunk ids
- [x] Bedrock Titan V2 embedder
- [x] Local numpy vector store (cosine top-k, save/load)
- [x] `search_documents` tool with `[source: ...]` labeled passages
- [x] LangGraph agent loop (agent node + tools node + conditional routing)
- [x] Session budgets: max steps, max tokens, max wall-clock seconds
- [x] Forced final answer when a budget is hit (tool access removed)
- [x] Tool errors returned to the model as error ToolMessages
- [x] JSONL per-session trace logger (model calls, tool calls, errors, limits)
- [x] CLI: `ingest` and `ask`
- [x] 20-case eval set over the sample corpus
- [x] Eval runner scoring retrieval relevance and grounding (+ retrieval-only mode)
- [x] Unit tests with fake LLM (no AWS needed): loop, limits, errors, chunking, store, trace
- [x] readme, PLAN, PROJECT_STRUCTURE, claude.md, docs/*
- [x] Public GitHub repo

## Backlog

- [ ] GitHub Actions CI (pytest on push)
- [ ] LLM-as-judge grounding metric (optional flag on run_evals.py)
- [ ] Hybrid BM25 + vector retrieval
- [ ] PDF ingestion via pypdf
- [ ] Streamed answers in the CLI
