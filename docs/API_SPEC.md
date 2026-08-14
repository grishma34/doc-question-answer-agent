# API Specification

The project exposes two surfaces: a CLI and a Python API. There is no HTTP
server (out of scope for v0.1 and unnecessary for the cost budget).

## CLI

Run from the repo root with `PYTHONPATH=src`.

### `python -m doc_qa_agent.cli ingest`

Chunk, embed, and index a directory of documents.

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--docs` | yes | — | Directory containing `.md`/`.txt` files |
| `--index` | no | `index` | Output directory for the index |

Output: `index/vectors.npy` (float32 matrix, L2-normalized) and
`index/chunks.json` (list of `{chunk_id, doc_name, text}`).
Exit non-zero if the docs directory contains no ingestible files.

### `python -m doc_qa_agent.cli ask`

Answer one question against an existing index.

| Arg/Flag | Required | Default | Meaning |
|---|---|---|---|
| `question` (positional) | yes | — | The question |
| `--index` | no | `index` | Index directory |

Prints the cited answer, then a session summary JSON:

```json
{
  "steps": 2,
  "tokens_used": 3421,
  "stop_reason": "completed",
  "session_id": "ab12cd34ef56"
}
```

`stop_reason` is `"completed"` or names the budget hit, e.g.
`"max_steps (6)"`. The trace file path is printed last.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `DOCQA_MODEL_ID` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock chat model |
| `DOCQA_EMBED_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Bedrock embedding model |
| `AWS_REGION` | `us-east-1` | Bedrock region |

AWS credentials come from the standard boto3 chain (env vars, profile,
SSO, instance role).

## Python API

```python
from doc_qa_agent.config import Settings
from doc_qa_agent.embeddings import BedrockEmbedder
from doc_qa_agent.vector_store import VectorStore
from doc_qa_agent.tools import make_search_tool
from doc_qa_agent.agent import DocQAAgent
from doc_qa_agent.trace import TraceLogger
from langchain_aws import ChatBedrockConverse

settings = Settings()
store = VectorStore.load("index")
embedder = BedrockEmbedder(settings.embedding_model_id, settings.region,
                           settings.embedding_dimensions)
tool = make_search_tool(store, embedder, top_k=settings.top_k)
llm = ChatBedrockConverse(model_id=settings.model_id,
                          region_name=settings.region,
                          max_tokens=settings.max_response_tokens)
agent = DocQAAgent(llm, [tool], settings, TraceLogger("traces"))

result = agent.ask("How do refunds work?")
# result: {"answer": str, "steps": int, "tokens_used": int,
#          "stop_reason": str, "session_id": str}
```

`DocQAAgent(llm, tools, settings, trace)` accepts any LangChain chat model
implementing `bind_tools`/`invoke` — tests inject a fake; production uses
`ChatBedrockConverse`.

## Trace file format

`traces/session-<id>.jsonl`, one JSON object per line:

```json
{"ts": 1755200000.123, "session_id": "ab12cd34ef56", "event": "tool_call",
 "name": "search_documents", "args": {"query": "refund policy"}, "id": "toolu_..."}
```

Event types: `session_start`, `llm_call`, `llm_response`, `tool_call`,
`tool_result`, `tool_error`, `limit_hit`, `session_end`.

## Eval report format

`evals/results.json`:

```json
{
  "summary": {"cases": 20, "retrieval_relevance": 0.95, "grounding": 0.90,
              "total_tokens": 61234},
  "rows": [{"id": "e01", "question": "...", "retrieval_relevant": true,
            "retrieved": ["aurora_platform.md#1", "..."],
            "answer": "...", "cites_expected_source": true,
            "keywords_found": ["rollback"], "grounded": true}]
}
```

In `--retrieval-only` mode the `grounding`, `answer`, and token fields are
omitted.
