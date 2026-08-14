# Project Structure

```
doc-question-answer-agent/
├── PLAN.md                     # Build plan and milestones
├── PROJECT_STRUCTURE.md        # This file
├── claude.md                   # Guidance for AI coding assistants
├── readme.md                   # Overview, quick start, cost model
├── tasks.md                    # Task tracker
├── requirements.txt            # Python dependencies
├── docs/
│   ├── API_SPEC.md             # CLI + Python API contracts
│   ├── ARCHITECTURE.md         # System design: agent loop, retrieval, tracing
│   ├── REQUIREMENTS.md         # Functional + non-functional requirements
│   └── TEST_STRATEGY.md        # Unit tests + eval harness strategy
├── data/
│   └── sample_docs/            # Sample corpus (fictional Aurora platform docs)
│       ├── aurora_platform.md
│       ├── billing_faq.md
│       └── security_policy.md
├── src/
│   └── doc_qa_agent/
│       ├── __init__.py
│       ├── config.py           # Settings: models, limits, retrieval params
│       ├── chunking.py         # Paragraph-aware chunking with overlap
│       ├── embeddings.py       # Bedrock Titan V2 embedder
│       ├── vector_store.py     # Local numpy vector store (cosine top-k)
│       ├── tools.py            # search_documents tool
│       ├── agent.py            # LangGraph tool-calling loop + limits
│       ├── trace.py            # Per-session JSONL trace logger
│       └── cli.py              # ingest / ask commands
├── evals/
│   ├── eval_set.json           # 20 cases: question, sources, keywords
│   └── run_evals.py            # Scores retrieval relevance + grounding
├── infra/                      # Hosted demo (CloudFront + S3 + Lambda)
│   ├── deploy.py               # boto3 deployment script (rerunnable)
│   ├── lambda_handler.py       # Dependency-free port of the agent loop
│   └── frontend/
│       └── index.html          # Static demo page
└── tests/
    ├── conftest.py             # Fake tool-calling LLM
    ├── test_chunking.py
    ├── test_vector_store.py
    ├── test_agent.py           # Loop, limits, error feedback
    └── test_trace.py
```

Generated at runtime (git-ignored): `index/` (vectors + chunk metadata),
`traces/` (session JSONL logs), `evals/results.json`.
