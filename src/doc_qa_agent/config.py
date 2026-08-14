"""Central configuration for the agent, retrieval pipeline, and cost caps."""

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # --- Bedrock models ---
    # Claude Haiku 4.5 on Bedrock (cross-region inference profile).
    # Cheapest Claude tier: $1 / $5 per million input/output tokens.
    model_id: str = os.environ.get(
        "DOCQA_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )
    # Titan Text Embeddings V2: $0.00002 per 1K input tokens.
    embedding_model_id: str = os.environ.get(
        "DOCQA_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"
    )
    region: str = os.environ.get("AWS_REGION", "us-east-1")

    # --- Retrieval ---
    chunk_size: int = 1200      # characters per chunk
    chunk_overlap: int = 200    # characters of overlap between chunks
    top_k: int = 4              # chunks returned per search
    embedding_dimensions: int = 512  # Titan V2 supports 256/512/1024

    # --- Per-session cost / latency caps (resume bullet #2) ---
    max_steps: int = 6                 # max agent<->tool round trips
    max_session_tokens: int = 20_000   # input+output token budget per session
    max_session_seconds: float = 60.0  # wall-clock budget per session
    max_response_tokens: int = 1024    # max_tokens per single model call

    # --- Paths ---
    index_dir: str = "index"
    trace_dir: str = "traces"

    extra: dict = field(default_factory=dict)


DEFAULT_SETTINGS = Settings()
