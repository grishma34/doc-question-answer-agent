"""Command-line interface.

    python -m doc_qa_agent.cli ingest --docs data/sample_docs --index index
    python -m doc_qa_agent.cli ask "What payment methods are accepted?" --index index
"""

import argparse
import json

from .chunking import chunk_directory
from .config import Settings
from .embeddings import BedrockEmbedder
from .trace import TraceLogger
from .vector_store import VectorStore


def cmd_ingest(args, settings: Settings) -> None:
    chunks = chunk_directory(args.docs, settings.chunk_size, settings.chunk_overlap)
    embedder = _embedder(settings)
    print(f"Embedding {len(chunks)} chunks with {settings.embedding_model_id}...")
    vectors = embedder.embed_batch([c.text for c in chunks])
    VectorStore(vectors, chunks).save(args.index)
    print(f"Index written to {args.index}/ ({len(chunks)} chunks)")


def cmd_ask(args, settings: Settings) -> None:
    from langchain_aws import ChatBedrockConverse

    from .agent import DocQAAgent
    from .tools import make_search_tool

    store = VectorStore.load(args.index)
    embedder = _embedder(settings)
    search_tool = make_search_tool(store, embedder, top_k=settings.top_k)
    llm = ChatBedrockConverse(
        model_id=settings.model_id,
        region_name=settings.region,
        max_tokens=settings.max_response_tokens,
    )
    trace = TraceLogger(settings.trace_dir)
    agent = DocQAAgent(llm, [search_tool], settings, trace)
    result = agent.ask(args.question)

    print("\n=== Answer ===")
    print(result["answer"])
    print("\n=== Session ===")
    print(json.dumps({k: v for k, v in result.items() if k != "answer"}, indent=2))
    print(f"Trace: {trace.path}")


def _embedder(settings: Settings) -> BedrockEmbedder:
    return BedrockEmbedder(
        settings.embedding_model_id, settings.region, settings.embedding_dimensions
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="doc-qa-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Chunk, embed, and index documents")
    p_ingest.add_argument("--docs", required=True, help="Directory of .md/.txt files")
    p_ingest.add_argument("--index", default="index", help="Output index directory")

    p_ask = sub.add_parser("ask", help="Ask a question against an index")
    p_ask.add_argument("question")
    p_ask.add_argument("--index", default="index", help="Index directory")

    args = parser.parse_args()
    settings = Settings()
    if args.command == "ingest":
        cmd_ingest(args, settings)
    elif args.command == "ask":
        cmd_ask(args, settings)


if __name__ == "__main__":
    main()
