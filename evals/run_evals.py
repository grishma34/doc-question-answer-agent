"""Evaluation harness: retrieval relevance + answer grounding.

Two scored dimensions per case (resume bullet #4):

* retrieval relevance — did the top-k search results include a chunk from
  the expected source document?
* grounding — did the agent's final answer (a) cite a [source: ...] id from
  the expected document and (b) contain the expected answer keywords?

Run modes:

    python evals/run_evals.py --retrieval-only   # embeddings only, ~free
    python evals/run_evals.py                    # full agent, ~$0.02-0.05

Because scores are deterministic given the index and eval set, a prompt or
chunking change becomes a measurable regression instead of a guess:
run before, run after, diff the report.
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from doc_qa_agent.config import Settings  # noqa: E402
from doc_qa_agent.embeddings import BedrockEmbedder  # noqa: E402
from doc_qa_agent.trace import TraceLogger  # noqa: E402
from doc_qa_agent.vector_store import VectorStore  # noqa: E402


def score_retrieval(results, expected_sources: list[str]) -> bool:
    retrieved_docs = {r.chunk.doc_name for r in results}
    return any(src in retrieved_docs for src in expected_sources)


def score_grounding(answer: str, expected_sources: list[str], keywords: list[str]) -> dict:
    cited = re.findall(r"\[source:\s*([^\]#]+)#\d+\]", answer)
    cites_expected = any(src.strip() in expected_sources for src in cited)
    answer_lower = answer.lower()
    keywords_found = [k for k in keywords if k.lower() in answer_lower]
    return {
        "cites_expected_source": cites_expected,
        "keywords_found": keywords_found,
        "keywords_expected": keywords,
        "grounded": cites_expected and len(keywords_found) > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="index")
    parser.add_argument("--eval-set", default=str(Path(__file__).parent / "eval_set.json"))
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Skip the agent; score retrieval only (near-zero cost)")
    parser.add_argument("--out", default=str(Path(__file__).parent / "results.json"))
    args = parser.parse_args()

    settings = Settings()
    cases = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))["cases"]
    store = VectorStore.load(args.index)
    embedder = BedrockEmbedder(
        settings.embedding_model_id, settings.region, settings.embedding_dimensions
    )

    agent = None
    if not args.retrieval_only:
        from langchain_aws import ChatBedrockConverse

        from doc_qa_agent.agent import DocQAAgent
        from doc_qa_agent.tools import make_search_tool

        llm = ChatBedrockConverse(
            model_id=settings.model_id,
            region_name=settings.region,
            max_tokens=settings.max_response_tokens,
        )
        search_tool = make_search_tool(store, embedder, top_k=settings.top_k)
        agent = DocQAAgent(llm, [search_tool], settings, TraceLogger(settings.trace_dir))

    rows = []
    for case in cases:
        results = store.search(embedder.embed(case["question"]), top_k=settings.top_k)
        row = {
            "id": case["id"],
            "question": case["question"],
            "retrieval_relevant": score_retrieval(results, case["expected_sources"]),
            "retrieved": [r.chunk.chunk_id for r in results],
        }
        if agent is not None:
            outcome = agent.ask(case["question"])
            row["answer"] = outcome["answer"]
            row["tokens_used"] = outcome["tokens_used"]
            row.update(score_grounding(
                outcome["answer"], case["expected_sources"], case["expected_keywords"]
            ))
        rows.append(row)
        status = "PASS" if row["retrieval_relevant"] else "FAIL"
        print(f"[{case['id']}] retrieval={status}"
              + (f" grounded={'PASS' if row.get('grounded') else 'FAIL'}" if agent else ""))

    n = len(rows)
    summary = {
        "cases": n,
        "retrieval_relevance": sum(r["retrieval_relevant"] for r in rows) / n,
    }
    if agent is not None:
        summary["grounding"] = sum(bool(r.get("grounded")) for r in rows) / n
        summary["total_tokens"] = sum(r.get("tokens_used", 0) for r in rows)

    report = {"summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
