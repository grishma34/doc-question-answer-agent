"""Hosted-demo backend: a slim, dependency-free port of the QA agent.

The canonical implementation is the LangGraph agent in src/doc_qa_agent/.
This Lambda re-implements the same loop with only boto3 + stdlib so the
deployment zip needs no third-party packages:

  - the chunk index (vectors + text) is bundled into the zip as index.json
  - query embedding via Bedrock Titan V2, cosine top-k in pure Python
  - tool loop via the Bedrock Converse API with the same budgets
    (max steps, max tokens) and the same error-feedback behavior
  - a DynamoDB daily request counter hard-caps spend from the public URL

GET /api/ask?q=<question>  ->  {"answer", "sources", "steps", "tokens_used",
                                "stop_reason"}
"""

import datetime
import json
import math
import os
from pathlib import Path

import boto3

REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
MODEL_ID = os.environ.get("MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
USAGE_TABLE = os.environ.get("USAGE_TABLE", "")
DAILY_REQUEST_LIMIT = int(os.environ.get("DAILY_REQUEST_LIMIT", "40"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "4"))
MAX_SESSION_TOKENS = int(os.environ.get("MAX_SESSION_TOKENS", "8000"))
MAX_RESPONSE_TOKENS = int(os.environ.get("MAX_RESPONSE_TOKENS", "400"))
TOP_K = 4

_rt = boto3.client("bedrock-runtime", region_name=REGION)
_ddb = boto3.client("dynamodb", region_name=REGION) if USAGE_TABLE else None

_INDEX = json.loads((Path(__file__).parent / "index.json").read_text())

SYSTEM_PROMPT = (
    "You are a document question-answering assistant for a small sample "
    "corpus about the fictional Aurora platform. Always call "
    "search_documents before answering; never answer from prior knowledge. "
    "Answer ONLY from retrieved passages and cite every claim with the "
    "source ids shown, e.g. [source: billing_faq.md#2]. If the passages "
    "don't contain the answer, say the documents don't cover it. Be concise."
)

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "search_documents",
                "description": (
                    "Search the indexed documents and return the most "
                    "relevant passages, each labeled with a [source: ...] id."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"],
                    }
                },
            }
        }
    ]
}


def _embed(text: str) -> list[float]:
    body = json.dumps({"inputText": text, "dimensions": 512, "normalize": True})
    resp = _rt.invoke_model(
        modelId=EMBED_MODEL_ID, body=body,
        contentType="application/json", accept="application/json",
    )
    return json.loads(resp["body"].read())["embedding"]


def _search(query: str) -> tuple[str, list[str]]:
    query = (query or "").strip()
    if not query:
        raise ValueError("query must be a non-empty string")
    q = _embed(query)
    qn = math.sqrt(sum(x * x for x in q)) or 1.0
    scored = []
    for entry in _INDEX:
        v = entry["vector"]  # stored normalized
        score = sum(a * b for a, b in zip(q, v)) / qn
        scored.append((score, entry))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:TOP_K]
    blocks = [
        f"[source: {e['chunk_id']}] (score={s:.3f})\n{e['text']}" for s, e in top
    ]
    return "\n\n---\n\n".join(blocks), [e["chunk_id"] for _, e in top]


def _check_daily_limit() -> bool:
    if _ddb is None:
        return True
    today = datetime.date.today().isoformat()
    resp = _ddb.update_item(
        TableName=USAGE_TABLE,
        Key={"pk": {"S": f"day#{today}"}},
        UpdateExpression="ADD #c :one",
        ExpressionAttributeNames={"#c": "count"},
        ExpressionAttributeValues={":one": {"N": "1"}},
        ReturnValues="UPDATED_NEW",
    )
    return int(resp["Attributes"]["count"]["N"]) <= DAILY_REQUEST_LIMIT


def _answer(question: str) -> dict:
    messages = [{"role": "user", "content": [{"text": question}]}]
    tokens_used = 0
    sources: list[str] = []
    stop_reason = "completed"

    for step in range(MAX_STEPS + 1):
        over_budget = step >= MAX_STEPS or tokens_used >= MAX_SESSION_TOKENS
        if over_budget:
            stop_reason = "budget_exceeded"
            messages.append({
                "role": "user",
                "content": [{"text": (
                    "Session budget reached. Do not use tools again. Answer "
                    "now from the passages already retrieved, with citations."
                )}],
            })
        resp = _rt.converse(
            modelId=MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig=TOOL_CONFIG,
            inferenceConfig={"maxTokens": MAX_RESPONSE_TOKENS},
        )
        tokens_used += resp["usage"]["totalTokens"]
        msg = resp["output"]["message"]
        messages.append(msg)

        if resp["stopReason"] != "tool_use" or over_budget:
            answer = "".join(b.get("text", "") for b in msg["content"])
            return {
                "answer": answer,
                "sources": sorted(set(sources)),
                "steps": step + 1,
                "tokens_used": tokens_used,
                "stop_reason": stop_reason,
            }

        results = []
        for block in msg["content"]:
            if "toolUse" not in block:
                continue
            tu = block["toolUse"]
            try:
                output, ids = _search(tu["input"].get("query", ""))
                sources.extend(ids)
                results.append({
                    "toolResult": {
                        "toolUseId": tu["toolUseId"],
                        "content": [{"text": output}],
                    }
                })
            except Exception as exc:  # feed the error back, don't crash
                results.append({
                    "toolResult": {
                        "toolUseId": tu["toolUseId"],
                        "content": [{"text": f"Tool error ({type(exc).__name__}): {exc}. "
                                             "Adjust your input and try again."}],
                        "status": "error",
                    }
                })
        messages.append({"role": "user", "content": results})

    return {"answer": "", "sources": [], "steps": MAX_STEPS,
            "tokens_used": tokens_used, "stop_reason": "budget_exceeded"}


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json",
                    "cache-control": "no-store"},
        "body": json.dumps(body, ensure_ascii=False),
    }


def handler(event, context):
    # Only CloudFront knows this secret (injected as an origin custom
    # header), so direct calls to the function URL are rejected.
    secret = os.environ.get("ORIGIN_SECRET", "")
    if secret:
        sent = (event.get("headers") or {}).get("x-origin-verify", "")
        if sent != secret:
            return _response(403, {"error": "Forbidden"})

    params = event.get("queryStringParameters") or {}
    question = (params.get("q") or "").strip()
    if not question:
        return _response(400, {"error": "Missing query parameter: q"})
    if len(question) > 500:
        return _response(400, {"error": "Question too long (max 500 chars)"})

    try:
        if not _check_daily_limit():
            return _response(429, {"error": (
                "The demo's daily question limit has been reached. "
                "Try again tomorrow, or run the agent yourself from the repo."
            )})
        result = _answer(question)
        return _response(200, result)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return _response(500, {"error": "Something went wrong answering that question."})
