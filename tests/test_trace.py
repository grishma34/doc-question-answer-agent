import json

from doc_qa_agent.trace import TraceLogger


def test_trace_writes_jsonl(tmp_path):
    trace = TraceLogger(tmp_path)
    trace.log("tool_call", name="search_documents", args={"query": "hi"})
    trace.log("tool_result", name="search_documents", chars=42)

    lines = trace.path.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event"] == "tool_call"
    assert first["session_id"] == trace.session_id
    assert first["args"] == {"query": "hi"}
    assert "ts" in first
