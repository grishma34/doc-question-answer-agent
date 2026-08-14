from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from conftest import FakeToolCallingLLM, ai_answer, ai_tool_call
from doc_qa_agent.agent import DocQAAgent
from doc_qa_agent.config import Settings
from doc_qa_agent.trace import NullTraceLogger


@tool
def search_documents(query: str) -> str:
    """Search the documents (test stub)."""
    if query == "boom":
        raise RuntimeError("index unavailable")
    return f"[source: doc.md#0] result for {query}"


def make_agent(responses, **setting_overrides):
    settings = Settings(**setting_overrides) if setting_overrides else Settings()
    llm = FakeToolCallingLLM(responses)
    trace = NullTraceLogger()
    return DocQAAgent(llm, [search_documents], settings, trace), llm, trace


def test_happy_path_tool_then_answer():
    agent, _, trace = make_agent([
        ai_tool_call("search_documents", {"query": "refunds"}),
        ai_answer("Refunds are platform credit [source: doc.md#0]."),
    ])
    result = agent.ask("How do refunds work?")
    assert "[source: doc.md#0]" in result["answer"]
    assert result["stop_reason"] == "completed"
    events = [e["event"] for e in trace.events]
    assert "tool_call" in events and "tool_result" in events


def test_tool_error_is_fed_back_not_raised():
    agent, llm, trace = make_agent([
        ai_tool_call("search_documents", {"query": "boom"}),
        ai_answer("The index was unavailable, retrying was not possible."),
    ])
    result = agent.ask("anything")
    assert result["stop_reason"] == "completed"
    # the error surfaced to the model as a ToolMessage with error status
    error_messages = [
        m for call in llm.calls for m in call
        if isinstance(m, ToolMessage) and getattr(m, "status", None) == "error"
    ]
    assert error_messages and "index unavailable" in error_messages[0].content
    assert any(e["event"] == "tool_error" for e in trace.events)


def test_unknown_tool_is_reported_as_error():
    agent, llm, _ = make_agent([
        ai_tool_call("nonexistent_tool", {"x": 1}),
        ai_answer("done"),
    ])
    result = agent.ask("anything")
    assert result["stop_reason"] == "completed"
    error_messages = [
        m for call in llm.calls for m in call
        if isinstance(m, ToolMessage) and getattr(m, "status", None) == "error"
    ]
    assert error_messages


def test_max_steps_forces_final_answer():
    # model wants to call tools forever; cap at 2 steps
    responses = [ai_tool_call("search_documents", {"query": f"q{i}"}, call_id=f"c{i}")
                 for i in range(10)]
    agent, _, trace = make_agent(responses, max_steps=2)
    result = agent.ask("looping question")
    assert result["stop_reason"].startswith("max_steps")
    assert result["steps"] <= 3  # 2 tool steps + 1 forced final call
    assert any(e["event"] == "limit_hit" for e in trace.events)


def test_token_budget_forces_final_answer():
    responses = [
        ai_tool_call("search_documents", {"query": "q"}, tokens=5000),
        ai_tool_call("search_documents", {"query": "q2"}, call_id="c2", tokens=5000),
    ]
    agent, _, _ = make_agent(responses, max_session_tokens=4000)
    result = agent.ask("expensive question")
    assert result["stop_reason"].startswith("max_session_tokens")


def test_time_budget_forces_final_answer():
    agent, _, _ = make_agent(
        [ai_tool_call("search_documents", {"query": "q"})], max_session_seconds=0.0
    )
    result = agent.ask("slow question")
    assert result["stop_reason"].startswith("max_session_seconds")
