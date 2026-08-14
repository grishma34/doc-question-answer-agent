"""The tool-calling agent loop, built as a LangGraph state graph.

Graph shape:

    START -> agent -> (tool_calls? within budget?) -> tools -> agent -> ... -> END

Three properties from the resume bullets are implemented here:

1. Grounded answers: the system prompt requires the model to search before
   answering and to cite [source: ...] ids from tool results.
2. Cost/latency caps: the loop stops after ``max_steps`` round trips,
   ``max_session_tokens`` total tokens, or ``max_session_seconds`` wall time,
   and forces a final answer from whatever context was gathered.
3. Error feedback: a tool exception is returned to the model as an error
   ToolMessage instead of crashing the run, so the model can retry with
   corrected input.
"""

import time
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .config import Settings
from .trace import TraceLogger

SYSTEM_PROMPT = """\
You are a document question-answering assistant.

Rules:
- Always call search_documents before answering. Never answer from prior
  knowledge alone.
- Answer ONLY from the retrieved passages. Cite every claim with the source
  ids shown in the results, e.g. [source: billing_faq.md#2].
- If the retrieved passages don't contain the answer, try one rephrased
  search; if it still isn't there, say the documents don't cover it.
- Keep answers concise and factual.
"""

FORCE_ANSWER_PROMPT = (
    "Session budget reached ({reason}). Do not call any more tools. "
    "Answer the user's question now using only the passages already "
    "retrieved above, with citations. If they are insufficient, say so."
)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    steps: int
    tokens_used: int
    started_at: float
    stop_reason: str


class DocQAAgent:
    def __init__(self, llm, tools: list, settings: Settings, trace: TraceLogger):
        self.llm = llm
        self.llm_with_tools = llm.bind_tools(tools)
        self.tools_by_name = {t.name: t for t in tools}
        self.settings = settings
        self.trace = trace
        self.graph = self._build_graph()

    # --- graph nodes -----------------------------------------------------

    def _agent_node(self, state: AgentState) -> dict:
        limit = self._limit_hit(state)
        messages = list(state["messages"])
        if limit:
            # Budget exhausted: strip tool access and force a final answer
            # from the context gathered so far.
            self.trace.log("limit_hit", reason=limit, steps=state["steps"],
                           tokens_used=state["tokens_used"])
            messages.append(HumanMessage(content=FORCE_ANSWER_PROMPT.format(reason=limit)))
            model = self.llm
        else:
            model = self.llm_with_tools

        self.trace.log("llm_call", step=state["steps"], forced_final=bool(limit))
        response: AIMessage = model.invoke(messages)
        tokens = _total_tokens(response)
        self.trace.log(
            "llm_response",
            step=state["steps"],
            tokens=tokens,
            tool_calls=[tc["name"] for tc in getattr(response, "tool_calls", []) or []],
        )
        update = {
            "messages": [FORCE_ANSWER_MARKER, response] if limit else [response],
            "steps": state["steps"] + 1,
            "tokens_used": state["tokens_used"] + tokens,
        }
        if limit:
            update["stop_reason"] = limit
        return update

    def _tools_node(self, state: AgentState) -> dict:
        last = state["messages"][-1]
        results: list[ToolMessage] = []
        for call in last.tool_calls:
            self.trace.log("tool_call", name=call["name"], args=call["args"], id=call["id"])
            tool = self.tools_by_name.get(call["name"])
            try:
                if tool is None:
                    raise KeyError(f"Unknown tool: {call['name']}")
                output = tool.invoke(call["args"])
                results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
                self.trace.log("tool_result", name=call["name"], id=call["id"],
                               chars=len(str(output)))
            except Exception as exc:  # feed the error back instead of crashing
                error_text = f"Tool error ({type(exc).__name__}): {exc}. " \
                             "Adjust your input and try again, or answer from what you have."
                results.append(
                    ToolMessage(content=error_text, tool_call_id=call["id"], status="error")
                )
                self.trace.log("tool_error", name=call["name"], id=call["id"], error=str(exc))
        return {"messages": results}

    # --- control flow ----------------------------------------------------

    def _limit_hit(self, state: AgentState) -> str | None:
        s = self.settings
        if state["steps"] >= s.max_steps:
            return f"max_steps ({s.max_steps})"
        if state["tokens_used"] >= s.max_session_tokens:
            return f"max_session_tokens ({s.max_session_tokens})"
        if time.monotonic() - state["started_at"] >= s.max_session_seconds:
            return f"max_session_seconds ({s.max_session_seconds})"
        return None

    def _route(self, state: AgentState) -> str:
        if state["stop_reason"] != "completed":
            return END  # a forced final answer always ends the session
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", self._route, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
        return graph.compile()

    # --- public API ------------------------------------------------------

    def ask(self, question: str) -> dict:
        """Run one QA session. Returns answer text plus session stats."""
        self.trace.log("session_start", question=question)
        initial: AgentState = {
            "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)],
            "steps": 0,
            "tokens_used": 0,
            "started_at": time.monotonic(),
            "stop_reason": "completed",
        }
        final = self.graph.invoke(
            initial, config={"recursion_limit": 4 * self.settings.max_steps + 10}
        )
        answer = _last_ai_text(final["messages"])
        result = {
            "answer": answer,
            "steps": final["steps"],
            "tokens_used": final["tokens_used"],
            "stop_reason": final["stop_reason"],
            "session_id": self.trace.session_id,
        }
        self.trace.log("session_end", **{k: v for k, v in result.items() if k != "answer"})
        return result


# A marker message so the transcript records that the final answer was forced.
FORCE_ANSWER_MARKER = HumanMessage(content="[budget-exceeded: final answer forced]")


def _total_tokens(message: AIMessage) -> int:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("total_tokens", 0))


def _last_ai_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, list):  # Bedrock can return content blocks
                return "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                )
            return str(content)
    return ""
