import sys
from collections import deque
from pathlib import Path

from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class FakeToolCallingLLM:
    """Stands in for ChatBedrockConverse: returns scripted AIMessages."""

    def __init__(self, responses: list[AIMessage]):
        self.responses = deque(responses)
        self.calls: list[list] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        if self.responses:
            return self.responses.popleft()
        return AIMessage(
            content="fallback answer",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


def ai_tool_call(name: str, args: dict, call_id: str = "call-1", tokens: int = 100) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        usage_metadata={"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens},
    )


def ai_answer(text: str, tokens: int = 50) -> AIMessage:
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": tokens, "output_tokens": tokens, "total_tokens": tokens},
    )
