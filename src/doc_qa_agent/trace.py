"""JSONL tracing for agent sessions.

Every model call, tool call, tool result, tool error, and limit event is
appended to one JSONL file per session so a bad answer can be replayed
step by step (resume bullet #2).
"""

import json
import time
import uuid
from pathlib import Path


class TraceLogger:
    def __init__(self, trace_dir: str | Path, session_id: str | None = None):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.path = Path(trace_dir) / f"session-{self.session_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, **data) -> None:
        record = {
            "ts": round(time.time(), 3),
            "session_id": self.session_id,
            "event": event_type,
            **data,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class NullTraceLogger(TraceLogger):
    """Trace sink for tests — records in memory, writes nothing."""

    def __init__(self):  # noqa: D107 - intentionally skips file setup
        self.session_id = "test"
        self.path = None
        self.events: list[dict] = []

    def log(self, event_type: str, **data) -> None:
        self.events.append({"event": event_type, **data})
