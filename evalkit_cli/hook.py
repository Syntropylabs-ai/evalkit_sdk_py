from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from . import codex_capture, turns
from .config import ConfigStore
from .otlp import attr, log_record, send_logs, send_traces
from .turns import LOGS, TRACES, Record

_STDIN_CAP = 1024 * 1024
_HARD_TIMEOUT = 5.0

CLAUDE_CODE = "claude-code"
CODEX = "codex"
CLAUDE_EVENT_PREFIX = "claude_code"
CLAUDE_EVENTS = {
    "PostToolUse": "tool_output",
    "SubagentStart": "subagent_start",
    "SubagentStop": "subagent_stop",
}
_SUBAGENT_FIELDS = (
    ("session_id", "session.id"),
    ("agent_id", "agent.id"),
    ("agent_type", "agent.type"),
    ("cwd", "cwd"),
    ("permission_mode", "permission_mode"),
)
_TOOL_FIELDS = (
    ("session_id", "session.id"),
    ("tool_use_id", "tool_use_id"),
    ("tool_name", "tool_name"),
    ("duration_ms", "duration_ms"),
    ("agent_id", "agent.id"),
    ("agent_type", "agent.type"),
)
_CONTENT_FIELD = {
    "tool_output": ("tool_response", "tool_output"),
    "subagent_stop": ("last_assistant_message", "agent.last_message"),
}


def _read_payload() -> Dict[str, Any]:
    try:
        raw = sys.stdin.read(_STDIN_CAP)
        data = json.loads(raw) if raw.strip() else {}
        return data if isinstance(data, dict) else {"payload": data}
    except Exception:
        return {}


def _fields(name: str) -> Tuple[Tuple[str, str], ...]:
    return _TOOL_FIELDS if name == "tool_output" else _SUBAGENT_FIELDS


def _claude_record(event: str, payload: Dict[str, Any], capture_content: bool) -> Optional[Dict[str, Any]]:
    name = CLAUDE_EVENTS.get(event)
    if name is None:
        return None
    attrs = [attr(key, payload[field]) for field, key in _fields(name) if payload.get(field) not in (None, "")]
    content = _CONTENT_FIELD.get(name)
    if capture_content and content and payload.get(content[0]) not in (None, ""):
        attrs.append(attr(content[1], payload[content[0]]))
    return log_record(CLAUDE_CODE, name, attrs, prefix=CLAUDE_EVENT_PREFIX)


def build_records(vendor: str, event: str, payload: Dict[str, Any], capture_content: bool, now_ns: Optional[int] = None) -> List[Record]:
    if vendor == CLAUDE_CODE:
        record = _claude_record(event, payload, capture_content)
        return [(LOGS, record)] if record is not None else []
    return turns.build(vendor, event, payload, capture_content, now_ns if now_ns is not None else time.time_ns())


def run_hook(vendor: str, event: str) -> int:
    # Retries stop a second short of the hard timeout, so the last answer is read before the hook returns.
    deadline = time.monotonic() + _HARD_TIMEOUT - 1

    def work() -> None:
        try:
            store = ConfigStore()
            cfg = store.load()
            if not cfg.key:
                return
            payload = _read_payload()
            if vendor == CODEX:
                codex_capture.hook_emit(store, cfg, payload)
                return
            timeout = min(4.0, _HARD_TIMEOUT - 1)
            for signal, body in build_records(vendor, event, payload, cfg.capture == "full"):
                sender = send_traces if signal == TRACES else send_logs
                sender(cfg.endpoint, cfg.key, body, timeout=timeout, deadline=deadline)
        except Exception:
            return

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(_HARD_TIMEOUT)
    return 0
