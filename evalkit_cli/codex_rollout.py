from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ASSISTANT_MESSAGE = "assistant_message"
REASONING = "reasoning"
TOOL_CALL = "tool_call"
TOOL_OUTPUT = "tool_output"
USAGE = "usage"

CODEX_DIR = ".codex"
CODEX_HOME_ENV = "CODEX_HOME"
SESSIONS_DIR = "sessions"
INDEX_FILE = "session_index.jsonl"
ROLLOUT_PREFIX = "rollout-"
ROLLOUT_GLOB = "rollout-*.jsonl"
THREAD_ID_PARTS = 5
ASSISTANT_ROLE = "assistant"
METADATA_KEY = "internal_chat_message_metadata_passthrough"
TURN_FIELDS = ("turn_id", "turn-id", "root_turn_id")
THREAD_FIELDS = ("thread_id", "thread-id", "session_id", "conversation_id")

TOOL_CALLS = {
    "function_call": "",
    "custom_tool_call": "",
    "local_shell_call": "local_shell",
    "web_search_call": "web_search",
    "tool_search_call": "tool_search",
}
TOOL_OUTPUTS = {
    "function_call_output": "",
    "custom_tool_call_output": "",
    "local_shell_call_output": "local_shell",
    "web_search_call_output": "web_search",
    "tool_search_output": "tool_search",
}
USAGE_KINDS = ("token_usage_record",)
TASK_STARTED = "task_started"
TASK_STARTED_HINT = b"task_started"
TRACE_FIELDS = ("trace_id", "trace-id")


@dataclass(frozen=True)
class Item:
    kind: str
    turn_id: str = ""
    item_id: str = ""
    text: str = ""
    tool_name: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    at: str = ""
    phase: str = ""
    ordinal: int = 0
    trace_id: str = ""


@dataclass(frozen=True)
class Chunk:
    items: List[Item] = field(default_factory=list)
    offset: int = 0
    read: int = 0
    traces: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Session:
    thread_id: str
    name: str
    updated_at: str
    path: Path


def codex_home(home: Optional[Path] = None) -> Path:
    override = os.getenv(CODEX_HOME_ENV)
    if override:
        return Path(override)
    return (home or Path.home()) / CODEX_DIR


def thread_id(path: Path) -> str:
    stem = path.name[: -len(".jsonl")] if path.name.endswith(".jsonl") else path.name
    if stem.startswith(ROLLOUT_PREFIX):
        stem = stem[len(ROLLOUT_PREFIX) :]
    parts = stem.split("-")
    if len(parts) < THREAD_ID_PARTS:
        return stem
    return "-".join(parts[-THREAD_ID_PARTS:])


def rollouts(home: Optional[Path] = None) -> List[Path]:
    root = codex_home(home) / SESSIONS_DIR
    if not root.is_dir():
        return []
    return [path for path in root.rglob(ROLLOUT_GLOB) if path.is_file()]


def index(home: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    path = codex_home(home) / INDEX_FILE
    out: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return out
    for line in _lines(path):
        row = _loads(line)
        if isinstance(row, dict) and row.get("id"):
            out[str(row["id"])] = row
    return out


def sessions(home: Optional[Path] = None) -> List[Session]:
    names = index(home)
    found: Dict[str, Session] = {}
    for path in sorted(rollouts(home), key=_mtime, reverse=True):
        identifier = thread_id(path)
        if identifier in found:
            continue
        row = names.get(identifier) or {}
        updated = str(row.get("updated_at") or "") or _iso(_mtime(path))
        found[identifier] = Session(identifier, str(row.get("thread_name") or ""), updated, path)
    return sorted(found.values(), key=lambda session: session.updated_at, reverse=True)


def session_for(identifier: str, home: Optional[Path] = None) -> Optional[Session]:
    for session in sessions(home):
        if session.thread_id == identifier:
            return session
    return None


def path_for(identifier: str, home: Optional[Path] = None) -> Optional[Path]:
    session = session_for(identifier, home)
    return session.path if session else None


def read(path: Path, offset: int = 0) -> Chunk:
    try:
        with open(path, "rb") as handle:
            handle.seek(max(offset, 0))
            blob = handle.read()
    except OSError:
        return Chunk([], max(offset, 0), 0)
    cut = blob.rfind(b"\n")
    if cut < 0:
        return Chunk([], max(offset, 0), 0)
    body = blob[: cut + 1]
    items: List[Item] = []
    traces: Dict[str, str] = {}
    position = max(offset, 0)
    for raw in body.split(b"\n"):
        line = position
        position += len(raw) + 1
        if not raw.strip():
            continue
        record = _loads(raw.decode("utf-8", "replace"))
        _collect_trace(record, traces)
        item = normalise(record, line)
        if item is not None:
            items.append(item)
    if offset > 0 and any(item.turn_id and item.turn_id not in traces for item in items):
        for turn, trace in traces_in(path, max(offset, 0)).items():
            traces.setdefault(turn, trace)
    return Chunk([replace(item, trace_id=traces.get(item.turn_id, "")) for item in items], max(offset, 0) + len(body), len(body), traces)


def traces_in(path: Path, limit: Optional[int] = None) -> Dict[str, str]:
    traces: Dict[str, str] = {}
    if limit is not None and limit <= 0:
        return traces
    try:
        with open(path, "rb") as handle:
            blob = handle.read() if limit is None else handle.read(limit)
    except OSError:
        return traces
    for raw in blob.split(b"\n"):
        if TASK_STARTED_HINT in raw:
            _collect_trace(_loads(raw.decode("utf-8", "replace")), traces)
    return traces


def _collect_trace(record: Any, traces: Dict[str, str]) -> None:
    if not isinstance(record, dict):
        return
    _, payload = _split(record)
    if payload.get("type") != TASK_STARTED:
        return
    turn = _first(payload, TURN_FIELDS)
    trace = _first(payload, TRACE_FIELDS)
    if turn and trace:
        traces[turn] = trace


def normalise(record: Any, position: int = 0) -> Optional[Item]:
    if not isinstance(record, dict):
        return None
    outer, payload = _split(record)
    kind = str(payload.get("type") or "") or outer
    at = _timestamp(record, payload)
    turn = _first(payload, TURN_FIELDS) or _first(_metadata(payload), TURN_FIELDS) or _first(record, TURN_FIELDS)
    ordinal = _int(record.get("ordinal")) or _int(payload.get("ordinal")) or max(position, 1)
    if kind == "message":
        return _message(payload, turn, at, ordinal)
    if kind == "reasoning":
        return Item(REASONING, turn, _text(payload.get("id")), _reasoning_text(payload), at=at, ordinal=ordinal)
    if kind in TOOL_CALLS:
        return _tool(TOOL_CALL, TOOL_CALLS[kind], payload, turn, at, ordinal)
    if kind in TOOL_OUTPUTS:
        return _tool(TOOL_OUTPUT, TOOL_OUTPUTS[kind], payload, turn, at, ordinal)
    if outer in USAGE_KINDS or kind in USAGE_KINDS:
        return _usage(payload, turn, at, ordinal)
    return None


def _message(payload: Dict[str, Any], turn: str, at: str, ordinal: int) -> Optional[Item]:
    if _text(payload.get("role")) != ASSISTANT_ROLE:
        return None
    return Item(
        ASSISTANT_MESSAGE,
        turn,
        _text(payload.get("id")),
        _content(payload.get("content")),
        at=at,
        phase=_text(payload.get("phase")),
        ordinal=ordinal,
    )


def _tool(kind: str, label: str, payload: Dict[str, Any], turn: str, at: str, ordinal: int) -> Item:
    name = _text(payload.get("name")) or label
    body = payload.get("output") if kind == TOOL_OUTPUT else payload.get("arguments")
    if body is None:
        body = payload.get("input") if kind == TOOL_CALL else payload.get("result")
    if body is None:
        body = payload.get("action") or payload.get("tools")
    return Item(
        kind,
        turn,
        _text(payload.get("call_id")) or _text(payload.get("id")),
        _content(body),
        tool_name=name,
        at=at,
        ordinal=ordinal,
    )


def _usage(payload: Dict[str, Any], turn: str, at: str, ordinal: int) -> Item:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        usage = payload.get("turn_token_usage") if isinstance(payload.get("turn_token_usage"), dict) else {}
    return Item(
        USAGE,
        turn,
        _text(payload.get("response_id")),
        tokens_in=_int(usage.get("input_tokens")),
        tokens_out=_int(usage.get("output_tokens")),
        at=at,
        ordinal=ordinal,
    )


def _split(record: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    payload = record.get("payload")
    if isinstance(payload, dict):
        return _text(record.get("type")), payload
    return _text(record.get("type")), record


def _metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    value = payload.get(METADATA_KEY)
    return value if isinstance(value, dict) else {}


def _first(source: Dict[str, Any], fields: Tuple[str, ...]) -> str:
    for name in fields:
        found = _text(source.get(name))
        if found:
            return found
    return ""


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_content(entry) for entry in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
        for key in ("content", "output", "query", "input"):
            if key in value:
                return _content(value[key])
        return json.dumps(value, separators=(",", ":"), default=str)
    return "" if value is None else str(value)


def _reasoning_text(payload: Dict[str, Any]) -> str:
    for key in ("summary", "content", "text"):
        found = _content(payload.get(key))
        if found:
            return found
    return ""


def _timestamp(record: Dict[str, Any], payload: Dict[str, Any]) -> str:
    stamp = _text(record.get("timestamp")) or _text(payload.get("timestamp"))
    if stamp:
        return stamp
    for source in (payload, _metadata(payload)):
        seconds = _number(source.get("create_time"))
        if seconds is None:
            millis = _number(source.get("completed_at_ms"))
            seconds = None if millis is None else millis / 1000.0
        if seconds is not None:
            return _iso(seconds)
    return ""


def _number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace("+00:00", "Z")


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _lines(path: Path) -> List[str]:
    try:
        return path.read_text(errors="replace").splitlines()
    except OSError:
        return []


def _loads(line: str) -> Any:
    try:
        return json.loads(line)
    except Exception:
        return None


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0
