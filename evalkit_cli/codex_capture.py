from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from . import codex_rollout as rollout
from .config import ConfigStore, SharedConfig
from .files import write_atomic
from .otlp import MAX_CONTENT, RETRY_BUDGET, attr, json_text, log_envelope, log_item, send_logs
from .spans import attributes

VENDOR = "codex"
SERVICE = "codex_cli_rs"
ASSISTANT_EVENT = "codex.assistant_message"
REASONING_EVENT = "codex.assistant_reasoning"
OFFSETS_FILE = "codex-offsets.json"
BATCH = 16
MAX_BATCH_BYTES = 1_000_000
TURN_MEMORY = 64
DIGEST = 16
POST_TIMEOUT = 4.0
FULL = "full"
LAST_MESSAGE_FIELDS = ("last_assistant_message", "last-assistant-message", "lastAssistantMessage")
TRANSCRIPT_FIELDS = ("transcript_path", "transcript-path", "agent_transcript_path")
FRACTION = re.compile(r"\.(\d{6})\d+")


@dataclass
class Built:
    records: List[Dict[str, Any]] = field(default_factory=list)
    messages: int = 0
    reasoning: int = 0


@dataclass
class Emission:
    messages: int = 0
    reasoning: int = 0
    read: int = 0
    offset: int = 0
    posted: bool = True


class Offsets:
    def __init__(self, directory: Path) -> None:
        self.path = directory / OFFSETS_FILE

    def _doc(self) -> Dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text() or "{}")
        except (OSError, ValueError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _section(self, doc: Dict[str, Any], name: str) -> Dict[str, Any]:
        value = doc.get(name)
        return dict(value) if isinstance(value, dict) else {}

    def get(self, path: Path) -> int:
        entry = self._section(self._doc(), "files").get(str(path))
        try:
            offset = int(entry.get("offset") if isinstance(entry, dict) else entry)
        except (TypeError, ValueError):
            return 0
        if offset < 0 or offset > _size(path):
            return 0
        return offset

    def seen(self, turn: str) -> bool:
        return bool(turn) and turn in self._section(self._doc(), "turns")

    def commit(self, path: Optional[Path] = None, offset: int = 0, turn: str = "") -> None:
        if path is None and not turn:
            return
        doc = self._doc()
        if path is not None:
            files = self._section(doc, "files")
            files[str(path)] = {"offset": max(int(offset), 0), "at": _now_iso()}
            doc["files"] = files
        if turn:
            turns = self._section(doc, "turns")
            turns[turn] = _now_iso()
            doc["turns"] = dict(sorted(turns.items(), key=lambda row: row[1])[-TURN_MEMORY:])
        write_atomic(self.path, json.dumps(doc, indent=2, sort_keys=True) + "\n", mode=0o600)


def build(items: Iterable[rollout.Item], thread_id: str, capture_full: bool) -> Built:
    out = Built()
    for item in items:
        record = _record(item, thread_id, capture_full)
        if record is None:
            continue
        out.records.append(record)
        if item.kind == rollout.ASSISTANT_MESSAGE:
            out.messages += 1
        else:
            out.reasoning += 1
    return out


def post(
    cfg: SharedConfig, records: Sequence[Dict[str, Any]], timeout: float = POST_TIMEOUT, retry: bool = False
) -> bool:
    # A hook leaves the offsets where they were when a post fails, so its records go out with the next hook;
    # only a backfill, which nothing follows, waits out an overloaded receiver.
    deadline = time.monotonic() + RETRY_BUDGET if retry else None
    for batch in _batches(records):
        status, _ = send_logs(
            cfg.endpoint, cfg.key, log_envelope(VENDOR, batch, service=SERVICE), timeout=timeout, deadline=deadline
        )
        if status is None or status >= 300:
            return False
    return True


def _batches(records: Sequence[Dict[str, Any]]) -> Iterator[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    size = 0
    for record in records:
        weight = len(json_text(record))
        if batch and (len(batch) >= BATCH or size + weight > MAX_BATCH_BYTES):
            yield batch
            batch, size = [], 0
        batch.append(record)
        size += weight
    if batch:
        yield batch


def emit(cfg: SharedConfig, chunk: rollout.Chunk, thread_id: str, dry_run: bool = False) -> Emission:
    built = build(chunk.items, thread_id, cfg.capture == FULL)
    posted = True
    if built.records and not dry_run:
        posted = post(cfg, built.records, retry=True)
    return Emission(built.messages, built.reasoning, chunk.read, chunk.offset, posted)


def cmd_backfill(args, store: ConfigStore, out) -> int:
    cfg = store.load()
    if not args.dry_run and not cfg.key:
        print("no key configured; run `evalkit coding install --vendor codex` first", file=sys.stderr)
        return 2
    found = _select(rollout.sessions(store.home), args)
    if not found:
        wanted = f"session {args.session}" if args.session else "sessions"
        print(f"no codex {wanted} found under {rollout.codex_home(store.home) / rollout.SESSIONS_DIR}", file=out)
        return 0
    offsets = Offsets(store.dir)
    failed = False
    for session in found:
        start = 0 if args.force else offsets.get(session.path)
        result = emit(cfg, rollout.read(session.path, start), session.thread_id, args.dry_run)
        if result.posted and not args.dry_run:
            offsets.commit(session.path, result.offset)
        failed = failed or not result.posted
        print(_line(session, result, cfg, args.dry_run), file=out)
    return 1 if failed else 0


def hook_emit(store: ConfigStore, cfg: SharedConfig, payload: Dict[str, Any]) -> int:
    thread = _field(payload, rollout.THREAD_FIELDS)
    turn = _field(payload, rollout.TURN_FIELDS)
    last = _field(payload, LAST_MESSAGE_FIELDS)
    path = _transcript(payload, thread, store.home)
    offsets = Offsets(store.dir)
    chunk = rollout.read(path, offsets.get(path)) if path is not None else rollout.Chunk()
    if not thread and path is not None:
        thread = rollout.thread_id(path)
    built = build(chunk.items, thread, cfg.capture == FULL)
    fallback = _fallback_id(thread, turn, last) if last else ""
    carried = bool(fallback) and _carries(chunk.items, last)
    spoken = bool(fallback) and not carried and not offsets.seen(fallback)
    if spoken:
        trace = chunk.traces.get(turn) or (rollout.traces_in(path).get(turn, "") if path is not None else "")
        spare = rollout.Item(rollout.ASSISTANT_MESSAGE, turn, fallback, last, at=_now_iso(), trace_id=trace)
        record = _record(spare, thread, cfg.capture == FULL)
        if record is not None:
            built.records.append(record)
            built.messages += 1
    if built.records and not post(cfg, built.records):
        return 0
    offsets.commit(path, chunk.offset, fallback if spoken or carried else "")
    return built.messages


def _fallback_id(thread_id: str, turn: str, message: str) -> str:
    digest = hashlib.sha256(message.strip().encode("utf-8", "replace")).hexdigest()[:DIGEST]
    return f"{thread_id}:{turn}:{digest}"


def _record(item: rollout.Item, thread_id: str, capture_full: bool) -> Optional[Dict[str, Any]]:
    if item.kind == rollout.REASONING:
        if not capture_full or not item.text.strip():
            return None
        name = REASONING_EVENT
    elif item.kind == rollout.ASSISTANT_MESSAGE:
        name = ASSISTANT_EVENT
    else:
        return None
    values = {
        "conversation.id": thread_id,
        "turn.id": item.turn_id,
        "item.id": item_id(item, thread_id),
        "message.phase": item.phase,
    }
    attrs = attributes(values)
    if capture_full and item.text:
        attrs.append(attr("completion", item.text, limit=MAX_CONTENT))
    return log_item(name, attrs, trace=item.trace_id or None, at_ns=_stamp_ns(item))


def item_id(item: rollout.Item, thread_id: str) -> str:
    return item.item_id or f"{thread_id}:{item.turn_id}:{item.ordinal}"


def _stamp_ns(item: rollout.Item) -> int:
    base = epoch_ns(item.at)
    return (base if base is not None else time.time_ns()) + max(item.ordinal, 0)


def _select(sessions: List[rollout.Session], args) -> List[rollout.Session]:
    if args.session:
        return [session for session in sessions if session.thread_id == args.session]
    since = epoch_ns(args.since) if args.since else None
    if since is not None:
        sessions = [session for session in sessions if (epoch_ns(session.updated_at) or 0) >= since]
    if args.all:
        return sessions
    if args.last:
        return sessions[: max(args.last, 0)]
    return sessions[:1]


def _line(session: rollout.Session, result: Emission, cfg: SharedConfig, dry_run: bool) -> str:
    parts = [
        session.thread_id,
        session.name or "(unnamed)",
        f"{result.messages} messages",
        f"{result.read} bytes",
    ]
    if result.reasoning:
        parts.append(f"{result.reasoning} reasoning")
    if cfg.capture != FULL:
        parts.append("text off (capture minimal)")
    if not result.posted:
        parts.append("post failed, offset kept")
    line = " · ".join(parts)
    return f"dry-run: {line}" if dry_run else line


def _transcript(payload: Dict[str, Any], thread_id: str, home: Path) -> Optional[Path]:
    for name in TRANSCRIPT_FIELDS:
        value = payload.get(name)
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser()
            if path.is_file():
                return path
    return rollout.path_for(thread_id, home) if thread_id else None


def _carries(items: Sequence[rollout.Item], message: str) -> bool:
    wanted = message.strip()
    return any(item.kind == rollout.ASSISTANT_MESSAGE and item.text.strip() == wanted for item in items)


def _field(payload: Dict[str, Any], fields: Tuple[str, ...]) -> str:
    for name in fields:
        value = payload.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def epoch_ns(stamp: Optional[str]) -> Optional[int]:
    text = (stamp or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(FRACTION.sub(r".\1", text))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000_000_000)


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
