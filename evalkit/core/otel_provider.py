from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from .content import compact_json, message_content_text, parse_json_text
from .context import AGENT_NAME_KEYS, SESSION_ID_KEYS, USER_ID_KEYS, first_attr, tracing_suppressed
from .types import SpanEvent, TraceEnvelope

_log = logging.getLogger("evalkit")

_started = False
_processor: Any = None


def otel_active() -> bool:
    return _started


def flush_otel(timeout_millis: int = 5000) -> None:
    processor = _processor
    if processor is None:
        return
    try:
        processor.force_flush(timeout_millis)
    except Exception as exc:
        _log.debug("otel force_flush failed: %s", exc)

_SKIP_INSTRUMENTORS = {
    "openai", "anthropic", "google-generativeai", "google-genai", "google_genai",
    "vertexai", "cohere", "mistralai", "groq", "langchain", "litellm",
    "replicate", "transformers",
    "asgi", "wsgi", "fastapi", "flask", "django", "starlette",
    "requests", "httpx", "httpx2", "aiohttp-client",
    "redis", "pymongo", "asyncpg", "psycopg", "psycopg2", "pymysql",
}

_OWNED_SCOPES: set = set()


def mark_owned_scope(*scopes: str) -> None:
    for scope in scopes:
        if scope:
            _OWNED_SCOPES.add(scope)


def _duplicates_own_span(scope: str) -> bool:
    return bool(scope) and scope in _OWNED_SCOPES


def _ns_to_iso(ns: Optional[int]) -> Optional[str]:
    if not ns:
        return None
    try:
        return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def _latency_ms(span: Any) -> Optional[int]:
    start = getattr(span, "start_time", None)
    end = getattr(span, "end_time", None)
    if start and end and end >= start:
        return int((end - start) / 1_000_000)
    return None


def _infer_span_type(attrs: Dict[str, Any], scope: str) -> str:
    if any(k in attrs for k in ("gen_ai.request.model", "gen_ai.system", "gen_ai.response.model",
                                "gen_ai.provider.name")):
        return "llm_call"
    if any(k in attrs for k in ("db.system", "db.statement", "db.operation", "db.mongodb.collection")):
        return "db_query"
    if any(k in attrs for k in ("http.method", "http.request.method", "url.full", "http.url")):
        return "http_call"
    s = (scope or "").lower()
    if any(x in s for x in ("sql", "mongo", "redis", "mysql", "postgres", "cassandra", "elastic", "memcache")):
        return "db_query"
    if any(x in s for x in ("http", "requests", "urllib", "grpc", "kafka", "pika", "celery", "botocore")):
        return "http_call"
    return "function_call"


def _count_foreign_llm() -> bool:
    import os
    return os.environ.get("EVALKIT_COUNT_FOREIGN_LLM", "").strip().lower() in ("1", "true", "yes")


_TOKENS_IN_KEYS = ("gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens")
_TOKENS_OUT_KEYS = ("gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens")
_CACHE_READ_KEYS = (
    "gen_ai.usage.cache_read_tokens",
    "gen_ai.usage.cache_read.input_tokens",
    "gen_ai.usage.cached_tokens",
)
_CACHE_CREATION_KEYS = (
    "gen_ai.usage.cache_creation_tokens",
    "gen_ai.usage.cache_creation.input_tokens",
    "gen_ai.usage.cache_write.input_tokens",
)
_PROVIDER_KEYS = ("gen_ai.system", "gen_ai.provider.name")
_LLM_MARKER_KEYS = (
    "gen_ai.system", "gen_ai.provider.name", "gen_ai.operation.name",
    "gen_ai.request.model", "gen_ai.response.model",
) + _TOKENS_IN_KEYS + _TOKENS_OUT_KEYS


def _token_count(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        n = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None
    return n if n > 0 else None


def _first_token_count(attrs: Dict[str, Any], keys: tuple) -> Optional[int]:
    for key in keys:
        n = _token_count(attrs.get(key))
        if n is not None:
            return n
    return None


def _fold_exclusive_cache(
    tokens_in: Optional[int], cache_read: Optional[int], cache_creation: Optional[int]
) -> Optional[int]:
    cached = (cache_read or 0) + (cache_creation or 0)
    if tokens_in is None or cached == 0:
        return tokens_in
    return tokens_in + cached if tokens_in < cached else tokens_in


_SPAN_KIND_NAMES = ("INTERNAL", "SERVER", "CLIENT", "PRODUCER", "CONSUMER")


def _as_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _span_kind(span: Any) -> Optional[str]:
    kind = getattr(span, "kind", None)
    if kind is None:
        return None
    name = getattr(kind, "name", None)
    if isinstance(name, str) and name in _SPAN_KIND_NAMES:
        return name
    try:
        index = int(kind)
    except (TypeError, ValueError):
        return None
    return _SPAN_KIND_NAMES[index] if 0 <= index < len(_SPAN_KIND_NAMES) else None


def _span_events(span: Any, fallback_ts: str) -> Optional[List[SpanEvent]]:
    out: List[SpanEvent] = []
    for ev in (getattr(span, "events", None) or ()):
        try:
            out.append(SpanEvent(
                name=_as_str(getattr(ev, "name", None)) or "",
                timestamp=_ns_to_iso(getattr(ev, "timestamp", None)) or fallback_ts,
                attributes=dict(getattr(ev, "attributes", None) or {}) or None,
            ))
        except Exception:
            continue
    return out or None


# (user_id, session_id, device_id, agent_name) as they were where the span started.
_Scope = Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]

# Spans that start but never end must not grow this without bound.
_MAX_TRACKED_SPANS = 10000
_started_scopes: "OrderedDict[Tuple[int, int], _Scope]" = OrderedDict()
# Re-entrant, as the export path can run again from a garbage-collected stream on the same thread.
_scopes_lock = threading.RLock()


def _reset_lock_after_fork() -> None:
    # A lock another thread held at the fork stays held in the child.
    global _scopes_lock
    _scopes_lock = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_lock_after_fork)
# OTel trace id -> (evalkit trace id, evalkit parent span id, OTel root span id). A span from
# another OpenTelemetry library that starts with no OTel parent while an evalkit span is current
# belongs to that evalkit trace; without this it and its children would export as a trace of
# their own, split off from the turn that caused them.
_adopted_traces: "OrderedDict[int, Tuple[str, str, int]]" = OrderedDict()

# Request spans a web framework opens itself, before any middleware runs, by instrumentation scope.
# FastAPI's built-in telemetry ("fastapi") starts its server span outside the app's middleware stack,
# so EvalKit's middleware always runs inside it. The middleware's request span takes that span's ids:
# everything opened under the framework span (by the framework or another OpenTelemetry library)
# then lands in the request's one trace, and the framework's copy of the request is not exported.
_FRAMEWORK_SERVER_SCOPES = frozenset({
    "fastapi",
    "starlette",
    "opentelemetry.instrumentation.asgi",
    "opentelemetry.instrumentation.fastapi",
    "opentelemetry.instrumentation.starlette",
})
# Framework scopes whose internal spans (FastAPI's dependencies, endpoint, serialization and
# background_task spans) are left out of a request EvalKit's middleware records. EvalKit's own spans
# hang off the request span, not off these, so keeping them would draw siblings that only overlap
# the real work, and add three or four spans to every request. A span opened under one is
# re-parented to the nearest span that is kept.
_FRAMEWORK_INTERNAL_SCOPES = frozenset({"fastapi"})
# (OTel trace id, span id) of framework request spans the middleware's request span replaced.
_claimed_spans: "OrderedDict[Tuple[int, int], None]" = OrderedDict()
# OTel trace ids whose request span the middleware claimed.
_owned_requests: "OrderedDict[int, None]" = OrderedDict()
# (OTel trace id, span id) of a left-out framework span -> span id of its nearest kept ancestor.
_elided_spans: "OrderedDict[Tuple[int, int], int]" = OrderedDict()
# (OTel trace id, span id) -> id of the EvalKit span that was current when it started.
_started_inside: "OrderedDict[Tuple[int, int], str]" = OrderedDict()


def _remember(store: "OrderedDict", key: Any, value: Any) -> None:
    store[key] = value
    while len(store) > _MAX_TRACKED_SPANS:
        store.popitem(last=False)


def _scope_name(span: Any) -> str:
    return getattr(getattr(span, "instrumentation_scope", None), "name", "") or ""


class FrameworkSpan(NamedTuple):
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    span: Any


def claim_framework_server_span() -> Optional[FrameworkSpan]:
    """The request span a framework already opened for this request, for an HTTP middleware to take over.

    Returns its ids (hex) and the span itself, or None when the middleware is not running inside one.
    The claimed span is dropped at export; the middleware's own request span stands in for it.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.trace import SpanKind

        span = trace.get_current_span()
        if not span.is_recording() or getattr(span, "kind", None) != SpanKind.SERVER:
            return None
        if _scope_name(span) not in _FRAMEWORK_SERVER_SCOPES:
            return None
        sc = span.get_span_context()
        if not sc.is_valid:
            return None
        parent = getattr(span, "parent", None)
        parent_id = parent.span_id if parent is not None and parent.is_valid else None
    except Exception:
        return None
    with _scopes_lock:
        _remember(_claimed_spans, (sc.trace_id, sc.span_id), None)
        _remember(_owned_requests, sc.trace_id, None)
    return FrameworkSpan(
        format(sc.trace_id, "032x"),
        format(sc.span_id, "016x"),
        format(parent_id, "016x") if parent_id else None,
        span,
    )


def release_framework_server_span(claimed: Optional[FrameworkSpan]) -> None:
    """Hand a claimed span back when the middleware could not open its own: it exports as recorded."""
    if claimed is None:
        return
    try:
        trace_id, span_id = int(claimed.trace_id, 16), int(claimed.span_id, 16)
    except (TypeError, ValueError):
        return
    with _scopes_lock:
        _claimed_spans.pop((trace_id, span_id), None)
        _owned_requests.pop(trace_id, None)


def _note_framework_internal(span: Any, key: Tuple[int, int]) -> None:
    if not _owned_requests or _scope_name(span) not in _FRAMEWORK_INTERNAL_SCOPES:
        return
    parent = getattr(span, "parent", None)
    if parent is None:
        return
    with _scopes_lock:
        if key[0] not in _owned_requests:
            return
        # A parent left out itself already points at its own kept ancestor.
        _remember(_elided_spans, key, _elided_spans.get((key[0], parent.span_id), parent.span_id))


def _is_framework_duplicate(span: Any) -> bool:
    key = _span_key(span)
    if key is None or not (_claimed_spans or _elided_spans):
        return False
    with _scopes_lock:
        if key in _claimed_spans:
            del _claimed_spans[key]
            return True
        return key in _elided_spans


def _span_key(span: Any) -> Optional[Tuple[int, int]]:
    try:
        sc = span.get_span_context()
        return (sc.trace_id, sc.span_id)
    except Exception:
        return None


def capture_start_scope(span: Any) -> None:
    """Record the ambient identity and agent on the thread that starts the span.

    Export runs later on the BatchSpanProcessor worker thread, whose context holds none of the
    caller's set_user / session / set_agent scope, so the values have to be taken here.
    """
    key = _span_key(span)
    if key is None:
        return
    _note_framework_internal(span, key)
    from .context import _current_span, current_agent, current_identity

    ctx = _current_span.get()
    if ctx is not None:
        with _scopes_lock:
            _remember(_started_inside, key, ctx.span_id)
    if ctx is not None and getattr(span, "parent", None) is None:
        with _scopes_lock:
            _adopted_traces[key[0]] = (ctx.trace_id, ctx.span_id, key[1])
            while len(_adopted_traces) > _MAX_TRACKED_SPANS:
                _adopted_traces.popitem(last=False)
    identity = current_identity()
    scope: _Scope = (
        (ctx.user_id if ctx else None) or identity.user_id,
        (ctx.session_id if ctx else None) or identity.session_id,
        (ctx.device_id if ctx else None) or identity.device_id,
        (ctx.agent_name if ctx else None) or current_agent(),
    )
    if not any(scope):
        return
    with _scopes_lock:
        _started_scopes[key] = scope
        while len(_started_scopes) > _MAX_TRACKED_SPANS:
            _started_scopes.popitem(last=False)


# Spans another instrumentation started while a health probe was answered. They are exported from
# the processor's own thread, where the probe's context is gone, so they are told apart as they start.
_suppressed_spans: "OrderedDict[Tuple[int, int], None]" = OrderedDict()


def _note_suppressed(span: Any) -> None:
    key = _span_key(span)
    if key is not None:
        with _scopes_lock:
            _remember(_suppressed_spans, key, None)


def _take_suppressed(span: Any) -> bool:
    if not _suppressed_spans:
        return False
    key = _span_key(span)
    if key is None:
        return False
    with _scopes_lock:
        if key not in _suppressed_spans:
            return False
        del _suppressed_spans[key]
        return True


def _take_start_scope(span: Any) -> _Scope:
    key = _span_key(span)
    if key is None:
        return (None, None, None, None)
    with _scopes_lock:
        return _started_scopes.pop(key, None) or (None, None, None, None)


def _take_started_inside(span: Any) -> Optional[str]:
    key = _span_key(span)
    if key is None:
        return None
    with _scopes_lock:
        return _started_inside.pop(key, None)


# Model calls the SDK recorded itself, through its own instrumentation of the provider client: their
# span ids, and the ids of the spans they ran inside (the framework span that made the request, then
# its ancestors up to the nearest model-call span). A framework's model-call span around one of them,
# or inside one, is a second record of the same request.
_sdk_model_calls: "OrderedDict[str, None]" = OrderedDict()
_sdk_model_hosts: "OrderedDict[str, None]" = OrderedDict()


def note_sdk_model_call(span_id: Optional[str], parent_span_id: Optional[str]) -> None:
    """Record a model call the SDK's own instrumentation captured, before the span around it ends."""
    with _scopes_lock:
        if span_id:
            _remember(_sdk_model_calls, span_id, None)
        if parent_span_id:
            _remember(_sdk_model_hosts, parent_span_id, None)


# The GenAI semantic conventions' operations (gen_ai.operation.name) that agent frameworks such as
# Strands Agents record as spans of their own. The operation, not the generic gen_ai.* markers every
# one of these spans carries, says what the span is.
_AGENT_OPERATIONS = frozenset({"invoke_agent", "invoke_swarm", "invoke_graph"})
_MODEL_OPERATIONS = frozenset({"chat", "text_completion", "generate_content"})
_TOOL_OPERATIONS = frozenset({"execute_tool"})
# A framework's own structure around its model and tool calls (Strands' event-loop cycle): never either.
_INTERNAL_OPERATIONS = frozenset({"execute_event_loop_cycle"})
_NAMED_OPERATIONS = ("invoke_agent", "execute_tool")
_MESSAGE_EVENTS = {
    "gen_ai.system.message": "system",
    "gen_ai.user.message": "user",
    "gen_ai.assistant.message": "assistant",
    "gen_ai.tool.message": "tool",
}
_DETAILS_EVENT = "gen_ai.client.inference.operation.details"
_TEXT_MAX = 32768


class _FrameworkSpan(NamedTuple):
    span_type: str
    operation: Optional[str] = None
    agent_name: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    attributes: Dict[str, Any] = {}
    failed: bool = False


def _genai_operation(attrs: Dict[str, Any], name: str) -> Optional[str]:
    declared = attrs.get("gen_ai.operation.name")
    if isinstance(declared, str) and declared.strip():
        return declared.strip().lower()
    if not any(k.startswith("gen_ai.") for k in attrs):
        return None
    lowered = (name or "").strip().lower()
    for op in _NAMED_OPERATIONS:
        if lowered == op or lowered.startswith(op + " "):
            return op
    return lowered if lowered in _INTERNAL_OPERATIONS else None


def _name_suffix(name: str, op: str) -> Optional[str]:
    text = (name or "").strip()
    return text[len(op):].strip() or None if text.lower().startswith(op + " ") else None


def _event_list(span: Any) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    for ev in (getattr(span, "events", None) or ()):
        try:
            out.append((_as_str(getattr(ev, "name", None)) or "", dict(getattr(ev, "attributes", None) or {})))
        except Exception:
            continue
    return out


def _convention_messages(raw: Any) -> List[Tuple[str, Any]]:
    value = parse_json_text(raw)
    if not isinstance(value, list):
        return []
    return [
        (str(m.get("role") or "user"), m.get("parts", m.get("content", "")))
        for m in value if isinstance(m, dict)
    ]


def _input_messages(attrs: Dict[str, Any], events: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Any]]:
    """The span's input messages: one gen_ai.<role>.message event each (the convention before 1.37 and
    Strands' default), else the gen_ai.input.messages list on the details event or the span."""
    out: List[Tuple[str, Any]] = []
    for name, ea in events:
        if name in _MESSAGE_EVENTS:
            out.append((_MESSAGE_EVENTS[name], ea.get("content", ea.get("message", ""))))
        elif name == _DETAILS_EVENT:
            out.extend(_convention_messages(ea.get("gen_ai.input.messages")))
    return out or _convention_messages(attrs.get("gen_ai.input.messages"))


def _output_content(attrs: Dict[str, Any], events: List[Tuple[str, Dict[str, Any]]]) -> Any:
    for name, ea in events:
        if name == "gen_ai.choice":
            return ea.get("message", ea.get("content"))
        if name == _DETAILS_EVENT and ea.get("gen_ai.output.messages") is not None:
            return [part for _, content in _convention_messages(ea["gen_ai.output.messages"]) for part in _as_list(content)]
    messages = _convention_messages(attrs.get("gen_ai.output.messages"))
    return [part for _, content in messages for part in _as_list(content)] if messages else None


def _as_list(content: Any) -> List[Any]:
    value = parse_json_text(content)
    return value if isinstance(value, list) else [value]


def _text_of(content: Any) -> str:
    return message_content_text(parse_json_text(content)).strip()


def _tool_result_text(raw: Any) -> Optional[str]:
    """A tool result's content blocks as the text they hold; JSON the tool returned stays JSON."""
    value = parse_json_text(raw)
    if not isinstance(value, list):
        return _as_str(raw) if raw not in (None, "") else None
    parts: List[str] = []
    for block in value:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif isinstance(block, dict) and "json" in block:
            parts.append(compact_json(block["json"]))
    return "\n".join(p for p in parts if p) or _as_str(raw)


def _clip(text: Optional[str]) -> Optional[str]:
    return text[:_TEXT_MAX] if text else None


def _is_duplicate_model_call(span_id: str, parent_span_id: Optional[str], started_inside: Optional[str]) -> bool:
    with _scopes_lock:
        if span_id in _sdk_model_hosts:
            return True
        return any(sid in _sdk_model_calls for sid in (parent_span_id, started_inside) if sid)


def _framework_span(
    span: Any, attrs: Dict[str, Any], span_id: str, parent_span_id: Optional[str], started_inside: Optional[str]
) -> Optional[_FrameworkSpan]:
    """What a GenAI-convention framework span is, by its operation; None for any other span."""
    op = _genai_operation(attrs, span.name)
    hosts_sdk_call = False
    with _scopes_lock:
        hosts_sdk_call = span_id in _sdk_model_hosts
    if hosts_sdk_call and parent_span_id and op not in _MODEL_OPERATIONS:
        # A span between the framework's model call and the SDK's (an HTTP span, a retry wrapper) passes
        # the mark on, so the model-call span above it still sees the SDK's record of the request.
        with _scopes_lock:
            _remember(_sdk_model_hosts, parent_span_id, None)
    if op is None:
        return None
    events = _event_list(span)
    if op in _AGENT_OPERATIONS:
        users = [t for role, content in _input_messages(attrs, events) if role == "user" for t in [_text_of(content)] if t]
        return _FrameworkSpan(
            span_type="agent",
            agent_name=_as_str(attrs.get("gen_ai.agent.name")) or _name_suffix(span.name, op),
            prompt=_clip(users[-1] if users else None),
            completion=_clip(_text_of(_output_content(attrs, events)) or None),
        )
    if op in _TOOL_OPERATIONS:
        name = _as_str(attrs.get("gen_ai.tool.name")) or _name_suffix(span.name, op)
        arguments = attrs.get("gen_ai.tool.call.arguments")
        result = attrs.get("gen_ai.tool.call.result")
        for event, ea in events:
            if arguments is None and event == "gen_ai.tool.message":
                arguments = ea.get("content")
            if result is None and event == "gen_ai.choice":
                result = _tool_result_text(ea.get("message", ea.get("content")))
        extra: Dict[str, Any] = {}
        if name:
            extra["tool.name"] = name
        if arguments not in (None, ""):
            extra["tool.arguments"] = arguments if isinstance(arguments, str) else compact_json(arguments)
        if result not in (None, ""):
            extra["tool.result"] = result if isinstance(result, str) else compact_json(result)
        status = attrs.get("gen_ai.tool.status")
        return _FrameworkSpan(
            span_type="tool_call",
            operation=name,
            attributes=extra,
            failed=isinstance(status, str) and status.strip().lower() == "error",
        )
    if op in _MODEL_OPERATIONS:
        if not _count_foreign_llm() and _is_duplicate_model_call(span_id, parent_span_id, started_inside):
            return _FrameworkSpan(span_type="internal")
        lines = [f"{role}: {text}" for role, content in _input_messages(attrs, events) for text in [_text_of(content)] if text]
        system = attrs.get("gen_ai.system_instructions")
        if system is not None and not any(line.startswith("system: ") for line in lines):
            lines.insert(0, f"system: {_text_of(system)}")
        return _FrameworkSpan(
            span_type="llm_call",
            prompt=_clip("\n".join(lines) or None),
            completion=_clip(_text_of(_output_content(attrs, events)) or None),
        )
    if op in _INTERNAL_OPERATIONS:
        return _FrameworkSpan(span_type="internal")
    return None


def _readable_to_envelope(span: Any) -> TraceEnvelope:
    ctx = span.get_span_context()
    attrs: Dict[str, Any] = dict(span.attributes or {})
    scope = getattr(getattr(span, "instrumentation_scope", None), "name", "") or ""
    _explicit = attrs.get("evalkit.span_type")
    span_type = str(_explicit) if _explicit else _infer_span_type(attrs, scope)

    _own = "evalkit.span_type" in attrs or "evalkit.sdk_version" in attrs
    _foreign_llm = (
        any(k in attrs for k in _LLM_MARKER_KEYS)
        and not _own
        and not _count_foreign_llm()
    )
    if _foreign_llm:
        span_type = "function_call"
    parent = getattr(span, "parent", None)
    status = "OK"
    status_message = None
    try:
        from opentelemetry.trace.status import StatusCode
        if span.status and span.status.status_code == StatusCode.ERROR:
            status = "ERROR"
            status_message = _as_str(getattr(span.status, "description", None))
    except Exception:
        pass
    svc_name = None
    try:
        res = getattr(span, "resource", None)
        if res is not None:
            svc_name = (res.attributes or {}).get("service.name")
    except Exception:
        pass
    start_iso = _ns_to_iso(getattr(span, "start_time", None))
    end_iso = _ns_to_iso(getattr(span, "end_time", None))
    created_iso = start_iso or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started_user, started_session, started_device, started_agent = _take_start_scope(span)
    started_inside = _take_started_inside(span)
    trace_id = format(ctx.trace_id, "032x")
    span_id = format(ctx.span_id, "016x")
    parent_span_id = format(parent.span_id, "016x") if parent else None
    with _scopes_lock:
        adopted = _adopted_traces.get(ctx.trace_id)
        kept_parent = _elided_spans.get((ctx.trace_id, parent.span_id)) if parent else None
    if kept_parent is not None:
        parent_span_id = format(kept_parent, "016x")
    if adopted is not None:
        trace_id = adopted[0]
        if parent is None and ctx.span_id == adopted[2]:
            parent_span_id = adopted[1]

    operation = span.name
    prompt = _as_str(attrs.get("evalkit.prompt"))
    completion = _as_str(attrs.get("evalkit.completion"))
    agent_name = first_attr(attrs, AGENT_NAME_KEYS)
    framework = None if _own else _framework_span(span, attrs, span_id, parent_span_id, started_inside)
    if framework is not None:
        span_type = framework.span_type
        # A model call the framework made and the SDK did not record is the only record of it, so its
        # usage is the call's. Every other framework span's usage totals calls recorded elsewhere.
        _foreign_llm = span_type != "llm_call"
        operation = framework.operation or operation
        prompt = prompt or framework.prompt
        completion = completion or framework.completion
        agent_name = agent_name or framework.agent_name
        for key, value in framework.attributes.items():
            attrs.setdefault(key, value)
        if framework.failed and status != "ERROR":
            status = "ERROR"
            status_message = status_message or _as_str(attrs.get("tool.result"))
    _cache_read = None if _foreign_llm else _first_token_count(attrs, _CACHE_READ_KEYS)
    _cache_creation = None if _foreign_llm else _first_token_count(attrs, _CACHE_CREATION_KEYS)
    _tokens_in = None if _foreign_llm else _fold_exclusive_cache(
        _first_token_count(attrs, _TOKENS_IN_KEYS), _cache_read, _cache_creation
    )
    return TraceEnvelope(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        span_kind=_span_kind(span),
        span_type=span_type,
        status=status,
        status_message=status_message,
        operation=operation,
        service_name=svc_name,
        model=_as_str(attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model")),
        provider=_as_str(next((attrs[k] for k in _PROVIDER_KEYS if attrs.get(k)), None)),
        prompt=prompt,
        completion=completion,
        tokens_in=_tokens_in,
        tokens_out=None if _foreign_llm else _first_token_count(attrs, _TOKENS_OUT_KEYS),
        cache_read_tokens=_cache_read,
        cache_creation_tokens=_cache_creation,
        latency_ms=_latency_ms(span),
        attributes={k: v for k, v in attrs.items()} or None,
        events=_span_events(span, end_iso or created_iso),
        start_time=start_iso,
        end_time=end_iso,
        created_at=created_iso,
        environment=_as_str(attrs.get("evalkit.environment")),
        app_version=_as_str(attrs.get("evalkit.app_version")),
        user_id=first_attr(attrs, USER_ID_KEYS) or started_user,
        session_id=first_attr(attrs, SESSION_ID_KEYS) or started_session,
        device_id=_as_str(attrs.get("evalkit.device_id")) or started_device,
        agent_name=agent_name or started_agent,
        sdk_version=_as_str(attrs.get("evalkit.sdk_version")),
    )


def _register_all_instrumentors(provider: Any) -> None:
    try:
        from importlib.metadata import entry_points
    except Exception:
        return
    try:
        try:
            eps = list(entry_points(group="opentelemetry_instrumentor"))
        except TypeError:
            eps = list(entry_points().get("opentelemetry_instrumentor", []))
    except Exception:
        return
    for ep in eps:
        try:
            if ep.name in _SKIP_INSTRUMENTORS:
                continue
            ep.load()().instrument(tracer_provider=provider)
        except Exception:
            pass


def setup_otel(client: Any) -> bool:
    global _started, _processor
    if _started:
        return True
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
    except Exception:
        return False

    class _EnvelopeExporter(SpanExporter):
        def export(self, spans) -> "SpanExportResult":
            for sp in spans:
                try:
                    attrs = getattr(sp, "attributes", None) or {}
                    scope_name = getattr(getattr(sp, "instrumentation_scope", None), "name", "") or ""
                    if (
                        attrs.get("asgi.event.type") is not None
                        or scope_name == "gcp.vertex.agent"
                        or _duplicates_own_span(scope_name)
                        or _is_framework_duplicate(sp)
                    ):
                        _take_start_scope(sp)
                        _take_started_inside(sp)
                        continue
                    client.enqueue(_readable_to_envelope(sp))
                except Exception as exc:
                    _log.debug("otel→envelope failed: %s", exc)
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            return None

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True

    try:
        opts = getattr(client, "_opts", None)
        svc = getattr(opts, "service_name", None) or "unknown"
        class _ScopeCapturingProcessor(BatchSpanProcessor):
            def on_start(self, span: Any, parent_context: Any = None) -> None:
                try:
                    if tracing_suppressed():
                        _note_suppressed(span)
                    else:
                        capture_start_scope(span)
                except Exception:
                    pass
                super().on_start(span, parent_context=parent_context)

            def on_end(self, span: Any) -> None:
                try:
                    if _take_suppressed(span):
                        return
                except Exception:
                    pass
                super().on_end(span)

        processor = _ScopeCapturingProcessor(
            _EnvelopeExporter(),
            max_queue_size=getattr(opts, "max_queue_size", None) or 4096,
            max_export_batch_size=getattr(opts, "max_export_batch_size", None) or 512,
            schedule_delay_millis=getattr(opts, "scheduled_delay_ms", None) or 5000,
        )

        existing = trace.get_tracer_provider()
        if isinstance(existing, TracerProvider):
            existing.add_span_processor(processor)
            _processor = processor
            _started = True
            return True

        provider = TracerProvider(resource=Resource.create({"service.name": svc}))
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        if trace.get_tracer_provider() is not provider:
            _log.warning(
                "a foreign TracerProvider owns the global and rejects span processors; "
                "EvalKit will not receive OpenTelemetry spans from it"
            )
            return False
        _processor = processor
        _started = True
        _register_all_instrumentors(provider)
        return True
    except Exception as exc:
        _log.debug("otel setup failed: %s", exc)
        return False


if __name__ == "__main__":
    class _Stub:
        def __init__(self):
            self.got = []
            self._opts = type("O", (), {"service_name": "selfcheck"})()

        def enqueue(self, e):
            self.got.append(e)

    c = _Stub()
    if not setup_otel(c):
        print("otel foundation disabled (install syntropylabs-evalkit[otel]) — import OK, no-op verified")
    else:
        from opentelemetry import trace
        tr = trace.get_tracer("selfcheck")
        with tr.start_as_current_span("db.query") as s:
            s.set_attribute("db.system", "postgresql")
            s.set_attribute("db.statement", "SELECT 1")
        trace.get_tracer_provider().force_flush()
        assert any(e.span_type == "db_query" for e in c.got), c.got
        assert all(len(e.trace_id) == 32 and len(e.span_id) == 16 for e in c.got)
        print("otel foundation self-check OK — spans:", len(c.got))
