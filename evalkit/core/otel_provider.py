from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .context import AGENT_NAME_KEYS, SESSION_ID_KEYS, USER_ID_KEYS, first_attr
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
_scopes_lock = threading.Lock()
# OTel trace id -> (evalkit trace id, evalkit parent span id, OTel root span id). A span from
# another OpenTelemetry library that starts with no OTel parent while an evalkit span is current
# belongs to that evalkit trace; without this it and its children would export as a trace of
# their own, split off from the turn that caused them.
_adopted_traces: "OrderedDict[int, Tuple[str, str, int]]" = OrderedDict()


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
    from .context import _current_span, current_agent, current_identity

    ctx = _current_span.get()
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


def _take_start_scope(span: Any) -> _Scope:
    key = _span_key(span)
    if key is None:
        return (None, None, None, None)
    with _scopes_lock:
        return _started_scopes.pop(key, None) or (None, None, None, None)


def _readable_to_envelope(span: Any) -> TraceEnvelope:
    ctx = span.get_span_context()
    attrs: Dict[str, Any] = dict(span.attributes or {})
    scope = getattr(getattr(span, "instrumentation_scope", None), "name", "") or ""
    _explicit = attrs.get("evalkit.span_type")
    span_type = str(_explicit) if _explicit else _infer_span_type(attrs, scope)

    _foreign_llm = (
        any(k in attrs for k in _LLM_MARKER_KEYS)
        and "evalkit.span_type" not in attrs
        and "evalkit.sdk_version" not in attrs
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
    _cache_read = None if _foreign_llm else _first_token_count(attrs, _CACHE_READ_KEYS)
    _cache_creation = None if _foreign_llm else _first_token_count(attrs, _CACHE_CREATION_KEYS)
    _tokens_in = None if _foreign_llm else _fold_exclusive_cache(
        _first_token_count(attrs, _TOKENS_IN_KEYS), _cache_read, _cache_creation
    )
    start_iso = _ns_to_iso(getattr(span, "start_time", None))
    end_iso = _ns_to_iso(getattr(span, "end_time", None))
    created_iso = start_iso or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started_user, started_session, started_device, started_agent = _take_start_scope(span)
    trace_id = format(ctx.trace_id, "032x")
    parent_span_id = format(parent.span_id, "016x") if parent else None
    with _scopes_lock:
        adopted = _adopted_traces.get(ctx.trace_id)
    if adopted is not None:
        trace_id = adopted[0]
        if parent is None and ctx.span_id == adopted[2]:
            parent_span_id = adopted[1]
    return TraceEnvelope(
        trace_id=trace_id,
        span_id=format(ctx.span_id, "016x"),
        parent_span_id=parent_span_id,
        span_kind=_span_kind(span),
        span_type=span_type,
        status=status,
        status_message=status_message,
        operation=span.name,
        service_name=svc_name,
        model=_as_str(attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model")),
        provider=_as_str(next((attrs[k] for k in _PROVIDER_KEYS if attrs.get(k)), None)),
        prompt=_as_str(attrs.get("evalkit.prompt")),
        completion=_as_str(attrs.get("evalkit.completion")),
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
        agent_name=first_attr(attrs, AGENT_NAME_KEYS) or started_agent,
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
                    ):
                        _take_start_scope(sp)
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
                    capture_start_scope(span)
                except Exception:
                    pass
                super().on_start(span, parent_context=parent_context)

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
