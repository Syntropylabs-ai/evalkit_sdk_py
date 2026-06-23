from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .types import TraceEnvelope

_log = logging.getLogger("evalkit")

_started = False  # set_tracer_provider may only run once per process


def otel_active() -> bool:
    return _started

_SKIP_INSTRUMENTORS = {
    "openai", "anthropic", "google-generativeai", "google-genai", "google_genai",
    "vertexai", "cohere", "mistralai", "groq", "bedrock", "langchain", "litellm",
    "replicate", "transformers",
}


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
    if any(k in attrs for k in ("gen_ai.request.model", "gen_ai.system", "gen_ai.response.model")):
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


def _readable_to_envelope(span: Any) -> TraceEnvelope:
    ctx = span.get_span_context()
    attrs: Dict[str, Any] = dict(span.attributes or {})
    scope = getattr(getattr(span, "instrumentation_scope", None), "name", "") or ""
    _explicit = attrs.get("evalkit.span_type")
    span_type = str(_explicit) if _explicit else _infer_span_type(attrs, scope)

    _foreign_llm = (
        any(k in attrs for k in ("gen_ai.system", "gen_ai.request.model", "gen_ai.response.model",
                                  "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"))
        and "evalkit.span_type" not in attrs
        and "evalkit.sdk_version" not in attrs
        and not _count_foreign_llm()
    )
    if _foreign_llm:
        span_type = "function_call"
    parent = getattr(span, "parent", None)
    status = "OK"
    try:
        from opentelemetry.trace.status import StatusCode
        if span.status and span.status.status_code == StatusCode.ERROR:
            status = "ERROR"
    except Exception:
        pass
    return TraceEnvelope(
        trace_id=format(ctx.trace_id, "032x"),
        span_id=format(ctx.span_id, "016x"),
        parent_span_id=format(parent.span_id, "016x") if parent else None,
        span_type=span_type,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        operation=span.name,
        model=attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model"),
        provider=attrs.get("gen_ai.system"),
        tokens_in=None if _foreign_llm else attrs.get("gen_ai.usage.input_tokens"),
        tokens_out=None if _foreign_llm else attrs.get("gen_ai.usage.output_tokens"),
        latency_ms=_latency_ms(span),
        attributes={k: v for k, v in attrs.items()} or None,
        start_time=_ns_to_iso(getattr(span, "start_time", None)),
        end_time=_ns_to_iso(getattr(span, "end_time", None)),
        created_at=_ns_to_iso(getattr(span, "start_time", None)) or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def _register_all_instrumentors(provider: Any) -> None:
    """Auto-enable EVERY installed OTel community instrumentation (max compatibility),
    except the libraries EvalKit keeps hand-rolled (_SKIP_INSTRUMENTORS). Uses the
    standard `opentelemetry_instrumentor` entry-point group — the same discovery
    mechanism as the `opentelemetry-instrument` CLI. One broken instrumentor never
    breaks the others or the host."""
    try:
        from importlib.metadata import entry_points
    except Exception:
        return
    try:
        try:
            eps = list(entry_points(group="opentelemetry_instrumentor"))  # py3.10+
        except TypeError:
            eps = list(entry_points().get("opentelemetry_instrumentor", []))  # py3.9
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
    """Put the Python SDK on the OpenTelemetry SDK: an OTel TracerProvider whose spans
    are mapped to TraceEnvelope and pushed through the existing client pipeline
    (→ /v1/ingest → dashboard), plus OTel community instrumentation for libraries we
    don't hand-roll. No-op (returns False) unless opentelemetry-sdk is installed
    (pip install "syntropylabs-evalkit[otel]"). Never raises into the host."""
    global _started
    if _started:
        return True
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
    except Exception:
        return False

    class _EnvelopeExporter(SpanExporter):
        def export(self, spans) -> "SpanExportResult":
            for sp in spans:
                try:
                    client.enqueue(_readable_to_envelope(sp))
                except Exception as exc:
                    _log.debug("otel→envelope failed: %s", exc)
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:  # noqa: D401
            return None

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True

    try:
        svc = getattr(getattr(client, "_opts", None), "service_name", None) or "unknown"
        provider = TracerProvider(resource=Resource.create({"service.name": svc}))
        provider.add_span_processor(SimpleSpanProcessor(_EnvelopeExporter()))
        trace.set_tracer_provider(provider)
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
