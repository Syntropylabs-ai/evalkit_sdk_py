from __future__ import annotations

import asyncio
import functools
import logging
import secrets
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

from .._version import SDK_VERSION
from .context import SpanContext, _ambient_trace_id, _current_span, get_or_create_ambient_trace_id
from .exporter import _Exporter
from .types import EvalKitOptions, TraceEnvelope

_log = logging.getLogger("evalkit")

_IO_MAX = 4096


def _safe_io(value: Any, limit: int = _IO_MAX) -> str:
    """Serialize a function argument / return value for a span attribute. Never raises
    and never lets a value (large or unserializable) break tracing."""
    import json as _json

    try:
        s = _json.dumps(value, default=str)
    except Exception:
        try:
            s = repr(value)
        except Exception:
            s = "<unserializable>"
    return s if len(s) <= limit else s[:limit] + "…"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_trace_id() -> str:
    return secrets.token_hex(16)


def _new_span_id() -> str:
    return secrets.token_hex(8)


def _exc_attrs(exc: BaseException) -> Dict[str, Any]:
    """Return standard APM exception attributes including full stack trace.

    Stored as ``exception.*`` keys (OpenTelemetry semantic convention) so the
    dashboard can display file + line info for any error span.
    """
    import traceback as _tb
    attrs = {
        "exception.type": type(exc).__qualname__,
        "exception.message": str(exc),
        "exception.stacktrace": _tb.format_exc(),
    }
    try:
        frames = _tb.extract_tb(exc.__traceback__)
        frame = next((f for f in reversed(frames)
                      if "site-packages" not in f.filename and "/evalkit/" not in f.filename), None) \
            or (frames[-1] if frames else None)
        if frame is not None:
            attrs["error.source"] = f"{frame.filename}:{frame.lineno}"
    except Exception:
        pass
    return attrs


class EvalKitClient:
    def __init__(self, opts: EvalKitOptions) -> None:
        import atexit
        self._opts = opts
        self._exporter = _Exporter(opts)
        self._queue: List[TraceEnvelope] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._flush_now = threading.Event()
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()
        atexit.register(self.flush)

    def _meta_attrs(self) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {"evalkit.sdk_version": SDK_VERSION}
        if self._opts.environment:
            attrs["evalkit.environment"] = self._opts.environment
        if self._opts.app_version:
            attrs["evalkit.app_version"] = self._opts.app_version
        if self._opts.device_id:
            attrs["evalkit.device_id"] = self._opts.device_id
        return attrs

    def _apply_simulation(self, envelope: TraceEnvelope) -> None:
        from .context import _current_simulation
        sim = _current_simulation.get()
        if sim is None:
            return
        envelope.is_simulation = True
        if envelope.simulation_id is None:
            envelope.simulation_id = sim.simulation_id
        if envelope.run_id is None:
            envelope.run_id = sim.run_id
        if envelope.scenario_id is None:
            envelope.scenario_id = sim.scenario_id
        attrs = envelope.attributes or {}
        attrs.setdefault("evalkit.turn", sim.turn)
        if sim.scenario_name:
            attrs.setdefault("evalkit.scenario_name", sim.scenario_name)
        if sim.persona:
            attrs.setdefault("evalkit.persona", sim.persona)
        if sim.tags:
            attrs.setdefault("evalkit.tags", list(sim.tags))
        envelope.attributes = attrs
        if envelope.span_type in ("tool_call", "llm_call"):
            sim.collector.append(envelope)

    def _maybe_emit_tool_spans(self, envelope: TraceEnvelope) -> None:
        """Promote a tool call carried as a gen_ai.tool.call event on the llm_call
        span into a first-class child tool_call span — but ONLY when the SDK has the
        tool's result (server-side tools like OpenAI Responses web_search, where the
        result is in the LLM response).

        For client-side function tools the SDK only ever sees the model's *request*
        (name + arguments); the result is produced by the application after the call.
        Emitting a child span for those would be a zero-duration, output-less row, so
        we keep just the gen_ai.tool.call event on the llm_call span instead. To trace
        a client-side tool's execution (output + latency), wrap it with
        evalkit.trace_function. Simulation tool metrics still work: _tools_from_collector
        falls back to the llm_call span's tool events when no tool_call span exists."""
        if envelope.span_type != "llm_call" or not envelope.events:
            return
        ts = envelope.end_time or envelope.created_at
        for ev in envelope.events:
            attrs = getattr(ev, "attributes", None) or {}
            name = attrs.get("gen_ai.tool.name")
            if not name:
                continue
            result = attrs.get("gen_ai.tool.call.result")
            if result is None:
                # Client-side tool: no result available — keep the event only.
                continue
            tool_attrs = {
                "tool.name": name,
                "tool.arguments": attrs.get("gen_ai.tool.call.arguments"),
                "gen_ai.tool.name": name,
                "gen_ai.tool.call.id": attrs.get("gen_ai.tool.call.id"),
                "tool.result": attrs.get("gen_ai.tool.call.result"),
            }
            tool_attrs = {k: v for k, v in tool_attrs.items() if v is not None}
            self.enqueue(TraceEnvelope(
                trace_id=envelope.trace_id,
                span_id=_new_span_id(),
                parent_span_id=envelope.span_id,
                span_type="tool_call",
                status="OK",
                operation=str(name),
                service_name=envelope.service_name,
                attributes=tool_attrs,
                latency_ms=0,
                start_time=ts,
                end_time=ts,
                created_at=envelope.created_at,
                environment=envelope.environment,
                app_version=envelope.app_version,
                session_id=envelope.session_id,
                device_id=envelope.device_id,
                sdk_version=envelope.sdk_version,
            ))

    def enqueue(self, envelope: TraceEnvelope) -> None:
        # Never blocks and never raises into the caller's code path. Export happens
        # only on the daemon flush thread; here we just buffer and (when a batch is
        # ready) signal that thread to drain promptly.
        try:
            self._apply_simulation(envelope)
            try:
                self._maybe_emit_tool_spans(envelope)
            except Exception:
                pass
            full = False
            with self._lock:
                if len(self._queue) >= self._opts.max_queue_size:
                    self._queue.pop(0)
                self._queue.append(envelope)
                full = len(self._queue) >= self._opts.max_export_batch_size
            if full:
                self._flush_now.set()
        except Exception as exc:  # tracing must never break the host application
            _log.debug("enqueue failed: %s", exc)

    def _flush_loop(self) -> None:
        interval = self._opts.scheduled_delay_ms / 1000
        while not self._stop_event.is_set():
            self._flush_now.wait(interval)
            self._flush_now.clear()
            try:
                self._drain()
            except Exception as exc:
                _log.debug("flush error: %s", exc)

    def _drain(self) -> None:
        with self._lock:
            if not self._queue:
                return
            batch = self._queue[:]
            self._queue.clear()
        self._exporter.export(batch)

    def flush(self) -> None:
        self._drain()

    def destroy(self) -> None:
        self._stop_event.set()
        self._flush_now.set()
        self._drain()

    def start_trace(
        self,
        name: str,
        attributes: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
        device_id: Optional[str] = None,
    ) -> Tuple[str, Callable[[str], None], SpanContext]:
        # Join existing trace as a child span if one is active (e.g. tool called within request).
        # With no active span, fall back to the context's ambient trace so sibling spans in the
        # same execution context share ONE trace instead of each minting a new id.
        parent = _current_span.get()
        trace_id = parent.trace_id if parent else get_or_create_ambient_trace_id()
        _ambient_trace_id.set(trace_id)
        parent_span_id = parent.span_id if parent else None
        resolved_session_id = (parent.session_id if parent else None) or session_id
        resolved_device_id = device_id or (parent.device_id if parent else None) or self._opts.device_id
        span_id = _new_span_id()
        start_ms = time.time() * 1000
        start_iso = _now_iso()
        ctx = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            session_id=resolved_session_id,
            device_id=resolved_device_id,
        )
        # Set ContextVar so nested LLM/HTTP calls become children of this span
        token = _current_span.set(ctx)

        def end(status: str = "OK") -> None:
            end_iso = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            attrs = self._meta_attrs()
            if attributes:
                attrs.update(attributes)
            envelope = TraceEnvelope(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                span_type="log",
                status=status,  # type: ignore[arg-type]
                operation=name,
                service_name=self._opts.service_name,
                attributes=attrs or None,
                latency_ms=latency,
                start_time=start_iso,
                end_time=end_iso,
                created_at=start_iso,
                environment=self._opts.environment,
                app_version=self._opts.app_version,
                session_id=resolved_session_id,
                device_id=resolved_device_id,
                sdk_version=SDK_VERSION,
            )
            self.enqueue(envelope)
            try:
                _current_span.reset(token)
            except Exception:
                pass

        return trace_id, end, ctx

    def start_http_trace(
        self,
        name: str,
        *,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        request_body: Any = None,
        attributes: Optional[Dict[str, str]] = None,
        upstream_trace_id: Optional[str] = None,
        upstream_span_id: Optional[str] = None,
    ) -> Tuple[str, Callable[..., None], "SpanContext"]:
        import json

        trace_id = upstream_trace_id or _new_trace_id()
        span_id = _new_span_id()
        # Pin ambient trace_id so orphaned spans within this request join its trace.
        _ambient_trace_id.set(trace_id)
        start_ms = time.time() * 1000
        start_iso = _now_iso()
        ctx = SpanContext(trace_id=trace_id, span_id=span_id)

        def end(
            status: str = "OK",
            *,
            status_code: Optional[int] = None,
            response_body: Any = None,
            response_headers: Optional[Dict[str, str]] = None,
        ) -> None:
            end_iso = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            attrs: Dict[str, Any] = {
                "http.method": method,
                "http.url": url,
            }
            if headers:
                attrs["http.request.headers"] = json.dumps(headers)
            if request_body is not None:
                attrs["http.request.body"] = json.dumps(request_body) if not isinstance(request_body, str) else request_body
            if status_code is not None:
                attrs["http.status_code"] = status_code
            if response_body is not None:
                attrs["http.response.body"] = json.dumps(response_body) if not isinstance(response_body, str) else response_body
            if response_headers:
                attrs["http.response.headers"] = json.dumps(response_headers)
            attrs.update(self._meta_attrs())
            if attributes:
                attrs.update(attributes)
            envelope = TraceEnvelope(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=upstream_span_id,
                span_type="http_call",
                status=status,  # type: ignore[arg-type]
                operation=name,
                service_name=self._opts.service_name,
                attributes=attrs,
                latency_ms=latency,
                start_time=start_iso,
                end_time=end_iso,
                created_at=start_iso,
                environment=self._opts.environment,
                app_version=self._opts.app_version,
                sdk_version=SDK_VERSION,
            )
            self.enqueue(envelope)

        return trace_id, end, ctx

    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        parent_ctx: Optional[SpanContext] = None,
    ) -> Tuple[Callable[[str], None], SpanContext]:
        # No explicit parent and no active span → join the context's ambient trace so
        # orphan custom spans (tool calls, helpers) don't each start a separate trace.
        parent = parent_ctx or _current_span.get()
        trace_id = parent.trace_id if parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = parent.span_id if parent else None
        session_id = parent.session_id if parent else None
        device_id = (parent.device_id if parent else None) or self._opts.device_id
        start_ms = time.time() * 1000
        start_iso = _now_iso()
        ctx = SpanContext(trace_id=trace_id, span_id=span_id, session_id=session_id, device_id=device_id)

        def end(status: str = "OK") -> None:
            end_iso = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            attrs = self._meta_attrs()
            if attributes:
                attrs.update(attributes)
            span_type = str(attrs.pop("evalkit.span_type", "function_call"))
            envelope = TraceEnvelope(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                span_type=span_type,  # type: ignore[arg-type]
                status=status,  # type: ignore[arg-type]
                operation=name,
                service_name=self._opts.service_name,
                attributes=attrs or None,
                latency_ms=latency,
                start_time=start_iso,
                end_time=end_iso,
                created_at=start_iso,
                environment=self._opts.environment,
                app_version=self._opts.app_version,
                session_id=session_id,
                device_id=device_id,
                sdk_version=SDK_VERSION,
            )
            self.enqueue(envelope)

        return end, ctx

    def trace_function(
        self,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        capture_io: bool = True,
    ) -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            span_name = name or fn.__qualname__
            base_attrs = {"evalkit.span_type": "function_call", **(attributes or {})}

            def _mk_attrs(args: tuple, kwargs: dict) -> Dict[str, Any]:
                a = dict(base_attrs)
                if capture_io:
                    if args:
                        a["function.args"] = _safe_io(list(args))
                    if kwargs:
                        a["function.kwargs"] = _safe_io(kwargs)
                return a

            if asyncio.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    try:
                        attrs = _mk_attrs(args, kwargs)
                        end, ctx = self.start_span(span_name, attrs)
                        token = _current_span.set(ctx)
                    except Exception:
                        return await fn(*args, **kwargs)  # tracing must never break the host
                    try:
                        result = await fn(*args, **kwargs)
                        if capture_io and result is not None:
                            attrs["function.result"] = _safe_io(result)
                        end("OK")
                        return result
                    except Exception as exc:
                        try:
                            attrs.update(_exc_attrs(exc))
                            end("ERROR")
                        except Exception:
                            pass
                        raise
                    finally:
                        try:
                            _current_span.reset(token)
                        except Exception:
                            pass
                return async_wrapper  # type: ignore[return-value]
            else:
                @functools.wraps(fn)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    try:
                        attrs = _mk_attrs(args, kwargs)
                        end, ctx = self.start_span(span_name, attrs)
                        token = _current_span.set(ctx)
                    except Exception:
                        return fn(*args, **kwargs)  # tracing must never break the host
                    try:
                        result = fn(*args, **kwargs)
                        if capture_io and result is not None:
                            attrs["function.result"] = _safe_io(result)
                        end("OK")
                        return result
                    except Exception as exc:
                        try:
                            attrs.update(_exc_attrs(exc))
                            end("ERROR")
                        except Exception:
                            pass
                        raise
                    finally:
                        try:
                            _current_span.reset(token)
                        except Exception:
                            pass
                return sync_wrapper  # type: ignore[return-value]

        return decorator

    def trace_tool(
        self,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        capture_io: bool = True,
    ) -> Callable[[F], F]:
        """Like ``trace_function`` but emits a first-class ``tool_call`` span — it
        renders in the trace UI's Input/Output panels and counts toward tool
        metrics. Use it on your own client-side tools so they behave exactly like
        server-side / LangChain tools. Decorator; works on sync or async functions."""
        def decorator(fn: F) -> F:
            tool_name = name or fn.__qualname__

            def _mk_attrs(args: tuple, kwargs: dict) -> Dict[str, Any]:
                a: Dict[str, Any] = {
                    "evalkit.span_type": "tool_call",
                    "tool.name": tool_name,
                    "gen_ai.tool.name": tool_name,
                    **(attributes or {}),
                }
                if capture_io:
                    if args and kwargs:
                        a["tool.arguments"] = _safe_io({"args": list(args), "kwargs": kwargs})
                    elif kwargs:
                        a["tool.arguments"] = _safe_io(kwargs)
                    elif args:
                        a["tool.arguments"] = _safe_io(args[0] if len(args) == 1 else list(args))
                return a

            if asyncio.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    try:
                        attrs = _mk_attrs(args, kwargs)
                        end, ctx = self.start_span(tool_name, attrs)
                        token = _current_span.set(ctx)
                    except Exception:
                        return await fn(*args, **kwargs)  # tracing must never break the host
                    try:
                        result = await fn(*args, **kwargs)
                        if capture_io and result is not None:
                            attrs["tool.result"] = _safe_io(result)
                        end("OK")
                        return result
                    except Exception as exc:
                        try:
                            attrs.update(_exc_attrs(exc))
                            end("ERROR")
                        except Exception:
                            pass
                        raise
                    finally:
                        try:
                            _current_span.reset(token)
                        except Exception:
                            pass
                return async_wrapper  # type: ignore[return-value]
            else:
                @functools.wraps(fn)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    try:
                        attrs = _mk_attrs(args, kwargs)
                        end, ctx = self.start_span(tool_name, attrs)
                        token = _current_span.set(ctx)
                    except Exception:
                        return fn(*args, **kwargs)  # tracing must never break the host
                    try:
                        result = fn(*args, **kwargs)
                        if capture_io and result is not None:
                            attrs["tool.result"] = _safe_io(result)
                        end("OK")
                        return result
                    except Exception as exc:
                        try:
                            attrs.update(_exc_attrs(exc))
                            end("ERROR")
                        except Exception:
                            pass
                        raise
                    finally:
                        try:
                            _current_span.reset(token)
                        except Exception:
                            pass
                return sync_wrapper  # type: ignore[return-value]

        return decorator

    @contextmanager
    def with_trace(self, ctx: SpanContext) -> Generator[SpanContext, None, None]:
        token = _current_span.set(ctx)
        try:
            yield ctx
        finally:
            _current_span.reset(token)
