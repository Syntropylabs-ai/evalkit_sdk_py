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
from .context import (
    AGENT_NAME_KEYS,
    SESSION_ID_KEYS,
    USER_ID_KEYS,
    SpanContext,
    _ambient_trace_id,
    _current_span,
    current_agent,
    current_identity,
    current_span_ctx,
    first_attr,
    get_or_create_ambient_trace_id,
)
from .exporter import _Exporter
from .types import EvalKitOptions, TraceEnvelope

_log = logging.getLogger("evalkit")

_IO_MAX = 4096
# Long enough to diagnose a failure from the trace list; a stack trace belongs in exception.stacktrace.
MAX_STATUS_MESSAGE = 2000


def _safe_io(value: Any, limit: int = _IO_MAX) -> str:
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


def _first(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value:
            return value
    return None


def _stamp_agent(attrs: Dict[str, Any], agent_name: Optional[str]) -> None:
    # Both spellings: our receiver reads either, other OpenTelemetry backends only the gen_ai one.
    if agent_name:
        attrs["evalkit.agent_name"] = agent_name
        attrs["gen_ai.agent.name"] = agent_name


def _exc_attrs(exc: BaseException) -> Dict[str, Any]:
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

    def _apply_identity(self, envelope: TraceEnvelope) -> None:
        # Envelope field > span attribute (any spelling) > enclosing span > ambient scope.
        ctx = current_span_ctx()
        identity = current_identity()
        attrs = envelope.attributes
        if envelope.user_id is None:
            envelope.user_id = _first(
                first_attr(attrs, USER_ID_KEYS), ctx.user_id if ctx else None, identity.user_id
            )
        if envelope.session_id is None:
            envelope.session_id = _first(
                first_attr(attrs, SESSION_ID_KEYS), ctx.session_id if ctx else None, identity.session_id
            )
        if envelope.device_id is None:
            envelope.device_id = (
                (ctx.device_id if ctx else None) or identity.device_id or self._opts.device_id
            )
        if envelope.agent_name is None:
            envelope.agent_name = _first(
                first_attr(attrs, AGENT_NAME_KEYS), ctx.agent_name if ctx else None, current_agent()
            )
        if envelope.agent_name:
            stamped = dict(attrs or {})
            _stamp_agent(stamped, envelope.agent_name)
            envelope.attributes = stamped

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

    def _apply_prompt(self, envelope: TraceEnvelope) -> None:
        if envelope.span_type != "llm_call":
            return
        from .context import _current_prompt
        ref = _current_prompt.get()
        if ref is None:
            return
        attrs = envelope.attributes or {}
        for key, value in ref.attributes().items():
            attrs.setdefault(key, value)
        envelope.attributes = attrs

    def _maybe_emit_tool_spans(self, envelope: TraceEnvelope) -> None:
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
                user_id=envelope.user_id,
                session_id=envelope.session_id,
                device_id=envelope.device_id,
                agent_name=envelope.agent_name,
                sdk_version=envelope.sdk_version,
            ))

    def _redact(self, envelope: TraceEnvelope) -> Optional[TraceEnvelope]:
        from .redact import apply_mask, strip_content
        from .secrets import scrub_secrets

        if not self._opts.capture_content:
            envelope = strip_content(envelope)
        if not self._opts.capture_secrets:
            envelope = scrub_secrets(envelope)
        mask = self._opts.mask
        if mask is None:
            return envelope
        return apply_mask(mask, envelope)

    def enqueue(self, envelope: TraceEnvelope) -> None:
        try:
            self._apply_identity(envelope)
            self._apply_simulation(envelope)
            self._apply_prompt(envelope)
            try:
                self._maybe_emit_tool_spans(envelope)
            except Exception:
                pass
            redacted = self._redact(envelope)
            if redacted is None:
                return
            envelope = redacted
            full = False
            with self._lock:
                if len(self._queue) >= self._opts.max_queue_size:
                    self._queue.pop(0)
                self._queue.append(envelope)
                full = len(self._queue) >= self._opts.max_export_batch_size
            if full:
                self._flush_now.set()
        except Exception as exc:
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

    def _drain(self) -> bool:
        with self._lock:
            if not self._queue:
                return True
            batch = self._queue[:]
            self._queue.clear()
        return self._exporter.export(batch)

    def _flush_bridge(self) -> None:
        try:
            from .otel_provider import flush_otel
        except Exception:
            return
        flush_otel()

    def _flush_stream_bodies(self) -> None:
        try:
            from ..instrumentation.http._stream_body import flush_pending
        except Exception:
            return
        try:
            flush_pending()
        except Exception:
            pass

    def flush(self) -> bool:
        self._flush_stream_bodies()
        self._flush_bridge()
        return self._drain()

    def destroy(self) -> None:
        self._stop_event.set()
        self._flush_now.set()
        self._flush_stream_bodies()
        self._flush_bridge()
        self._drain()

    def _resolve_scope(
        self,
        attributes: Optional[Dict[str, Any]],
        parent: Optional[SpanContext],
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Explicit argument > span attribute (any spelling) > parent span > ambient scope.

        Resolved at span start so children opened inside this span inherit the same ids.
        """
        identity = current_identity()
        resolved_session = _first(
            session_id,
            first_attr(attributes, SESSION_ID_KEYS),
            parent.session_id if parent else None,
            identity.session_id,
        )
        resolved_user = _first(
            user_id,
            first_attr(attributes, USER_ID_KEYS),
            parent.user_id if parent else None,
            identity.user_id,
        )
        resolved_device = _first(
            device_id,
            first_attr(attributes, ("evalkit.device_id",)),
            parent.device_id if parent else None,
            identity.device_id,
            self._opts.device_id,
        )
        resolved_agent = _first(
            agent,
            first_attr(attributes, AGENT_NAME_KEYS),
            parent.agent_name if parent else None,
            current_agent(),
        )
        return resolved_session, resolved_user, resolved_device, resolved_agent

    def start_trace(
        self,
        name: str,
        attributes: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
        device_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent: Optional[str] = None,
        *,
        join_trace_id: Optional[str] = None,
    ) -> Tuple[str, Callable[..., None], SpanContext]:
        parent = current_span_ctx()
        # A root trace is its own unit of work: reusing the context's ambient id would fold
        # every turn of a loop (and so every turn of a session) into one trace. The ambient
        # id follows the trace while it is open and is restored when it ends.
        # `join_trace_id` adds another root to a trace already open, as a simulation run does
        # with one root per scenario.
        trace_id = parent.trace_id if parent else (join_trace_id or _new_trace_id())
        ambient_token = _ambient_trace_id.set(trace_id)
        parent_span_id = parent.span_id if parent else None
        resolved_session_id, resolved_user_id, resolved_device_id, resolved_agent = self._resolve_scope(
            attributes, parent, session_id=session_id, user_id=user_id, device_id=device_id, agent=agent
        )
        span_id = _new_span_id()
        start_ms = time.time() * 1000
        start_iso = _now_iso()
        ctx = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            session_id=resolved_session_id,
            device_id=resolved_device_id,
            user_id=resolved_user_id,
            agent_name=resolved_agent,
        )
        token = _current_span.set(ctx)
        start_attributes = attributes

        def end(
            status: str = "OK",
            *,
            status_message: Optional[str] = None,
            prompt: Optional[str] = None,
            completion: Optional[str] = None,
            attributes: Optional[Dict[str, Any]] = None,
        ) -> None:
            # A root is often the only span a trace list shows, so what it was asked,
            # what it answered and why it failed go on it rather than only on a child.
            end_iso = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            attrs = self._meta_attrs()
            if start_attributes:
                attrs.update(start_attributes)
            if attributes:
                attrs.update(attributes)
            _stamp_agent(attrs, resolved_agent)
            envelope = TraceEnvelope(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                span_type="log",
                status=status,
                status_message=status_message[:MAX_STATUS_MESSAGE] if status_message else None,
                operation=name,
                prompt=prompt,
                completion=completion,
                service_name=self._opts.service_name,
                attributes=attrs or None,
                latency_ms=latency,
                start_time=start_iso,
                end_time=end_iso,
                created_at=start_iso,
                environment=self._opts.environment,
                app_version=self._opts.app_version,
                user_id=resolved_user_id,
                session_id=resolved_session_id,
                device_id=resolved_device_id,
                agent_name=resolved_agent,
                sdk_version=SDK_VERSION,
            )
            self.enqueue(envelope)
            try:
                _current_span.reset(token)
            except Exception:
                pass
            try:
                _ambient_trace_id.reset(ambient_token)
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
        _ambient_trace_id.set(trace_id)
        identity = current_identity()
        start_ms = time.time() * 1000
        start_iso = _now_iso()
        ctx = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            session_id=identity.session_id,
            device_id=identity.device_id or self._opts.device_id,
            user_id=identity.user_id,
            agent_name=current_agent(),
        )

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
                status=status,
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
        parent = parent_ctx or current_span_ctx()
        trace_id = parent.trace_id if parent else get_or_create_ambient_trace_id()
        span_id = _new_span_id()
        parent_span_id = parent.span_id if parent else None
        session_id, user_id, device_id, agent_name = self._resolve_scope(attributes, parent)
        start_ms = time.time() * 1000
        start_iso = _now_iso()
        ctx = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            session_id=session_id,
            device_id=device_id,
            user_id=user_id,
            agent_name=agent_name,
        )

        def end(status: str = "OK") -> None:
            end_iso = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            attrs = self._meta_attrs()
            if attributes:
                attrs.update(attributes)
            _stamp_agent(attrs, agent_name)
            span_type = str(attrs.pop("evalkit.span_type", "function_call"))
            envelope = TraceEnvelope(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                span_type=span_type,
                status=status,
                operation=name,
                service_name=self._opts.service_name,
                attributes=attrs or None,
                latency_ms=latency,
                start_time=start_iso,
                end_time=end_iso,
                created_at=start_iso,
                environment=self._opts.environment,
                app_version=self._opts.app_version,
                user_id=user_id,
                session_id=session_id,
                device_id=device_id,
                agent_name=agent_name,
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
                        return await fn(*args, **kwargs)
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
                return async_wrapper
            else:
                @functools.wraps(fn)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    try:
                        attrs = _mk_attrs(args, kwargs)
                        end, ctx = self.start_span(span_name, attrs)
                        token = _current_span.set(ctx)
                    except Exception:
                        return fn(*args, **kwargs)
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
                return sync_wrapper

        return decorator

    def trace_tool(
        self,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        capture_io: bool = True,
    ) -> Callable[[F], F]:
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
                        return await fn(*args, **kwargs)
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
                return async_wrapper
            else:
                @functools.wraps(fn)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    try:
                        attrs = _mk_attrs(args, kwargs)
                        end, ctx = self.start_span(tool_name, attrs)
                        token = _current_span.set(ctx)
                    except Exception:
                        return fn(*args, **kwargs)
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
                return sync_wrapper

        return decorator

    @contextmanager
    def with_trace(self, ctx: SpanContext) -> Generator[SpanContext, None, None]:
        token = _current_span.set(ctx)
        try:
            yield ctx
        finally:
            _current_span.reset(token)
