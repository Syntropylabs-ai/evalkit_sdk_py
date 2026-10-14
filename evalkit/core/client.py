from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import secrets
import threading
import time
import weakref
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
    _llm_call_owner,
    current_agent,
    current_identity,
    current_span_ctx,
    first_attr,
    get_or_create_ambient_trace_id,
    tracing_suppressed,
)
from .exporter import _Exporter
from .otel_provider import note_sdk_model_call
from .secrets import is_secret_name, scrub_value
from .types import EvalKitOptions, TraceEnvelope
from ..utils.headers import MASK

_log = logging.getLogger("evalkit")

_IO_MAX = 4096
# The longest the flush before the process ends may take, all of it: a receiver that is down or
# unreachable must not hold up a deploy, a worker restart or a reload.
EXIT_FLUSH_BUDGET_S = 3.0
# Long enough to diagnose a failure from the trace list; a stack trace belongs in exception.stacktrace.
MAX_STATUS_MESSAGE = 2000


# How much of an object's str() is scrubbed before it is cut to the attribute limit; the tail
# past the cut is dropped as well, because a secret the cut runs through no longer has the
# shape it is recognised by.
_STR_SCAN_MAX = 64 * 1024
_STR_CUT_MARGIN = 512


def _scrubbed_str(value: Any) -> str:
    text = str(value)
    if len(text) <= _STR_SCAN_MAX:
        return scrub_value(text)
    return scrub_value(text[:_STR_SCAN_MAX])[:-_STR_CUT_MARGIN] + "…"


def _safe_io(value: Any, limit: int = _IO_MAX, scrub: bool = True) -> str:
    import json as _json

    # Secrets are masked in the value itself, before it is serialized and cut to the limit:
    # cut JSON no longer parses, and the export-time scrubber could then only recognise a key by
    # its shape, so an api_key of any other shape went out as written.
    if scrub:
        try:
            value = scrub_value(value)
        except Exception:
            return "<unserializable>"
    try:
        s = _json.dumps(value, default=_scrubbed_str if scrub else str)
    except Exception:
        try:
            s = _scrubbed_str(value) if scrub else repr(value)
        except Exception:
            s = "<unserializable>"
    return s if len(s) <= limit else s[:limit] + "…"


# Automatic function tracing and trace_module skip a callable that carries this marker, so a
# function a decorator already traces is not wrapped, and recorded, a second time. functools.wraps
# copies it onto any decorator stacked on top of the traced function.
TRACED_MARKER = "_evalkit_traced"


def _mark_traced(wrapper: F) -> F:
    try:
        setattr(wrapper, TRACED_MARKER, True)
    except Exception:
        pass
    return wrapper


_RECEIVER_NAMES = ("self", "cls")


class _IOParams:
    """The parameter names a traced callable binds its arguments to, read once when it is wrapped.

    Recorded arguments stay a positional list and a keyword map, the shape the console and the
    trace readers parse, but each value is matched to its parameter so that one named like a
    credential is withheld wherever the caller put it. A function whose own name says it handles
    a credential (get_api_key, presented_token) records neither its arguments nor its result.
    """

    __slots__ = ("secret_positions", "secret_rest", "skip_receiver", "secret_function")

    def __init__(self, fn: Callable[..., Any], name: str) -> None:
        names: List[str] = []
        rest: Optional[str] = None
        try:
            for param in inspect.signature(fn).parameters.values():
                if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                    names.append(param.name)
                elif param.kind is param.VAR_POSITIONAL:
                    rest = param.name
        except Exception:
            # No signature (a builtin, an odd callable): values are still scrubbed by shape and key.
            pass
        self.secret_positions = [is_secret_name(n) for n in names]
        self.secret_rest = bool(rest) and is_secret_name(rest)
        self.skip_receiver = bool(names) and names[0] in _RECEIVER_NAMES
        own_name = getattr(fn, "__name__", "") or ""
        self.secret_function = is_secret_name(own_name) or is_secret_name(name.rsplit(".", 1)[-1])

    def bind(self, args: tuple, kwargs: dict, scrub: bool) -> Tuple[List[Any], Dict[str, Any]]:
        positional: List[Any] = []
        for index, value in enumerate(args):
            if index == 0 and self.skip_receiver:
                continue
            secret = self.secret_positions[index] if index < len(self.secret_positions) else self.secret_rest
            positional.append(MASK if scrub and secret else value)
        named = {k: MASK if scrub and is_secret_name(str(k)) else v for k, v in kwargs.items()}
        return positional, named


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


def _restart_in_forked_children(client: "EvalKitClient") -> None:
    register = getattr(os, "register_at_fork", None)
    if register is None:
        return
    ref = weakref.ref(client)

    def after_in_child() -> None:
        target = ref()
        if target is None:
            return
        try:
            target._after_fork_in_child()
        except Exception:
            pass

    try:
        register(after_in_child=after_in_child)
    except Exception:
        pass


def flush_on_termination_signals(client: "EvalKitClient") -> None:
    """Send the queue before SIGTERM or SIGINT ends the process by its default action.

    A process killed by a signal's default action runs no atexit handler, so what was queued would
    be lost. The handler already installed is kept and called: one of the application's own (or the
    default SIGINT one, which raises KeyboardInterrupt and so still reaches atexit) is called as it
    was, and only for the default action does the queue go out first, within the exit budget, before
    the signal is raised again with its default action, so the process ends exactly as it would have.
    """
    try:
        import signal
    except Exception:
        return
    if threading.current_thread() is not threading.main_thread():
        return
    for name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            previous = signal.getsignal(sig)
        except Exception:
            continue
        if previous in (None, signal.SIG_IGN) or getattr(previous, "_evalkit", False):
            continue

        def handler(signum: int, frame: Any, _previous: Any = previous) -> Any:
            if _previous is signal.SIG_DFL:
                try:
                    client.flush_before_exit()
                except Exception:
                    pass
                signal.signal(signum, signal.SIG_DFL)
                signal.raise_signal(signum)
                return None
            return _previous(signum, frame)

        handler._evalkit = True  # type: ignore[attr-defined]
        try:
            signal.signal(sig, handler)
        except Exception:
            pass


class EvalKitClient:
    def __init__(self, opts: EvalKitOptions) -> None:
        import atexit
        self._opts = opts
        self._exporter = _Exporter(opts)
        self._queue: List[TraceEnvelope] = []
        # Re-entrant: a stream or HTTP body finalised by the garbage collector can enqueue its span
        # from inside a section that already holds the lock, on the same thread.
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._flush_now = threading.Event()
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()
        atexit.register(self.flush_before_exit)
        _restart_in_forked_children(self)

    def _after_fork_in_child(self) -> None:
        """Start over in a process forked after init(): gunicorn --preload, multiprocessing's
        fork start method, Celery's prefork pool.

        A fork copies only the thread that called it, so the export thread is gone in the child,
        and a lock another thread held at that moment would stay held for good. The child gets new
        locks, an empty queue (what was queued belongs to the parent, which still exports it), its
        own export thread, and a flush when the child ends, which a multiprocessing child does
        without running atexit.
        """
        self._lock = threading.RLock()
        self._queue = []
        self._stop_event = threading.Event()
        self._flush_now = threading.Event()
        try:
            self._exporter.after_fork_in_child()
        except Exception:
            pass
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()
        register_at_shutdown = getattr(threading, "_register_atexit", None)
        if register_at_shutdown is not None:
            try:
                register_at_shutdown(self.flush_before_exit)
            except Exception:
                pass

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
            if tracing_suppressed():
                return
            owner = _llm_call_owner.get()
            if owner is not None and envelope.span_type == "llm_call" and envelope.sdk_version:
                try:
                    if owner(envelope):
                        return
                except Exception:
                    pass
            if envelope.span_type == "llm_call" and envelope.sdk_version:
                # Recorded while the framework span around the call is still open, so the bridge can tell
                # that span is a second record of this request when it exports it.
                note_sdk_model_call(envelope.span_id, envelope.parent_span_id)
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
            # Swapped rather than copied and cleared, so a span enqueued re-entrantly while the
            # new list is made lands in the batch being exported, not in a list then cleared.
            pending = self._queue
            self._queue = []
        size = max(1, int(self._opts.max_export_batch_size or 1))
        ok = True
        for start in range(0, len(pending), size):
            if self._exporter.export(pending[start:start + size]):
                continue
            ok = False
            # Behind a receiver that does not answer, or refuses the key, every later batch would meet
            # the same failure, and a flush at exit would wait out the retries of each one.
            if getattr(self._exporter, "halted", False):
                left = len(pending) - start - size
                if left > 0:
                    _log.warning("export stopped; dropping the %d queued span(s) after the failed batch", left)
                break
        return ok

    def _flush_bridge(self, timeout_millis: int = 5000) -> None:
        try:
            from .otel_provider import flush_otel
        except Exception:
            return
        flush_otel(timeout_millis)

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

    def flush_before_exit(self, budget_s: float = EXIT_FLUSH_BUDGET_S) -> bool:
        """Send what is queued as the process ends, within budget_s seconds in all.

        What the receiver has not taken by then is dropped rather than waited for.
        """
        deadline = time.monotonic() + max(0.0, budget_s)
        try:
            self._exporter.exit_deadline = deadline
        except Exception:
            pass
        try:
            self._flush_stream_bodies()
            self._flush_bridge(timeout_millis=max(1, int((deadline - time.monotonic()) * 1000)))
            return self._drain()
        except Exception:
            return False
        finally:
            try:
                self._exporter.exit_deadline = None
            except Exception:
                pass

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
        span_id: Optional[str] = None,
    ) -> Tuple[str, Callable[..., None], "SpanContext"]:
        """Open the server span of one incoming request; `end` closes it.

        `span_id` reuses the id of a request span the framework already opened, so what the framework
        records under it stays in this trace. `end(route=...)` names the span by the framework's route
        template (`GET /orders/{order_id}`), which a raw path with ids in it cannot be grouped by.
        """
        import json

        trace_id = upstream_trace_id or _new_trace_id()
        span_id = span_id or _new_span_id()
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
            route: Optional[str] = None,
        ) -> None:
            end_iso = _now_iso()
            latency = int(time.time() * 1000 - start_ms)
            attrs: Dict[str, Any] = {
                "http.method": method,
                "http.url": url,
            }
            if route:
                attrs["http.route"] = route
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
                span_kind="SERVER",
                span_type="http_call",
                status=status,
                operation=f"{method} {route}" if route else name,
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

    def _records_io(self, capture_io: bool, params: "_IOParams") -> bool:
        return capture_io and (self._opts.capture_secrets or not params.secret_function)

    def trace_function(
        self,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        capture_io: bool = True,
    ) -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            span_name = name or fn.__qualname__
            base_attrs = {"evalkit.span_type": "function_call", **(attributes or {})}
            params = _IOParams(fn, span_name)

            def _mk_attrs(args: tuple, kwargs: dict) -> Dict[str, Any]:
                a = dict(base_attrs)
                if self._records_io(capture_io, params):
                    scrub = not self._opts.capture_secrets
                    positional, named = params.bind(args, kwargs, scrub)
                    if positional:
                        a["function.args"] = _safe_io(positional, scrub=scrub)
                    if named:
                        a["function.kwargs"] = _safe_io(named, scrub=scrub)
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
                    except Exception as exc:
                        try:
                            attrs.update(_exc_attrs(exc))
                            end("ERROR")
                        except Exception:
                            pass
                        raise
                    else:
                        # Recording the result must never turn the call's success into an error.
                        try:
                            if result is not None and self._records_io(capture_io, params):
                                attrs["function.result"] = _safe_io(result, scrub=not self._opts.capture_secrets)
                            end("OK")
                        except Exception:
                            pass
                        return result
                    finally:
                        try:
                            _current_span.reset(token)
                        except Exception:
                            pass
                return _mark_traced(async_wrapper)
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
                    except Exception as exc:
                        try:
                            attrs.update(_exc_attrs(exc))
                            end("ERROR")
                        except Exception:
                            pass
                        raise
                    else:
                        # Recording the result must never turn the call's success into an error.
                        try:
                            if result is not None and self._records_io(capture_io, params):
                                attrs["function.result"] = _safe_io(result, scrub=not self._opts.capture_secrets)
                            end("OK")
                        except Exception:
                            pass
                        return result
                    finally:
                        try:
                            _current_span.reset(token)
                        except Exception:
                            pass
                return _mark_traced(sync_wrapper)

        return decorator

    def trace_tool(
        self,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        capture_io: bool = True,
    ) -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            tool_name = name or fn.__qualname__
            params = _IOParams(fn, tool_name)

            def _mk_attrs(args: tuple, kwargs: dict) -> Dict[str, Any]:
                a: Dict[str, Any] = {
                    "evalkit.span_type": "tool_call",
                    "tool.name": tool_name,
                    "gen_ai.tool.name": tool_name,
                    **(attributes or {}),
                }
                if self._records_io(capture_io, params):
                    scrub = not self._opts.capture_secrets
                    positional, named = params.bind(args, kwargs, scrub)
                    if positional and named:
                        a["tool.arguments"] = _safe_io({"args": positional, "kwargs": named}, scrub=scrub)
                    elif named:
                        a["tool.arguments"] = _safe_io(named, scrub=scrub)
                    elif positional:
                        a["tool.arguments"] = _safe_io(
                            positional[0] if len(positional) == 1 else positional, scrub=scrub,
                        )
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
                    except Exception as exc:
                        try:
                            attrs.update(_exc_attrs(exc))
                            end("ERROR")
                        except Exception:
                            pass
                        raise
                    else:
                        # Recording the result must never turn the call's success into an error.
                        try:
                            if result is not None and self._records_io(capture_io, params):
                                attrs["tool.result"] = _safe_io(result, scrub=not self._opts.capture_secrets)
                            end("OK")
                        except Exception:
                            pass
                        return result
                    finally:
                        try:
                            _current_span.reset(token)
                        except Exception:
                            pass
                return _mark_traced(async_wrapper)
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
                    except Exception as exc:
                        try:
                            attrs.update(_exc_attrs(exc))
                            end("ERROR")
                        except Exception:
                            pass
                        raise
                    else:
                        # Recording the result must never turn the call's success into an error.
                        try:
                            if result is not None and self._records_io(capture_io, params):
                                attrs["tool.result"] = _safe_io(result, scrub=not self._opts.capture_secrets)
                            end("OK")
                        except Exception:
                            pass
                        return result
                    finally:
                        try:
                            _current_span.reset(token)
                        except Exception:
                            pass
                return _mark_traced(sync_wrapper)

        return decorator

    @contextmanager
    def with_trace(self, ctx: SpanContext) -> Generator[SpanContext, None, None]:
        token = _current_span.set(ctx)
        try:
            yield ctx
        finally:
            _current_span.reset(token)
