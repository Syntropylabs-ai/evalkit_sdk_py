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
from .bounded import RateLimiter
from .errors import FINGERPRINT_ATTR, apply_exception, exc_attributes
from .exporter import _MAX_SPAN_BYTES, _Exporter
from .health import DISABLED, PROBE_SPANS, ExportHealth
from .ignore import ignored_callable
from .limits import cap_envelope_values, estimate_envelope_bytes, fit_envelope
from .otel_provider import note_sdk_model_call
from .sampling import Sampler
from .secrets import is_secret_name, scrub_value
from .serialize import bounded_json
from .types import EvalKitOptions, TraceEnvelope
from .untraced import UNTRACED_MARKER
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
    """A recorded argument or result as JSON, at most ``limit`` characters and ``…``.

    The writer stops at the limit instead of serializing the whole value and then cutting it, so a
    large argument costs the traced call no more than a small one. Secrets are masked as the value is
    walked, before anything is cut: cut JSON no longer parses, and the export-time scrubber could then
    only recognise a key by its shape, so an api_key of any other shape would go out as written.
    """
    try:
        return bounded_json(value, limit, scrub=scrub, fallback=str)
    except Exception:
        return "<unserializable>"


# Automatic function tracing and trace_module skip a callable that carries this marker, so a
# function a decorator already traces is not wrapped, and recorded, a second time. functools.wraps
# copies it onto any decorator stacked on top of the traced function.
TRACED_MARKER = "_evalkit_traced"


def _left_untraced(fn: Any) -> bool:
    """Whether function tracing leaves ``fn`` as it is: it is wrapped in ``evalkit.untraced``, or its
    qualified name matches ignore_functions. Either wins over an explicit decorator."""
    try:
        return bool(getattr(fn, UNTRACED_MARKER, False)) or ignored_callable(fn)
    except Exception:
        return False


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
    """The span attributes recording ``exc``, for a call site that can only hand attributes on.

    A span given the exception itself (``TraceEnvelope(exception=...)`` or ``end(exception=...)``) also
    carries the ``exception`` event; see ``evalkit.core.errors``.
    """
    try:
        return exc_attributes(exc)
    except Exception:
        return {"exception.type": type(exc).__qualname__}


def _is_coroutine_function(fn: Any) -> bool:
    if inspect.iscoroutinefunction(fn):
        return True
    # A function marked the way asyncio.iscoroutinefunction recognises, read without the call that
    # Python 3.14 deprecates.
    marker = getattr(asyncio.coroutines, "_is_coroutine", None)
    return marker is not None and getattr(fn, "_is_coroutine", None) is marker


def release_ambient_trace(trace_id: Optional[str]) -> None:
    """Forget a request's trace id as the context's ambient one once the request is over.

    A WSGI server answers request after request on the same thread, so an id left set would put a
    span recorded later on that thread, outside any request, into the previous request's trace.
    """
    try:
        if trace_id and _ambient_trace_id.get() == trace_id:
            _ambient_trace_id.set(None)
    except Exception:
        pass


def _reset_ambient(token: Any, trace_id: str) -> None:
    try:
        _ambient_trace_id.reset(token)
    except Exception:
        # Ended from another context, or a second time.
        release_ambient_trace(trace_id)


_UNSET: Any = object()


class _NullExporter:
    """The exporter of a client that records nothing."""

    halted = False
    _disabled = True
    exit_deadline: Optional[float] = None

    def export(self, envelopes: List[TraceEnvelope]) -> bool:
        return False

    def after_fork_in_child(self) -> None:
        pass


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
    def __init__(self, opts: EvalKitOptions, *, enabled: bool = True) -> None:
        self._opts = opts
        self._enabled = bool(enabled)
        self._health = ExportHealth()
        rate = getattr(opts, "sample_rate", None)
        self._sampler = Sampler(1.0 if rate is None else float(rate))
        self._queue: List[TraceEnvelope] = []
        self._queue_bytes = 0
        # Re-entrant: a stream or HTTP body finalised by the garbage collector can enqueue its span
        # from inside a section that already holds the lock, on the same thread.
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._flush_now = threading.Event()
        if not self._enabled:
            # Records nothing: no export thread, no exit hook, nothing patched.
            self._exporter: Any = _NullExporter()
            self._thread: Optional[threading.Thread] = None
            return
        self._exporter = _Exporter(opts, health=self._health)
        self._thread = threading.Thread(target=self._flush_loop, daemon=True, name="evalkit-export")
        self._thread.start()
        import atexit
        atexit.register(self.flush_before_exit)
        _restart_in_forked_children(self)

    @property
    def enabled(self) -> bool:
        """Whether this client records spans: False after an invalid configuration or the kill switch."""
        return self._enabled

    def _after_fork_in_child(self) -> None:
        """Start over in a process forked after init(): gunicorn --preload, multiprocessing's
        fork start method, Celery's prefork pool.

        A fork copies only the thread that called it, so the export thread is gone in the child,
        and a lock another thread held at that moment would stay held for good. The child gets new
        locks, an empty queue (what was queued belongs to the parent, which still exports it), its
        own export thread, and a flush when the child ends, which a multiprocessing child does
        without running atexit.
        """
        if not self._enabled:
            return
        self._lock = threading.RLock()
        self._queue = []
        self._queue_bytes = 0
        self._stop_event = threading.Event()
        self._flush_now = threading.Event()
        try:
            self._health.after_fork_in_child()
        except Exception:
            pass
        try:
            self._exporter.after_fork_in_child()
        except Exception:
            pass
        self._thread = threading.Thread(target=self._flush_loop, daemon=True, name="evalkit-export")
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

    def _accepting(self) -> bool:
        if getattr(self._exporter, "_disabled", False):
            return False
        return self._health.circuit.accepting()

    def enqueue(self, envelope: TraceEnvelope) -> None:
        try:
            if not self._enabled or tracing_suppressed():
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
            if not self._sampler.keeps_everything and not self._sampler.keeps(envelope):
                self._health.count("dropped_sampled")
                return
            self._apply_identity(envelope)
            self._apply_simulation(envelope)
            self._apply_prompt(envelope)
            try:
                self._maybe_emit_tool_spans(envelope)
            except Exception:
                pass
            exc = envelope.exception
            if isinstance(exc, BaseException):
                # Let go of the exception (and the frames its traceback holds) before the span waits.
                envelope.exception = None
                apply_exception(
                    envelope, exc, escaped=envelope.exception_escaped is not False, timestamp=envelope.end_time,
                )
            elif exc is not None:
                envelope.exception = None
            if not self._accepting():
                self._health.count(
                    "dropped_export_failed" if getattr(self._exporter, "_disabled", False) else "dropped_circuit_open"
                )
                return
            redacted = self._redact(envelope)
            if redacted is None:
                return
            envelope = redacted
            self._queue_span(envelope, self._fit(envelope))
        except Exception as exc:
            _log.debug("enqueue failed: %s", exc)

    def _fit(self, envelope: TraceEnvelope) -> Optional[int]:
        """Cut the span's values to the per-value limit and the span to the span limit; its size after."""
        try:
            cap_envelope_values(envelope, int(self._opts.max_value_bytes or 0))
            return fit_envelope(envelope, _MAX_SPAN_BYTES)
        except Exception as exc:
            _log.debug("fitting a span failed: %s", exc)
            return None

    def _queue_span(self, envelope: TraceEnvelope, size: Optional[int] = None) -> None:
        if size is None:
            size = estimate_envelope_bytes(envelope)
        try:
            envelope._ek_size = size  # type: ignore[attr-defined]
        except Exception:
            pass
        max_spans = max(1, int(self._opts.max_queue_size or 1))
        max_bytes = int(getattr(self._opts, "max_queue_bytes", 0) or 0)
        full = False
        with self._lock:
            queue = self._queue
            while queue and (len(queue) >= max_spans or (max_bytes > 0 and self._queue_bytes + size > max_bytes)):
                reason = "dropped_queue_full" if len(queue) >= max_spans else "dropped_bytes_budget"
                index = self._victim(queue, envelope)
                self._health.count(reason)
                if index < 0:
                    return
                dropped = queue.pop(index)
                self._queue_bytes = max(0, self._queue_bytes - int(getattr(dropped, "_ek_size", 0) or 0))
            queue.append(envelope)
            self._queue_bytes += size
            full = len(queue) >= self._opts.max_export_batch_size
        if full:
            self._flush_now.set()

    @staticmethod
    def _victim(queue: List[TraceEnvelope], incoming: TraceEnvelope) -> int:
        """Which span gives way when the queue is full: the oldest that is not a model call; the incoming
        span, when it is not one either; else the oldest model call. -1 names the incoming span."""
        for index, queued in enumerate(queue):
            if queued.span_type != "llm_call":
                return index
        return -1 if incoming.span_type != "llm_call" else 0

    def _flush_loop(self) -> None:
        interval = self._opts.scheduled_delay_ms / 1000
        while not self._stop_event.is_set():
            self._flush_now.wait(interval)
            self._flush_now.clear()
            try:
                self._drain()
            except Exception as exc:
                _log.debug("flush error: %s", exc)

    def _export(self, batch: List[TraceEnvelope]) -> bool:
        exporter = self._exporter
        try:
            exporter.last_sent = None
        except Exception:
            pass
        try:
            ok = bool(exporter.export(batch))
        except Exception as exc:
            _log.debug("export raised: %s", exc)
            ok = False
        sent = getattr(exporter, "last_sent", None)
        if not isinstance(sent, int):
            sent = len(batch) if ok else 0
        if sent:
            self._health.export_succeeded(sent)
        if len(batch) > sent:
            self._health.count("dropped_export_failed", len(batch) - sent)
        return ok

    def _drain(self) -> bool:
        with self._lock:
            if not self._queue:
                return True
            # Swapped rather than copied and cleared, so a span enqueued re-entrantly while the
            # new list is made lands in the batch being exported, not in a list then cleared.
            pending = self._queue
            self._queue = []
            self._queue_bytes = 0
        if getattr(self._exporter, "_disabled", False):
            self._health.count("dropped_export_failed", len(pending))
            return False
        circuit = self._health.circuit
        if not circuit.accepting():
            # Queued just before the circuit opened: sending it now would retry into the receiver
            # the cool-down keeps the process away from.
            self._health.count("dropped_circuit_open", len(pending))
            return False
        size = max(1, int(self._opts.max_export_batch_size or 1))
        batches: List[List[TraceEnvelope]] = []
        start = 0
        if circuit.probing:
            # One small batch tells whether the receiver is back before the rest is sent into it.
            probe = min(size, PROBE_SPANS)
            batches.append(pending[:probe])
            start = probe
        batches.extend(pending[i:i + size] for i in range(start, len(pending), size))
        ok = True
        halted = False
        sent_batches = 0
        for index, batch in enumerate(batches):
            if not batch:
                continue
            sent_batches += 1
            if self._export(batch):
                continue
            ok = False
            # Behind a receiver that does not answer, or refuses the key, every later batch would meet
            # the same failure, and a flush at exit would wait out the retries of each one.
            if getattr(self._exporter, "halted", False):
                halted = True
                left = sum(len(b) for b in batches[index + 1:])
                self._health.count("dropped_export_failed", left)
                break
        if sent_batches:
            if halted:
                circuit.cycle_failed()
            else:
                circuit.cycle_succeeded()
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

    def flush(self, timeout: Optional[float] = 5.0) -> bool:
        """Send what is queued now, within ``timeout`` seconds (None: the export's own retry budget).

        Returns whether the receiver took all of it. Never raises: a receiver that is down, a bad
        base_url or an export that runs out of time only makes the answer False.
        """
        if not self._enabled:
            return True
        try:
            deadline = None
            if timeout is not None:
                deadline = time.monotonic() + max(0.0, float(timeout))
            self._flush_stream_bodies()
            bridge_ms = 5000 if deadline is None else max(1, int((deadline - time.monotonic()) * 1000))
            self._flush_bridge(timeout_millis=bridge_ms)
            setter = getattr(self._exporter, "set_call_deadline", None)
            previous = setter(deadline) if setter is not None else None
            try:
                return bool(self._drain())
            finally:
                if setter is not None:
                    setter(previous)
        except Exception as exc:
            _log.debug("flush failed: %s", exc)
            return False

    def flush_before_exit(self, budget_s: float = EXIT_FLUSH_BUDGET_S) -> bool:
        """Send what is queued as the process ends, within budget_s seconds in all.

        What the receiver has not taken by then is dropped rather than waited for.
        """
        if not self._enabled:
            return True
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
        if not self._enabled:
            return
        try:
            self._flush_stream_bodies()
            self._flush_bridge()
            self._drain()
        except Exception as exc:
            _log.debug("destroy failed: %s", exc)

    def stats(self) -> Dict[str, Any]:
        """How recording and exporting are going, for a health check or a debug page."""
        with self._lock:
            queued = len(self._queue)
            queued_bytes = self._queue_bytes
        snap = self._health.snapshot()
        state = snap["circuit_state"]
        if not self._enabled or getattr(self._exporter, "_disabled", False):
            state = DISABLED
        return {
            "queued": queued,
            "dropped_queue_full": snap["dropped_queue_full"],
            "dropped_bytes_budget": snap["dropped_bytes_budget"],
            "dropped_export_failed": snap["dropped_export_failed"],
            "export_failures": snap["export_failures"],
            "last_error": snap["last_error"],
            "circuit_state": state,
            "queued_bytes": queued_bytes,
            "dropped_circuit_open": snap["dropped_circuit_open"],
            "dropped_rate_limited": snap["dropped_rate_limited"],
            "dropped_sampled": snap["dropped_sampled"],
            "exported": snap["exported"],
            "enabled": self._enabled,
        }

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
            exception: Optional[BaseException] = None,
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
                exception=exception,
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
        `end(exception=...)` records the exception that failed the request, and `end(request_body=...)`
        a body read while the request ran. The request's trace id is the ambient one until `end`.
        """
        import json

        trace_id = upstream_trace_id or _new_trace_id()
        span_id = span_id or _new_span_id()
        ambient_token = _ambient_trace_id.set(trace_id)
        identity = current_identity()
        start_ms = time.time() * 1000
        start_iso = _now_iso()
        start_attributes = attributes
        start_body = request_body
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
            exception: Optional[BaseException] = None,
            exception_escaped: Optional[bool] = None,
            request_body: Any = _UNSET,
            attributes: Optional[Dict[str, Any]] = None,
        ) -> None:
            try:
                end_iso = _now_iso()
                latency = int(time.time() * 1000 - start_ms)
                body = start_body if request_body is _UNSET else request_body
                attrs: Dict[str, Any] = {
                    "http.method": method,
                    "http.url": url,
                }
                if route:
                    attrs["http.route"] = route
                if headers:
                    attrs["http.request.headers"] = json.dumps(headers)
                if body is not None:
                    attrs["http.request.body"] = json.dumps(body) if not isinstance(body, str) else body
                if status_code is not None:
                    attrs["http.status_code"] = status_code
                if response_body is not None:
                    attrs["http.response.body"] = (
                        json.dumps(response_body) if not isinstance(response_body, str) else response_body
                    )
                if response_headers:
                    attrs["http.response.headers"] = json.dumps(response_headers)
                attrs.update(self._meta_attrs())
                if start_attributes:
                    attrs.update(start_attributes)
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
                    exception=exception,
                    exception_escaped=exception_escaped,
                )
                self.enqueue(envelope)
            finally:
                _reset_ambient(ambient_token, trace_id)

        return trace_id, end, ctx

    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        parent_ctx: Optional[SpanContext] = None,
    ) -> Tuple[Callable[..., None], SpanContext]:
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

        def end(status: str = "OK", *, exception: Optional[BaseException] = None) -> None:
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
                exception=exception,
            )
            self.enqueue(envelope)

        return end, ctx

    def _records_io(self, capture_io: bool, params: "_IOParams") -> bool:
        return capture_io and (self._opts.capture_secrets or not params.secret_function)

    def _wrap(
        self,
        fn: F,
        span_name: str,
        mk_attrs: Callable[[tuple, dict], Dict[str, Any]],
        result_key: str,
        capture_io: bool,
        params: "_IOParams",
        rate_limit: Optional[float],
        fingerprint: Optional[str],
    ) -> F:
        """``fn`` traced: one span per call, or per iteration of a generator, ended however it ends.

        Every way out ends the span exactly once and is passed on as it was: a return, an exception, and
        also what is not an ``Exception`` (a cancelled task, Ctrl-C, ``sys.exit``), which would otherwise
        leave the span open for good. A cancelled task is recorded as an error with ``evalkit.cancelled``.
        Recording never changes what the call returns or raises.
        """
        client = self
        limiter = RateLimiter(rate_limit) if rate_limit else None

        def begin(args: tuple, kwargs: dict) -> Optional[Tuple[Callable[..., None], SpanContext, Dict[str, Any]]]:
            try:
                if tracing_suppressed():
                    # Nothing would be exported; the call runs as it is, and the rate is not spent on it.
                    return None
                if limiter is not None and not limiter.allow():
                    # A function called in a hot loop records the first calls of each second, not all.
                    client._health.count("dropped_rate_limited")
                    return None
                attrs = mk_attrs(args, kwargs)
                end, ctx = client.start_span(span_name, attrs)
                return end, ctx, attrs
            except Exception as exc:
                _log.debug("starting a span for %s failed: %s", span_name, exc)
                return None

        def finish_ok(opened: Any, result: Any = None, record_result: bool = True) -> None:
            # Recording the result must never turn the call's success into an error.
            try:
                end, _ctx, attrs = opened
                if record_result and result is not None and client._records_io(capture_io, params):
                    attrs[result_key] = _safe_io(result, scrub=not client._opts.capture_secrets)
                end("OK")
            except Exception:
                pass

        def finish_error(opened: Any, exc: BaseException) -> None:
            try:
                end, _ctx, attrs = opened
                if isinstance(exc, GeneratorExit) or (isinstance(exc, SystemExit) and exc.code in (0, None)):
                    end("OK")
                    return
                if fingerprint:
                    attrs[FINGERPRINT_ATTR] = fingerprint
                end("ERROR", exception=exc)
            except Exception:
                pass

        def enter(opened: Any) -> Any:
            return _current_span.set(opened[1]) if opened is not None else None

        def leave(token: Any) -> None:
            if token is not None:
                try:
                    _current_span.reset(token)
                except Exception:
                    pass

        if _is_coroutine_function(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                opened = begin(args, kwargs)
                if opened is None:
                    return await fn(*args, **kwargs)
                token = enter(opened)
                try:
                    result = await fn(*args, **kwargs)
                except BaseException as exc:
                    finish_error(opened, exc)
                    raise
                else:
                    finish_ok(opened, result)
                    return result
                finally:
                    leave(token)
            return _mark_traced(async_wrapper)  # type: ignore[return-value]

        if inspect.isasyncgenfunction(fn):
            @functools.wraps(fn)
            async def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Driven step by step rather than with `async for`, so what the consumer sends, throws or
                # closes reaches the generator, and the span is current only while the generator runs:
                # set across a yield it would be current in the consumer's code too.
                opened = begin(args, kwargs)
                agen = fn(*args, **kwargs)
                to_send: Any = None
                to_throw: Optional[BaseException] = None
                try:
                    while True:
                        token = enter(opened)
                        try:
                            if to_throw is not None:
                                thrown, to_throw = to_throw, None
                                item = await agen.athrow(thrown)
                            else:
                                item = await agen.asend(to_send)
                        finally:
                            leave(token)
                        try:
                            to_send = yield item
                        except GeneratorExit:
                            token = enter(opened)
                            try:
                                await agen.aclose()
                            finally:
                                leave(token)
                            raise
                        except BaseException as exc:
                            to_throw = exc
                except StopAsyncIteration:
                    if opened is not None:
                        finish_ok(opened, record_result=False)
                    return
                except GeneratorExit:
                    if opened is not None:
                        finish_ok(opened, record_result=False)
                    raise
                except BaseException as exc:
                    if opened is not None:
                        finish_error(opened, exc)
                    raise
            return _mark_traced(async_gen_wrapper)  # type: ignore[return-value]

        if inspect.isgeneratorfunction(fn):
            @functools.wraps(fn)
            def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                opened = begin(args, kwargs)
                gen = fn(*args, **kwargs)
                to_send: Any = None
                to_throw: Optional[BaseException] = None
                try:
                    while True:
                        token = enter(opened)
                        try:
                            if to_throw is not None:
                                thrown, to_throw = to_throw, None
                                item = gen.throw(thrown)
                            else:
                                item = gen.send(to_send)
                        finally:
                            leave(token)
                        try:
                            to_send = yield item
                        except GeneratorExit:
                            token = enter(opened)
                            try:
                                gen.close()
                            finally:
                                leave(token)
                            raise
                        except BaseException as exc:
                            to_throw = exc
                except StopIteration as stop:
                    if opened is not None:
                        finish_ok(opened, record_result=False)
                    return stop.value
                except GeneratorExit:
                    if opened is not None:
                        finish_ok(opened, record_result=False)
                    raise
                except BaseException as exc:
                    if opened is not None:
                        finish_error(opened, exc)
                    raise
            return _mark_traced(gen_wrapper)  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            opened = begin(args, kwargs)
            if opened is None:
                return fn(*args, **kwargs)
            token = enter(opened)
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                finish_error(opened, exc)
                raise
            else:
                finish_ok(opened, result)
                return result
            finally:
                leave(token)
        return _mark_traced(sync_wrapper)  # type: ignore[return-value]

    def trace_function(
        self,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        capture_io: bool = True,
        *,
        error_fingerprint: Optional[str] = None,
    ) -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            if not self._enabled or _left_untraced(fn):
                return fn
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

            return self._wrap(
                fn, span_name, _mk_attrs, "function.result", capture_io, params,
                getattr(self._opts, "function_rate_limit", 0), error_fingerprint,
            )

        return decorator

    def trace_tool(
        self,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        capture_io: bool = True,
        *,
        error_fingerprint: Optional[str] = None,
    ) -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            if not self._enabled or _left_untraced(fn):
                return fn
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

            # Tool calls are what an agent's trace is read and scored for, so none is left out by rate.
            return self._wrap(fn, tool_name, _mk_attrs, "tool.result", capture_io, params, None, error_fingerprint)

        return decorator

    @contextmanager
    def with_trace(self, ctx: SpanContext) -> Generator[SpanContext, None, None]:
        token = _current_span.set(ctx)
        try:
            yield ctx
        finally:
            _current_span.reset(token)
