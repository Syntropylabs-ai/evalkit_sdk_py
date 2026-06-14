from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass(frozen=True)
class SpanContext:
    trace_id: str            # 32 hex chars (W3C traceparent)
    span_id: str             # 16 hex chars
    session_id: Optional[str] = None  # groups all spans in one conversation
    device_id: Optional[str] = None   # identifies the originating client/device


_current_span: ContextVar[Optional[SpanContext]] = ContextVar(
    "evalkit_span", default=None
)

# True while an HTTP instrumentation (httpx/aiohttp/requests) is already tracing a
# call on this execution context. Some HTTP clients delegate to another under the
# hood — e.g. litellm routes the OpenAI SDK's httpx client through a custom aiohttp
# transport — which would otherwise emit TWO http_call spans for ONE network call.
# The outermost HTTP wrapper sets this; nested wrappers see it and skip emitting.
_in_http_span: ContextVar[bool] = ContextVar("evalkit_in_http_span", default=False)


@dataclass
class SimulationContext:
    simulation_id: str
    run_id: str
    scenario_id: str
    scenario_name: Optional[str] = None
    persona: Optional[str] = None
    turn: int = 0
    tags: Optional[List[str]] = None
    collector: List[Any] = field(default_factory=list)


_current_simulation: ContextVar[Optional[SimulationContext]] = ContextVar(
    "evalkit_simulation", default=None
)

# Ambient trace ID: auto-created the first time a log/span is emitted in a
# context that has no explicit trace (e.g. background threads, cron jobs,
# standalone function calls). All logs/spans in the same thread / async-task
# share one ambient trace_id so they're grouped in the trace dashboard.
_ambient_trace_id: ContextVar[Optional[str]] = ContextVar(
    "evalkit_ambient_trace_id", default=None
)


def get_or_create_ambient_trace_id() -> str:
    """Return the ambient trace_id for this execution context, creating one if needed."""
    from ..core.client import _new_trace_id  # local import avoids cycle
    tid = _ambient_trace_id.get()
    if tid is None:
        tid = _new_trace_id()
        _ambient_trace_id.set(tid)
    return tid


def get_or_create_ambient_ctx() -> SpanContext:
    """Return a SpanContext using the ambient trace_id (no parent — acts as a root span)."""
    from ..core.client import _new_span_id  # local import avoids cycle
    return SpanContext(trace_id=get_or_create_ambient_trace_id(), span_id=_new_span_id())
