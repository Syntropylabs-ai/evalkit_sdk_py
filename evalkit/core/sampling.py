"""Which traces are kept when the application asks for only a share of them.

The decision is made from the trace id, as OpenTelemetry's TraceIdRatioBased sampler makes it, so every
span of one trace gets the same answer, in this process and in any other service sampling at the same
ratio: a kept trace is kept whole. Model calls, tool calls and failed spans are always kept, because
they are what a trace is read for and what an evaluation scores.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

_log = logging.getLogger("evalkit")

_TRACE_ID_LIMIT = (1 << 64) - 1

ALWAYS_KEPT_SPAN_TYPES = frozenset({"llm_call", "tool_call"})

_ON = ("always_on", "parentbased_always_on")
_OFF = ("always_off", "parentbased_always_off")
_RATIO = ("traceidratio", "parentbased_traceidratio")


def _ratio(value: Any) -> Optional[float]:
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    if rate != rate:  # NaN
        return None
    return min(1.0, max(0.0, rate))


def resolve_sample_rate(explicit: Optional[float]) -> float:
    """The share of traces to keep: the option when given, else OTEL_TRACES_SAMPLER and its argument,
    else every trace."""
    if explicit is not None:
        rate = _ratio(explicit)
        return 1.0 if rate is None else rate
    sampler = (os.environ.get("OTEL_TRACES_SAMPLER") or "").strip().lower()
    if not sampler or sampler in _ON:
        return 1.0
    if sampler in _OFF:
        return 0.0
    if sampler in _RATIO:
        raw = os.environ.get("OTEL_TRACES_SAMPLER_ARG")
        if raw is None or not raw.strip():
            return 1.0
        rate = _ratio(raw)
        if rate is None:
            _log.warning("OTEL_TRACES_SAMPLER_ARG=%r is not a number between 0 and 1; keeping every trace", raw)
            return 1.0
        return rate
    # A sampler EvalKit does not know (jaeger_remote, xray) keeps every trace here.
    return 1.0


class Sampler:
    __slots__ = ("rate", "_bound")

    def __init__(self, rate: float) -> None:
        self.rate = rate
        self._bound = round(rate * (_TRACE_ID_LIMIT + 1))

    @property
    def keeps_everything(self) -> bool:
        return self.rate >= 1.0

    def trace_kept(self, trace_id: Any) -> bool:
        try:
            return (int(str(trace_id), 16) & _TRACE_ID_LIMIT) < self._bound
        except (TypeError, ValueError):
            # An id that is not hex (an application's own) cannot be decided on; it is kept.
            return True

    def keeps(self, envelope: Any) -> bool:
        if self.rate >= 1.0:
            return True
        if getattr(envelope, "span_type", None) in ALWAYS_KEPT_SPAN_TYPES:
            return True
        if getattr(envelope, "status", None) == "ERROR" or getattr(envelope, "exception", None) is not None:
            return True
        return self.trace_kept(getattr(envelope, "trace_id", None))
