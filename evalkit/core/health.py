"""How exporting is going: counters, a circuit breaker, and warnings that do not repeat themselves.

A receiver that is down must cost the application nothing: no warning per failed request, no queue
growing behind it, no export thread retrying into it every few seconds. After a few failed export cycles
in a row the circuit opens and spans are dropped as they are recorded, which only counts them; after a
cool-down one small batch probes the receiver, and the circuit closes again once it answers.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

_log = logging.getLogger("evalkit")

FAILURES_TO_OPEN = 3
BASE_COOLDOWN_S = 30.0
MAX_COOLDOWN_S = 300.0
PROBE_SPANS = 16
WARN_INTERVAL_S = 300.0
LOGGED_BODY_CHARS = 200

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"
DISABLED = "disabled"

COUNTERS = (
    "dropped_queue_full",
    "dropped_bytes_budget",
    "dropped_export_failed",
    "dropped_circuit_open",
    "dropped_rate_limited",
    "dropped_sampled",
    "exported",
    "export_failures",
)


def clip(text: Any, limit: int = LOGGED_BODY_CHARS) -> str:
    """A response body or error text short enough for one log line."""
    value = str(text or "").strip().replace("\n", " ")
    return value if len(value) <= limit else value[:limit] + "…"


class Circuit:
    """Closed while exports succeed; open, dropping spans, for a cool-down after repeated failures."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        self.state = CLOSED
        self.failures = 0
        self.cooldown = BASE_COOLDOWN_S
        self.open_until = 0.0

    def accepting(self) -> bool:
        """Whether a span recorded now is kept. Cheap: it runs on every span."""
        if self.state != OPEN:
            return True
        if self._clock() < self.open_until:
            return False
        self.state = HALF_OPEN
        return True

    @property
    def probing(self) -> bool:
        return self.state == HALF_OPEN

    def cycle_failed(self) -> bool:
        """Record an export cycle the receiver did not take; returns whether the circuit opened now."""
        if self.state == HALF_OPEN:
            self.cooldown = min(MAX_COOLDOWN_S, self.cooldown * 2)
            self._open()
            return True
        if self.state == OPEN:
            return False
        self.failures += 1
        if self.failures >= FAILURES_TO_OPEN:
            self.cooldown = BASE_COOLDOWN_S
            self._open()
            return True
        return False

    def cycle_succeeded(self) -> None:
        self.state = CLOSED
        self.failures = 0
        self.cooldown = BASE_COOLDOWN_S

    def _open(self) -> None:
        self.state = OPEN
        self.open_until = self._clock() + self.cooldown


class ExportHealth:
    """The counters ``evalkit.stats()`` reports, and the export warnings, at most one per interval."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, interval_s: float = WARN_INTERVAL_S) -> None:
        self._clock = clock
        self._interval = interval_s
        self._lock = threading.Lock()
        self.circuit = Circuit(clock)
        self.reset()

    def reset(self) -> None:
        for name in COUNTERS:
            setattr(self, name, 0)
        self.last_error: Optional[str] = None
        self._last_warning: Optional[float] = None
        self._failing = False
        self._failures_since = 0
        self._dropped_since = 0
        self.circuit.reset()

    def after_fork_in_child(self) -> None:
        self._lock = threading.Lock()

    def count(self, name: str, n: int = 1) -> None:
        if n <= 0:
            return
        setattr(self, name, getattr(self, name, 0) + n)
        if name in ("dropped_export_failed", "dropped_circuit_open"):
            self._dropped_since += n

    def export_failed(self, message: str, *, target: str = "", debug: bool = False) -> None:
        """One export request the receiver did not take."""
        message = clip(message, 400)
        with self._lock:
            self.export_failures += 1
            self.last_error = message
            self._failures_since += 1
            now = self._clock()
            due = self._last_warning is None or now - self._last_warning >= self._interval
            continuing = self._failing
            self._failing = True
            if not due:
                if debug:
                    _log.debug("export failed: %s", message)
                return
            failures, dropped = self._failures_since, self._dropped_since
            self._last_warning = now
            self._failures_since = 0
            self._dropped_since = 0
        where = f" to {target}" if target else ""
        if continuing:
            _log.warning(
                "exporting spans%s is still failing: %d failed export(s) and %d dropped span(s) since the "
                "last report; last error: %s",
                where, failures, dropped, message,
            )
        else:
            _log.warning(
                "exporting spans%s failed: %s. The application is not affected: spans that cannot be sent "
                "are dropped, and further export failures are reported at most every %d minutes "
                "(evalkit.stats() has the counts).",
                where, message, int(self._interval // 60) or 1,
            )

    def export_succeeded(self, spans: int = 0) -> None:
        with self._lock:
            if spans > 0:
                self.exported += spans
            recovered = self._failing
            self._failing = False
        if recovered:
            _log.info("exporting spans recovered")

    def snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {name: getattr(self, name, 0) for name in COUNTERS}
        out["last_error"] = self.last_error
        out["circuit_state"] = self.circuit.state
        return out
