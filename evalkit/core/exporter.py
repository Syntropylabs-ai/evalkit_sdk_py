from __future__ import annotations

import email.utils
import gzip
import json
import logging
import random
import threading
import time
import urllib.error
import urllib.request
from datetime import timezone
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from .health import ExportHealth, clip
from .limits import fit_dict
from .redact import _resolve_flag
from .types import EvalKitOptions, TraceEnvelope

_log = logging.getLogger("evalkit")


_AUTH_FAIL_LIMIT = 3


# The most one span may take once encoded, in UTF-8 bytes: under the 10 MiB a message may be on the
# receiver's queue, so a span is never refused for its size after it was sent. A larger span keeps the
# start and the end of its longest values.
_MAX_SPAN_BYTES = 8 * 1024 * 1024

# The most JSON one export request carries, measured before compression. A batch of large spans
# goes out as several requests, so no request comes near the 16 MiB that the proxy in front of the
# receiver accepts, and a refused request costs only its own spans, not the whole batch.
MAX_REQUEST_BYTES = 4 * 1024 * 1024

EXPORT_GZIP_ENV_VARS = ("EVALKIT_EXPORT_GZIP",)

# zlib's default: close to the ratio of level 9 for a fraction of the CPU time.
_GZIP_LEVEL = 6

_BODY_PREFIX = b'{"events":['
_BODY_SUFFIX = b"]}"

_SENT = "sent"
# The receiver refused this request's spans; the next request may still be accepted.
_REFUSED = "refused"
# No answer, only 5xx or 429 ones, or a refused key: every later request would meet the same failure.
_HALT = "halt"

# How long one export may spend waiting between attempts, every request in it included. The flush at
# exit waits for the export in flight, and behind the OpenTelemetry bridge the BatchSpanProcessor gives
# an export 30 s (OTEL_BSP_EXPORT_TIMEOUT).
_RETRY_BUDGET_S = 20.0
# The longest single wait, whatever Retry-After asks for, so one request cannot use up the budget.
_MAX_RETRY_WAIT_S = 10.0
_BACKOFF_S = (0.2, 0.6, 1.8)

_clock: Callable[[], float] = time.monotonic
_jitter: Callable[[], float] = random.random


def _retryable(status: int) -> bool:
    """429 and 5xx say the receiver could not take the request now; every other 4xx is about the request."""
    return status == 429 or status >= 500


def _retry_after_s(header: Optional[str], now: float) -> Optional[float]:
    """Retry-After as delta-seconds or an HTTP-date, measured from the wall-clock `now`; None when unreadable."""
    value = (header or "").strip()
    if not value:
        return None
    if value.isascii() and value.isdigit():
        return float(value)
    if not any(c.isalpha() for c in value):
        return None
    try:
        at = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return max(0.0, at.timestamp() - now)


def _retry_delay_s(attempt: int, retry_after: Optional[str], now: float) -> float:
    """What the receiver asked for, capped, or else the backoff step spread between half and one and a
    half times its value, so clients refused together do not return together."""
    asked = _retry_after_s(retry_after, now)
    if asked is not None:
        return min(asked, _MAX_RETRY_WAIT_S)
    step = _BACKOFF_S[min(attempt, len(_BACKOFF_S) - 1)]
    return step / 2 + _jitter() * step


def resolve_export_gzip(explicit: Optional[bool]) -> bool:
    return _resolve_flag(explicit, EXPORT_GZIP_ENV_VARS, True)


def _dumps(d: Dict[str, Any]) -> bytes:
    # UTF-8 rather than ASCII escapes: a character outside ASCII costs its UTF-8 size, not six or
    # twelve bytes, and what is measured here is what the receiver counts. A lone surrogate, which
    # UTF-8 cannot carry, is sent as U+FFFD.
    return json.dumps(d, ensure_ascii=False).encode("utf-8", "replace")


def _event_size(d: Dict[str, Any]) -> int:
    try:
        return len(_dumps(d))
    except Exception:
        return _MAX_SPAN_BYTES + 1


def _fit_event(d: Dict[str, Any], limit: int = _MAX_SPAN_BYTES) -> Optional[Dict[str, Any]]:
    """The span within ``limit`` bytes: its longest values keep their start and their end, each cut
    recorded as ``evalkit.truncated.<field>``; None when even that cannot make it fit."""
    size = _event_size(d)
    for _ in range(6):
        if size <= limit:
            return d
        if not fit_dict(d, size - limit + 1024):
            break
        size = _event_size(d)
    if size <= limit:
        return d
    d["attributes"] = {"trace.truncated": True}
    d.pop("events", None)
    d.pop("prompt", None)
    d.pop("completion", None)
    return d if _event_size(d) <= limit else None


class _Span(NamedTuple):
    data: bytes
    operation: str


def _encode(d: Dict[str, Any]) -> Optional[bytes]:
    try:
        data: Optional[bytes] = _dumps(d)
    except Exception:
        data = None
    if data is None or len(data) > _MAX_SPAN_BYTES:
        fitted = _fit_event(d)
        if fitted is None:
            return None
        try:
            data = _dumps(fitted)
        except Exception:
            return None
    return data


def _group(spans: Sequence[_Span], max_bytes: int, max_spans: int) -> List[List[_Span]]:
    """Split spans, in order, into request bodies of at most max_spans spans and max_bytes of JSON.

    A span larger than max_bytes on its own travels alone.
    """
    groups: List[List[_Span]] = []
    current: List[_Span] = []
    size = len(_BODY_PREFIX) + len(_BODY_SUFFIX)
    for span in spans:
        if current and (len(current) >= max_spans or size + 1 + len(span.data) > max_bytes):
            groups.append(current)
            current = []
            size = len(_BODY_PREFIX) + len(_BODY_SUFFIX)
        size += len(span.data) + (1 if current else 0)
        current.append(span)
    if current:
        groups.append(current)
    return groups


def _body(spans: Sequence[_Span]) -> bytes:
    return _BODY_PREFIX + b",".join(s.data for s in spans) + _BODY_SUFFIX


def _text(resp: Any) -> str:
    try:
        return resp.read().decode(errors="replace")
    except Exception:
        return ""


def _status_field(text: str) -> Any:
    try:
        return json.loads(text).get("status")
    except Exception:
        return None


class _Exporter:
    def __init__(self, opts: EvalKitOptions, health: Optional[ExportHealth] = None) -> None:
        self._url = str(opts.base_url or "").rstrip("/") + "/v1/ingest"
        self._key = str(opts.subscription_key or "")
        self._debug = opts.debug
        self._auth_failures = 0
        self._disabled = False
        self._gzip = bool(getattr(opts, "export_gzip", True))
        self._gzip_lock = threading.Lock()
        self._max_request_bytes = MAX_REQUEST_BYTES
        self._max_request_spans = max(1, int(opts.max_export_batch_size or 1))
        self._halted = False
        self.health = health if health is not None else ExportHealth()
        # How many spans the last export delivered.
        self.last_sent: Optional[int] = None
        # Set while the process is about to end: no export may wait past it.
        self.exit_deadline: Optional[float] = None
        # A deadline the calling thread's flush(timeout=...) set for its own export.
        self._local = threading.local()

    def after_fork_in_child(self) -> None:
        # The export thread that held this lock in the parent does not exist in the child.
        self._gzip_lock = threading.Lock()
        self._local = threading.local()

    def set_call_deadline(self, deadline: Optional[float]) -> Optional[float]:
        """Bound the exports this thread makes until it is set back; returns the previous deadline."""
        previous = getattr(self._local, "deadline", None)
        self._local.deadline = deadline
        return previous

    @property
    def disabled(self) -> bool:
        """The receiver refused the subscription key repeatedly: nothing is sent for the rest of the process."""
        return self._disabled

    @property
    def halted(self) -> bool:
        """The last export stopped before its last request: the receiver did not answer, or refused the key."""
        return self._halted

    def _failed(self, message: str) -> None:
        try:
            self.health.export_failed(message, target=self._url, debug=self._debug)
        except Exception:
            pass

    def export(self, envelopes: List[TraceEnvelope]) -> bool:
        self._halted = False
        self.last_sent = 0
        if self._disabled:
            return False
        spans: List[_Span] = []
        for e in envelopes:
            try:
                data = _encode(e.to_dict())
            except Exception:
                data = None
            if data is not None:
                spans.append(_Span(data, e.operation or e.span_type))
        if not spans:
            self.last_sent = len(envelopes)
            return True
        deadline = _clock() + _RETRY_BUDGET_S
        if self.exit_deadline is not None:
            deadline = min(deadline, self.exit_deadline)
        call_deadline = getattr(self._local, "deadline", None)
        if call_deadline is not None:
            deadline = min(deadline, call_deadline)
        ok = True
        sent = 0
        for group in _group(spans, self._max_request_bytes, self._max_request_spans):
            if self._disabled:
                ok = False
                break
            outcome, delivered = self._send(group, deadline)
            sent += delivered
            if outcome == _SENT:
                continue
            ok = False
            if outcome == _HALT:
                self._halted = True
                break
        # A span too large to encode at all was dropped on purpose; it is not an export failure.
        self.last_sent = sent + (len(envelopes) - len(spans))
        return ok

    def _send(self, spans: List[_Span], deadline: float) -> Tuple[str, int]:
        """Send one request's spans; the outcome, and how many of them the receiver took."""
        status, text, error = self._deliver(_body(spans), deadline)
        if 200 <= status < 300:
            self._auth_failures = 0
            if self._debug:
                _log.debug("exported %d span(s) — %s", len(spans), _status_field(text))
            return _SENT, len(spans)
        if status == 0 or _retryable(status):
            detail = f"HTTP {status}" if status else (error or "no response")
            self._failed(f"{detail} after retrying")
            return _HALT, 0
        if status in (401, 403):
            # One refusal per export, however many requests it takes: a receiver briefly unable to check
            # keys must not use up the auth-failure limit within one large export.
            self._on_auth_failure(status, text)
            return _HALT, 0
        if status == 413:
            if len(spans) > 1:
                half = len(spans) // 2
                first, first_sent = self._send(spans[:half], deadline)
                if first == _HALT:
                    return first, first_sent
                second, second_sent = self._send(spans[half:], deadline)
                return (first if second == _SENT else second), first_sent + second_sent
            self._failed(
                f"dropping span {spans[0].operation!r} ({len(spans[0].data)} bytes): the receiver rejected it "
                "as too large (HTTP 413)"
            )
            return _REFUSED, 0
        self._failed(f"export rejected {status}: {clip(text)}")
        return _REFUSED, 0

    def _deliver(self, body: bytes, deadline: float) -> Tuple[int, str, Optional[str]]:
        if not self._gzip:
            return self._post(body, compressed=False, deadline=deadline)
        status, text, error = self._post(gzip.compress(body, _GZIP_LEVEL), compressed=True, deadline=deadline)
        if status not in (400, 415):
            return status, text, error
        # A receiver that predates compressed ingest answers 415, or 400 when it parses the gzip bytes
        # as JSON. The same body sent uncompressed tells that 400 from one about the spans themselves,
        # which the receiver refuses again.
        plain_status, plain_text, plain_error = self._post(body, compressed=False, deadline=deadline)
        if status == 415 or plain_status != 400:
            self._stop_compressing(status)
        return plain_status, plain_text, plain_error

    def _stop_compressing(self, status: int) -> None:
        with self._gzip_lock:
            if not self._gzip:
                return
            self._gzip = False
        _log.warning(
            "the receiver at %s does not accept gzip-compressed spans (HTTP %d); sending them "
            "uncompressed from now on. Set EVALKIT_EXPORT_GZIP=false to skip the compressed attempt.",
            self._url, status,
        )

    def _on_auth_failure(self, code: int, detail: str) -> None:
        self._auth_failures += 1
        try:
            self.health.export_failures += 1
            self.health.last_error = f"HTTP {code}: the receiver refused the subscription key"
        except Exception:
            pass
        reason = clip(detail) or "no detail"
        if self._auth_failures >= _AUTH_FAIL_LIMIT:
            self._disabled = True
            _log.warning(
                "the receiver at %s refused the subscription key %d times in a row (HTTP %d: %s). EvalKit has "
                "stopped exporting spans for the rest of this process; check the subscription_key passed to "
                "evalkit.init().",
                self._url, self._auth_failures, code, reason,
            )
        elif self._auth_failures == 1:
            _log.warning(
                "the receiver at %s refused the subscription key (HTTP %d: %s). EvalKit stops exporting "
                "after %d refusals in a row; check the subscription_key passed to evalkit.init().",
                self._url, code, reason, _AUTH_FAIL_LIMIT,
            )
        elif self._debug:
            _log.debug("export rejected %d (auth failure %d/%d)", code, self._auth_failures, _AUTH_FAIL_LIMIT)

    def _post(
        self, payload: bytes, *, compressed: bool, deadline: float, max_retries: int = 3,
    ) -> Tuple[int, str, Optional[str]]:
        """POST one body, retrying no answer, 429 and 5xx while the export's budget lasts.

        A Retry-After on 429 or 503 sets the wait. Returns the status, the response text, and what went
        wrong when no response arrived; the status is 0 then. Never raises.
        """
        headers = {
            "Content-Type": "application/json",
            "X-Subscription-Key": self._key,
        }
        if compressed:
            headers["Content-Encoding"] = "gzip"
        try:
            req = urllib.request.Request(self._url, data=payload, headers=headers, method="POST")
        except Exception as e:
            # A base_url that is not a URL: nothing can be sent, and saying so is all there is to do.
            return 0, "", f"{type(e).__name__}: {e}"
        last_err: Optional[BaseException] = None
        status = 0
        for attempt in range(max_retries + 1):
            retry_after: Optional[str] = None
            # No single attempt outlives the export's deadline, so a receiver that never answers
            # cannot hold a flush at exit for the full socket timeout.
            timeout = min(10.0, max(0.05, deadline - _clock()))
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.status, _text(resp) if self._debug else "", None
            except urllib.error.HTTPError as e:
                if not _retryable(e.code):
                    return e.code, _text(e), None
                last_err = e
                status = e.code
                if e.code in (429, 503) and e.headers is not None:
                    retry_after = e.headers.get("Retry-After")
                # An unread error body holds its connection for as long as the wait lasts.
                try:
                    e.close()
                except Exception:
                    pass
            except Exception as e:
                last_err = e
                status = 0
            if attempt == max_retries:
                break
            wait = _retry_delay_s(attempt, retry_after, time.time())
            if _clock() + wait > deadline:
                break
            if self._debug:
                _log.debug("%s, retry %d/%d in %.1fs", f"HTTP {status}" if status else "network error",
                           attempt + 1, max_retries, wait)
            time.sleep(wait)
        error = None
        if last_err is not None:
            reason = getattr(last_err, "reason", None)
            error = clip(f"{type(last_err).__name__}: {reason if reason is not None else last_err}")
        return status, "", error
