from __future__ import annotations

import gzip
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from .redact import _resolve_flag
from .types import EvalKitOptions, TraceEnvelope

_log = logging.getLogger("evalkit")


_AUTH_FAIL_LIMIT = 3


_MAX_SPAN_BYTES = 8 * 1024 * 1024

# The most JSON one export request carries, measured before compression. A batch of large spans
# goes out as several requests, so no request comes near the 10 MiB that the proxy in front of the
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
# No answer, only 5xx ones, or a refused key: every later request would meet the same failure.
_HALT = "halt"


def resolve_export_gzip(explicit: Optional[bool]) -> bool:
    return _resolve_flag(explicit, EXPORT_GZIP_ENV_VARS, True)


def _event_size(d: Dict[str, Any]) -> int:
    try:
        return len(json.dumps(d))
    except Exception:
        return _MAX_SPAN_BYTES + 1


def _fit_event(d: Dict[str, Any], limit: int = _MAX_SPAN_BYTES) -> Optional[Dict[str, Any]]:
    if _event_size(d) <= limit:
        return d

    def note(n: int) -> str:
        return f"…[dropped {n} bytes — exceeded trace span size limit]"

    attrs = d.get("attributes")
    if isinstance(attrs, dict):
        for k in ("http.response.body", "http.request.body", "gen_ai.response.body",
                  "gen_ai.request.body", "response.body", "request.body"):
            v = attrs.get(k)
            if isinstance(v, str) and len(v) > 1024:
                attrs[k] = note(len(v))
    if _event_size(d) <= limit:
        return d

    for k in ("prompt", "completion"):
        v = d.get(k)
        if isinstance(v, str) and len(v) > 1024:
            d[k] = note(len(v))
    sm = d.get("statusMessage")
    if isinstance(sm, str) and len(sm) > 4096:
        d["statusMessage"] = sm[:4096]
    if _event_size(d) <= limit:
        return d

    if isinstance(attrs, dict):
        for k, v in list(attrs.items()):
            if isinstance(v, str) and len(v) > 512:
                attrs[k] = v[:512] + "…"
    if _event_size(d) <= limit:
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
        text = json.dumps(d)
    except Exception:
        text = None
    if text is None or len(text) > _MAX_SPAN_BYTES:
        fitted = _fit_event(d)
        if fitted is None:
            return None
        try:
            text = json.dumps(fitted)
        except Exception:
            return None
    # json.dumps escapes every non-ASCII character, so the length in characters is the length in bytes.
    return text.encode("ascii")


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
    def __init__(self, opts: EvalKitOptions) -> None:
        self._url = opts.base_url.rstrip("/") + "/v1/ingest"
        self._key = opts.subscription_key
        self._debug = opts.debug
        self._auth_failures = 0
        self._disabled = False
        self._gzip = bool(getattr(opts, "export_gzip", True))
        self._gzip_lock = threading.Lock()
        self._max_request_bytes = MAX_REQUEST_BYTES
        self._max_request_spans = max(1, int(opts.max_export_batch_size or 1))
        self._halted = False

    @property
    def halted(self) -> bool:
        """The last export stopped before its last request: the receiver did not answer, or refused the key."""
        return self._halted

    def export(self, envelopes: List[TraceEnvelope]) -> bool:
        self._halted = False
        if self._disabled:
            return False
        spans: List[_Span] = []
        for e in envelopes:
            data = _encode(e.to_dict())
            if data is not None:
                spans.append(_Span(data, e.operation or e.span_type))
        if not spans:
            return True
        ok = True
        for group in _group(spans, self._max_request_bytes, self._max_request_spans):
            if self._disabled:
                return False
            outcome = self._send(group)
            if outcome == _SENT:
                continue
            ok = False
            if outcome == _HALT:
                self._halted = True
                break
        return ok

    def _send(self, spans: List[_Span]) -> str:
        status, text = self._deliver(_body(spans))
        if 200 <= status < 300:
            self._auth_failures = 0
            if self._debug:
                _log.debug("exported %d span(s) — %s", len(spans), _status_field(text))
            return _SENT
        if status == 0 or status >= 500:
            return _HALT
        if status in (401, 403):
            # One refusal per export, however many requests it takes: a receiver briefly unable to check
            # keys must not use up the auth-failure limit within one large export.
            self._on_auth_failure(status, text)
            return _HALT
        if status == 413:
            if len(spans) > 1:
                half = len(spans) // 2
                first = self._send(spans[:half])
                if first == _HALT:
                    return first
                second = self._send(spans[half:])
                return first if second == _SENT else second
            _log.warning(
                "dropping span %r (%d bytes): the receiver rejected it as too large (HTTP 413)",
                spans[0].operation, len(spans[0].data),
            )
            return _REFUSED
        _log.warning("export rejected %d: %s", status, text)
        return _REFUSED

    def _deliver(self, body: bytes) -> Tuple[int, str]:
        if not self._gzip:
            return self._post(body, compressed=False)
        status, text = self._post(gzip.compress(body, _GZIP_LEVEL), compressed=True)
        if status not in (400, 415):
            return status, text
        # A receiver that predates compressed ingest answers 415, or 400 when it parses the gzip bytes
        # as JSON. The same body sent uncompressed tells that 400 from one about the spans themselves,
        # which the receiver refuses again.
        plain_status, plain_text = self._post(body, compressed=False)
        if status == 415 or plain_status != 400:
            self._stop_compressing(status)
        return plain_status, plain_text

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
        if self._auth_failures >= _AUTH_FAIL_LIMIT:
            self._disabled = True
            _log.warning(
                "export rejected %d (%s) — disabling tracing after %d consecutive "
                "auth failures; check your EvalKit subscription key.",
                code, detail.strip()[:200], self._auth_failures,
            )
        elif self._debug:
            _log.debug("export rejected %d (auth failure %d/%d)", code, self._auth_failures, _AUTH_FAIL_LIMIT)

    def _post(self, payload: bytes, *, compressed: bool, max_retries: int = 3) -> Tuple[int, str]:
        """POST one body, retrying network errors and 5xx answers.

        Returns the status and the response text. The status is 0 when no response arrived.
        """
        delays = [0.2, 0.6, 1.8]
        headers = {
            "Content-Type": "application/json",
            "X-Subscription-Key": self._key,
        }
        if compressed:
            headers["Content-Encoding"] = "gzip"
        req = urllib.request.Request(self._url, data=payload, headers=headers, method="POST")
        last_err: Optional[Exception] = None
        status = 0
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return resp.status, _text(resp) if self._debug else ""
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    return e.code, _text(e)
                last_err = e
                status = e.code
                if attempt < max_retries:
                    if self._debug:
                        _log.debug("HTTP %d, retry %d/%d", e.code, attempt + 1, max_retries)
                    time.sleep(delays[attempt] if attempt < len(delays) else 1.8)
            except Exception as e:
                last_err = e
                status = 0
                if attempt < max_retries:
                    if self._debug:
                        _log.debug("network error, retry %d/%d: %s", attempt + 1, max_retries, e)
                    time.sleep(delays[attempt] if attempt < len(delays) else 1.8)
        _log.warning("export failed after %d retries: %s", max_retries, last_err)
        return status, ""
