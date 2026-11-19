"""The HTTP the voice test runner makes: JSON calls to the SyntropyLabs API, the caller audio download
and the recording upload, with one retry policy.

Stdlib only (urllib), so the runner adds no dependency beyond the voice extra, and runs in a worker
thread so a slow request never stalls a call's 20 ms frame loop.
"""

from __future__ import annotations

import http.client
import json
import logging
import random
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, BinaryIO, Callable, Dict, Mapping, Optional, Tuple, TypeVar, Union
from urllib.parse import urlsplit, urlunsplit

from evalkit_cli.redirects import urlopen

log = logging.getLogger("evalkit.voicetest.net")

T = TypeVar("T")

# Statuses worth asking again: the server or a proxy in front of it was briefly unavailable.
RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
# A Retry-After longer than this is a quota, not a hiccup: waiting for it would only stall the run.
MAX_RETRY_AFTER_S = 30.0
_BODY_CAP = 64 * 1024
# ENETUNREACH and EHOSTUNREACH on Linux, macOS and Windows: no route, so nothing was sent.
_UNREACHABLE = frozenset({101, 113, 51, 65, 10051, 10065})


def display_url(url: str) -> str:
    """The URL without its query, user info or fragment, for messages: signed URLs carry their
    credentials in the query."""
    try:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))
    except ValueError:
        return "<url>"


class HttpError(Exception):
    """A request that failed: `status` is None when no answer came back at all."""

    def __init__(self, method: str, url: str, status: Optional[int], message: str,
                 retry_after_s: Optional[float] = None, unsent: bool = False) -> None:
        self.method = method
        self.url = display_url(url)
        self.status = status
        self.message = message
        self.retry_after_s = retry_after_s
        # The connection was never made (refused, unresolvable, unreachable), so the server cannot
        # have acted on the request: even a request that is not idempotent can be sent again.
        self.unsent = unsent
        where = f"{method} {self.url}"
        super().__init__(f"{where}: HTTP {status}: {message}" if status else f"{where}: {message}")

    @property
    def retryable(self) -> bool:
        if self.status is None:
            return True
        if self.status == 429:
            return self.retry_after_s is not None and self.retry_after_s <= MAX_RETRY_AFTER_S
        return self.status in RETRY_STATUSES

    @property
    def retryable_once_sent(self) -> bool:
        """Whether a request that is not idempotent may be sent again: only when the server did
        not take it (no connection, 503, or a 429 that names a short wait)."""
        if self.unsent:
            return True
        return self.status == 503 or (self.status == 429 and self.retryable)


@dataclass
class RetryPolicy:
    """`attempts` in all, waiting `base_s`, then twice as long each time up to `cap_s`, with up to a
    quarter of jitter so parallel calls do not retry in step."""

    attempts: int = 6
    base_s: float = 0.5
    cap_s: float = 8.0

    def delay(self, attempt: int, error: Optional[HttpError] = None) -> float:
        if error is not None and error.retry_after_s is not None:
            return min(error.retry_after_s, MAX_RETRY_AFTER_S)
        wait = min(self.cap_s, self.base_s * (2 ** attempt))
        return wait * (1.0 + 0.25 * random.random())


def with_retries(send: Callable[[], T], policy: RetryPolicy, what: str,
                 sleep: Callable[[float], None] = time.sleep, idempotent: bool = True) -> T:
    """`send()` until it succeeds, retrying the failures `HttpError.retryable` allows (for a request
    that is not `idempotent`, those `HttpError.retryable_once_sent` allows)."""
    for attempt in range(max(1, policy.attempts)):
        try:
            return send()
        except HttpError as exc:
            again = exc.retryable if idempotent else exc.retryable_once_sent
            if not again or attempt + 1 >= policy.attempts:
                raise
            wait = policy.delay(attempt, exc)
            log.info("%s failed (%s); retrying in %.1f s", what, exc, wait)
            sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover


def _retry_after(headers: Any) -> Optional[float]:
    value = headers.get("Retry-After") if headers is not None else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def _server_message(raw: bytes) -> str:
    text = raw[:_BODY_CAP].decode("utf-8", errors="replace")
    try:
        body = json.loads(text)
    except ValueError:
        return text.strip()[:300] or "no body"
    if isinstance(body, Mapping):
        message = body.get("message") or body.get("error")
        if isinstance(message, list):
            message = "; ".join(str(m) for m in message)
        if isinstance(message, Mapping):
            message = message.get("message")
        if message:
            return str(message)[:500]
    return text.strip()[:300]


def request(
    method: str,
    url: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    body: Union[None, bytes, BinaryIO] = None,
    timeout: float = 30.0,
    max_bytes: Optional[int] = None,
    sink: Optional[Callable[[bytes], None]] = None,
) -> Tuple[int, bytes]:
    """One request; returns the status and the body (empty when `sink` took it chunk by chunk).

    Raises HttpError for a non-2xx answer or when nothing came back. `max_bytes` caps a body read
    into memory or into `sink`.
    """
    req = urllib.request.Request(url, data=body, method=method, headers=dict(headers or {}))
    try:
        with urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            if sink is None:
                data = resp.read() if max_bytes is None else resp.read(max_bytes + 1)
                if max_bytes is not None and len(data) > max_bytes:
                    raise HttpError(method, url, None, f"the answer is over {max_bytes} bytes")
                return status, data
            seen = 0
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                seen += len(chunk)
                if max_bytes is not None and seen > max_bytes:
                    raise HttpError(method, url, None, f"the answer is over {max_bytes} bytes")
                sink(chunk)
            return status, b""
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read()
        except Exception:
            raw = b""
        raise HttpError(method, url, exc.code, _server_message(raw) or str(exc.reason),
                        retry_after_s=_retry_after(exc.headers)) from None
    except HttpError:
        raise
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError,
            http.client.HTTPException, OSError) as exc:
        reason = getattr(exc, "reason", None) or exc
        unsent = isinstance(reason, (ConnectionRefusedError, socket.gaierror)) or (
            isinstance(reason, OSError) and getattr(reason, "errno", None) in _UNREACHABLE)
        raise HttpError(method, url, None, f"{type(reason).__name__}: {reason}", unsent=unsent) from None


def json_request(method: str, url: str, *, headers: Dict[str, str], body: Optional[Dict[str, Any]] = None,
                 timeout: float = 30.0) -> Dict[str, Any]:
    data = None
    sent = dict(headers)
    sent.setdefault("Accept", "application/json")
    if body is not None:
        data = json.dumps(body, separators=(",", ":"), default=str).encode("utf-8")
        sent["Content-Type"] = "application/json"
    _, raw = request(method, url, headers=sent, body=data, timeout=timeout)
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        raise HttpError(method, url, None, "the answer is not JSON") from None
    if not isinstance(parsed, dict):
        raise HttpError(method, url, None, "the answer is not a JSON object")
    return parsed
