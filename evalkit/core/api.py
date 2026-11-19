from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, Mapping, Optional
from urllib.parse import urlsplit

from evalkit_cli.redirects import urlopen

from .context import _current_span

_BODY_CAP = 8192
_MESSAGE_CAP = 1000


class ApiError(urllib.error.HTTPError):
    """A call to the SyntropyLabs API that the server answered with an error status.

    ``str()`` names the call, the status and the server's own message ("POST /scenarios/generate:
    HTTP 400: No model to generate scenarios with ..."), since the message says what to fix where the
    status alone does not. It is an ``HTTPError``, so code that catches one or reads ``.code`` keeps
    working. ``status`` is the HTTP status, ``server_message`` the server's message (None when the body
    had none), ``response_text`` the start of the body, and ``retry_after`` the seconds the server asked
    to wait (``Retry-After``), None when it named none.
    """

    def __init__(self, method: str, url: str, status: int, server_message: Optional[str],
                 response_text: str, headers: Any = None, reason: Optional[str] = None) -> None:
        self.method = method
        self.path = urlsplit(url).path or "/"
        # `status` is HTTPError's own read-only alias of `code`.
        self.server_message = server_message
        self.response_text = response_text
        self.retry_after = _retry_after(headers)
        detail = server_message or reason or "no message"
        super().__init__(url, status, detail, headers, io.BytesIO(response_text.encode("utf-8")))

    def __str__(self) -> str:
        return f"{self.method} {self.path}: HTTP {self.code}: {self.server_message or self.reason}"

    def __repr__(self) -> str:
        return f"ApiError({self.method} {self.path}, {self.code}, {self.server_message!r})"


def server_message(text: str) -> Optional[str]:
    """The message of an error body: Nest's ``message`` (a list joined with "; "), FastAPI's ``detail``,
    an ``error``, else the start of the text. None for an empty body."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        body = json.loads(text)
    except ValueError:
        return text[:300]
    if isinstance(body, Mapping):
        for key in ("message", "detail", "error"):
            found = _message_of(body.get(key))
            if found:
                return found[:_MESSAGE_CAP]
    return text[:300]


def _message_of(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        parts = [_message_of(v) for v in value]
        joined = "; ".join(p for p in parts if p)
        return joined or None
    if isinstance(value, Mapping):
        for key in ("message", "msg", "error", "detail"):
            found = _message_of(value.get(key))
            if found:
                return found
    return None


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


def _scrubbed(text: str, secrets: Iterable[Any]) -> str:
    for value in secrets:
        if isinstance(value, str) and len(value) >= 4:
            text = text.replace(value, "***")
    return text


def post_json(
    client: Any,
    url: str,
    body: Dict[str, Any],
    *,
    timeout: float,
    span_name: str,
    redact: Iterable[str] = (),
    trace: bool = True,
) -> Dict[str, Any]:
    payload = json.dumps(body, default=str)
    hidden = set(redact)
    logged = json.dumps({k: v for k, v in body.items() if k not in hidden}, default=str) if hidden else payload
    secrets = [body.get(k) for k in hidden]
    return _request(client, "POST", url, payload.encode(), logged, timeout=timeout, span_name=span_name, trace=trace,
                    secrets=secrets)


def get_json(client: Any, url: str, *, timeout: float, span_name: str) -> Dict[str, Any]:
    return _request(client, "GET", url, None, None, timeout=timeout, span_name=span_name)


def _request(
    client: Any,
    method: str,
    url: str,
    data: Optional[bytes],
    logged_body: Optional[str],
    *,
    timeout: float,
    span_name: str,
    trace: bool = True,
    secrets: Iterable[Any] = (),
) -> Dict[str, Any]:
    """One call to our API. A refusal raises `ApiError` with the server's message, never the key that
    authenticated the call nor a credential the body carried."""
    headers = {"X-Subscription-Key": client._opts.subscription_key}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    attrs: Dict[str, Any] = {
        "evalkit.span_type": "http_call",
        "http.method": method,
        "http.url": url,
    }
    if logged_body is not None:
        attrs["http.request.body"] = logged_body[:_BODY_CAP]
    end_span, ctx = None, None
    if trace:
        try:
            end_span, ctx = client.start_span(span_name, attrs)
        except Exception:
            end_span, ctx = None, None

    try:
        token = _current_span.set(ctx) if ctx is not None else None
        try:
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                attrs["http.status_code"] = resp.getcode()
                attrs["http.response.body"] = raw[:_BODY_CAP].decode("utf-8", errors="replace")
        finally:
            if token is not None:
                _current_span.reset(token)
    except urllib.error.HTTPError as exc:
        attrs["http.status_code"] = exc.code
        try:
            text = exc.read()[:_BODY_CAP].decode("utf-8", errors="replace")
        except Exception:
            text = ""
        text = _scrubbed(text, [client._opts.subscription_key, *secrets])
        attrs["http.response.body"] = text
        if end_span:
            end_span("ERROR")
        raise ApiError(method, url, exc.code, server_message(text), text, exc.headers,
                       reason=str(exc.reason) if exc.reason else None) from None
    except Exception as exc:
        attrs["error.message"] = str(exc)
        if end_span:
            end_span("ERROR")
        raise

    if end_span:
        end_span("OK")
    return json.loads(raw)
