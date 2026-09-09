from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, Optional

from .context import _current_span

_BODY_CAP = 8192


def post_json(
    client: Any,
    url: str,
    body: Dict[str, Any],
    *,
    timeout: float,
    span_name: str,
    redact: Iterable[str] = (),
) -> Dict[str, Any]:
    payload = json.dumps(body, default=str)
    hidden = set(redact)
    logged = json.dumps({k: v for k, v in body.items() if k not in hidden}, default=str) if hidden else payload
    return _request(client, "POST", url, payload.encode(), logged, timeout=timeout, span_name=span_name)


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
) -> Dict[str, Any]:
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
    try:
        end_span, ctx = client.start_span(span_name, attrs)
    except Exception:
        end_span, ctx = None, None

    try:
        token = _current_span.set(ctx) if ctx is not None else None
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
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
            attrs["http.response.body"] = text
            exc.response_text = text
        except Exception:
            pass
        if end_span:
            end_span("ERROR")
        raise
    except Exception as exc:
        attrs["error.message"] = str(exc)
        if end_span:
            end_span("ERROR")
        raise

    if end_span:
        end_span("OK")
    return json.loads(raw)
