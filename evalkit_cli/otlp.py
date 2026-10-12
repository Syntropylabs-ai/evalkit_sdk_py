from __future__ import annotations

import email.utils
import json
import random
import time
import urllib.error
import urllib.request
from datetime import timezone
from typing import Any, Dict, Iterable, Optional, Tuple

_MAX_ATTR = 4096
MAX_CONTENT = 524288
SDK_NAME = "evalkit-cli"

# The retry rules of evalkit.core.exporter, kept here because a hook imports this module on every agent
# event and importing evalkit costs about 100 ms.
RETRY_BUDGET = 20.0
_MAX_RETRY_WAIT = 10.0
_BACKOFF = (0.2, 0.6, 1.8)
_MAX_RETRIES = 3
# Below this, a retry would have no time left for its own answer.
_MIN_ATTEMPT = 0.2


def json_text(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=str)


def attr(key: str, value: Any, limit: int = _MAX_ATTR) -> Dict[str, Any]:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    if isinstance(value, str):
        return {"key": key, "value": {"stringValue": value[:limit]}}
    return {"key": key, "value": {"stringValue": json_text(value)[:limit]}}


def log_item(
    name: str,
    attributes: Iterable[Dict[str, Any]],
    body: Optional[str] = None,
    trace: Optional[str] = None,
    span: Optional[str] = None,
    at_ns: Optional[int] = None,
) -> Dict[str, Any]:
    when = str(at_ns if at_ns else time.time_ns())
    record: Dict[str, Any] = {
        "timeUnixNano": when,
        "observedTimeUnixNano": str(time.time_ns()),
        "severityText": "INFO",
        "body": {"stringValue": body or name},
        "attributes": [attr("event.name", name), *attributes],
    }
    if trace:
        record["traceId"] = trace
    if span:
        record["spanId"] = span
    return record


def log_envelope(vendor: str, records: Iterable[Dict[str, Any]], service: Optional[str] = None) -> Dict[str, Any]:
    return {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        attr("service.name", service or vendor),
                        attr("agent.vendor", vendor),
                        attr("telemetry.sdk.name", SDK_NAME),
                    ]
                },
                "scopeLogs": [
                    {
                        "scope": {"name": SDK_NAME},
                        "logRecords": list(records),
                    }
                ],
            }
        ]
    }


def log_record(
    vendor: str,
    event: str,
    attributes: Iterable[Dict[str, Any]],
    body: Optional[str] = None,
    prefix: Optional[str] = None,
    trace: Optional[str] = None,
    span: Optional[str] = None,
    service: Optional[str] = None,
) -> Dict[str, Any]:
    name = f"{prefix or vendor}.{event}"
    return log_envelope(vendor, [log_item(name, attributes, body, trace, span)], service)


def _retryable(status: Optional[int]) -> bool:
    return status is None or status == 429 or status >= 500


def _retry_after(header: Optional[str], now: float) -> Optional[float]:
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


def _retry_wait(attempt: int, header: Optional[str]) -> float:
    asked = _retry_after(header, time.time())
    if asked is not None:
        return min(asked, _MAX_RETRY_WAIT)
    step = _BACKOFF[min(attempt, len(_BACKOFF) - 1)]
    return step / 2 + random.random() * step


def _post_once(req: urllib.request.Request, timeout: float) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), None, None
    except urllib.error.HTTPError as exc:
        retry_after = exc.headers.get("Retry-After") if exc.code in (429, 503) and exc.headers is not None else None
        try:
            exc.close()
        except Exception:
            pass
        return exc.code, exc.reason, retry_after
    except Exception as exc:
        return None, str(exc), None


def send(
    endpoint: str,
    key: str,
    signal: str,
    record: Dict[str, Any],
    timeout: float = 4.0,
    deadline: Optional[float] = None,
) -> Tuple[Optional[int], Optional[str]]:
    """POST one OTLP record. With a `deadline` (time.monotonic()), no answer, 429 and 5xx are retried while
    the wait and another attempt fit before it, a Retry-After on 429 or 503 setting the wait; without one the
    record is tried once, which is what a diagnostic wants to see."""
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/" + signal,
        data=json.dumps(record).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    attempt = 0
    while True:
        if deadline is not None:
            timeout = min(timeout, max(_MIN_ATTEMPT, deadline - time.monotonic()))
        status, error, retry_after = _post_once(req, timeout)
        if deadline is None or not _retryable(status) or attempt >= _MAX_RETRIES:
            return status, error
        wait = _retry_wait(attempt, retry_after)
        if time.monotonic() + wait + _MIN_ATTEMPT > deadline:
            return status, error
        time.sleep(wait)
        attempt += 1


def send_logs(
    endpoint: str, key: str, record: Dict[str, Any], timeout: float = 4.0, deadline: Optional[float] = None
) -> Tuple[Optional[int], Optional[str]]:
    return send(endpoint, key, "logs", record, timeout, deadline)


def send_traces(
    endpoint: str, key: str, record: Dict[str, Any], timeout: float = 4.0, deadline: Optional[float] = None
) -> Tuple[Optional[int], Optional[str]]:
    return send(endpoint, key, "traces", record, timeout, deadline)


def get_json(url: str, key: str, timeout: float = 8.0) -> Tuple[Optional[int], Any, Optional[str]]:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                return resp.getcode(), json.loads(raw or "{}"), None
            except ValueError:
                return resp.getcode(), None, "response was not JSON"
    except urllib.error.HTTPError as exc:
        return exc.code, None, exc.reason
    except Exception as exc:
        return None, None, str(exc)


def read_base(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    for suffix in ("/v1/otlp", "/otlp"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base
