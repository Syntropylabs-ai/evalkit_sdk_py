from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, Optional, Tuple

_MAX_ATTR = 4096
MAX_CONTENT = 524288
SDK_NAME = "evalkit-cli"


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


def send(endpoint: str, key: str, signal: str, record: Dict[str, Any], timeout: float = 4.0) -> Tuple[Optional[int], Optional[str]]:
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/" + signal,
        data=json.dumps(record).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), None
    except urllib.error.HTTPError as exc:
        return exc.code, exc.reason
    except Exception as exc:
        return None, str(exc)


def send_logs(endpoint: str, key: str, record: Dict[str, Any], timeout: float = 4.0) -> Tuple[Optional[int], Optional[str]]:
    return send(endpoint, key, "logs", record, timeout)


def send_traces(endpoint: str, key: str, record: Dict[str, Any], timeout: float = 4.0) -> Tuple[Optional[int], Optional[str]]:
    return send(endpoint, key, "traces", record, timeout)


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
