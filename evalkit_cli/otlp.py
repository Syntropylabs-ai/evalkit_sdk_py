from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, Optional, Tuple

_MAX_ATTR = 4096


def json_text(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=str)


def attr(key: str, value: Any) -> Dict[str, Any]:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    if isinstance(value, str):
        return {"key": key, "value": {"stringValue": value[:_MAX_ATTR]}}
    return {"key": key, "value": {"stringValue": json_text(value)[:_MAX_ATTR]}}


def log_record(
    vendor: str,
    event: str,
    attributes: Iterable[Dict[str, Any]],
    body: Optional[str] = None,
    prefix: Optional[str] = None,
    trace: Optional[str] = None,
    span: Optional[str] = None,
) -> Dict[str, Any]:
    now = str(time.time_ns())
    name = f"{prefix or vendor}.{event}"
    record: Dict[str, Any] = {
        "timeUnixNano": now,
        "observedTimeUnixNano": now,
        "severityText": "INFO",
        "body": {"stringValue": body or name},
        "attributes": [attr("event.name", name), *attributes],
    }
    if trace:
        record["traceId"] = trace
    if span:
        record["spanId"] = span
    return {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        attr("service.name", vendor),
                        attr("agent.vendor", vendor),
                        attr("telemetry.sdk.name", "evalkit-cli"),
                    ]
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "evalkit-cli"},
                        "logRecords": [record],
                    }
                ],
            }
        ]
    }


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
