from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .types import EvalKitOptions, TraceEnvelope

_log = logging.getLogger("evalkit")


# After this many consecutive auth failures (401/403) the exporter gives up and
# disables itself. An invalid subscription key never becomes valid by retrying,
# so hammering the ingest endpoint every flush just spams logs and wastes CPU.
_AUTH_FAIL_LIMIT = 3

# Hard ceiling on a single span's serialized size, kept under Kafka's default
# 1 MiB message limit. An oversized span (e.g. a captured base64 image body from
# an image-generation response) is trimmed — and dropped if still too big —
# instead of being sent and rejected with "Message Size Too Large". Tracing must
# never ship a payload the backend can't accept.
_MAX_SPAN_BYTES = 900 * 1024


def _event_size(d: Dict[str, Any]) -> int:
    try:
        return len(json.dumps(d))
    except Exception:
        return _MAX_SPAN_BYTES + 1


def _fit_event(d: Dict[str, Any], limit: int = _MAX_SPAN_BYTES) -> Optional[Dict[str, Any]]:
    """Trim a span dict to <= limit bytes so it can always be ingested. Drops the
    heaviest fields first (raw HTTP/GenAI bodies → prompt/completion → every large
    string attribute); returns None to skip the span if even a metadata-only
    skeleton won't fit."""
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

    # Last resort: metadata-only skeleton.
    d["attributes"] = {"trace.truncated": True}
    d.pop("events", None)
    d.pop("prompt", None)
    d.pop("completion", None)
    return d if _event_size(d) <= limit else None


class _Exporter:
    def __init__(self, opts: EvalKitOptions) -> None:
        self._url = opts.base_url.rstrip("/") + "/v1/ingest"
        self._key = opts.subscription_key
        self._debug = opts.debug
        self._auth_failures = 0
        self._disabled = False

    def export(self, envelopes: List[TraceEnvelope]) -> bool:
        if self._disabled:
            return False
        # Trim/drop oversized spans so we never POST a payload Kafka will reject
        # with "Message Size Too Large" (e.g. a multi-MB base64 image body).
        events = []
        for e in envelopes:
            fitted = _fit_event(e.to_dict())
            if fitted is not None:
                events.append(fitted)
        if not events:
            return True
        body = json.dumps({"events": events}).encode()
        return self._fetch_with_retry(body, len(events))

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

    def _fetch_with_retry(self, body: bytes, count: int, max_retries: int = 3) -> bool:
        delays = [0.2, 0.6, 1.8]
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Subscription-Key": self._key,
            },
            method="POST",
        )
        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if self._debug:
                        data = json.loads(resp.read())
                        _log.debug("exported %d span(s) — %s", count, data.get("status"))
                    self._auth_failures = 0  # success resets the auth-failure streak
                    return True
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    self._on_auth_failure(e.code, e.read().decode(errors="replace"))
                    return False
                if 400 <= e.code < 500:
                    _log.warning("export rejected %d: %s", e.code, e.read().decode(errors="replace"))
                    return False
                last_err = e
                if attempt < max_retries:
                    if self._debug:
                        _log.debug("HTTP %d, retry %d/%d", e.code, attempt + 1, max_retries)
                    time.sleep(delays[attempt] if attempt < len(delays) else 1.8)
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    if self._debug:
                        _log.debug("network error, retry %d/%d: %s", attempt + 1, max_retries, e)
                    time.sleep(delays[attempt] if attempt < len(delays) else 1.8)
        _log.warning("export failed after %d retries: %s", max_retries, last_err)
        return False
