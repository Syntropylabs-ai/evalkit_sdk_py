from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import List, Optional

from .types import EvalKitOptions, TraceEnvelope

_log = logging.getLogger("evalkit")


class _Exporter:
    def __init__(self, opts: EvalKitOptions) -> None:
        self._url = opts.base_url.rstrip("/") + "/v1/ingest"
        self._key = opts.subscription_key
        self._debug = opts.debug

    def export(self, envelopes: List[TraceEnvelope]) -> bool:
        body = json.dumps({"events": [e.to_dict() for e in envelopes]}).encode()
        return self._fetch_with_retry(body, len(envelopes))

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
                    return True
            except urllib.error.HTTPError as e:
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
