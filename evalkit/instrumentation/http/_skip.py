from __future__ import annotations

from urllib.parse import urlsplit

# The receiver's own ingest endpoint: an export must never trace itself.
_INGEST_PATH = "/v1/ingest"

# Probe endpoints that load balancers and orchestrators poll every few seconds. Matched as a whole
# path segment, so /healthcare and /api/health-records are an app's own calls and are recorded.
_PROBE_SEGMENTS = frozenset({
    "health", "healthz", "healthcheck", "health-check", "health_check",
    "livez", "readyz", "startupz",
})
# What follows a health segment in a probe path: /health/live, /actuator/health/readiness.
_PROBE_QUALIFIERS = frozenset({
    "live", "liveness", "ready", "readiness", "startup", "started", "alive",
})


def _path_segments(url: str) -> list:
    try:
        path = urlsplit(url).path if "://" in url else url.split("?", 1)[0].split("#", 1)[0]
    except Exception:
        return []
    return [segment.lower() for segment in path.split("/") if segment]


def is_health_check(url: str) -> bool:
    """Whether a URL is a health or readiness probe: its last path segment names one, or a health
    segment is followed only by a liveness or readiness qualifier."""
    segments = _path_segments(url)
    if not segments:
        return False
    if segments[-1] in _PROBE_SEGMENTS:
        return True
    return len(segments) >= 2 and segments[-1] in _PROBE_QUALIFIERS and segments[-2] in _PROBE_SEGMENTS


def should_skip(url: object) -> bool:
    try:
        text = url.decode("utf-8", "replace") if isinstance(url, (bytes, bytearray)) else str(url)
        return _INGEST_PATH in text or is_health_check(text)
    except Exception:
        return False
