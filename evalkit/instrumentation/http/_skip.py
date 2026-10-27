from __future__ import annotations

import re
from urllib.parse import urlsplit

from ...core.context import tracing_suppressed
from ...core.ignore import ignored_url, traces_health_checks

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


# A segment that names one record, or a route template's parameter standing for one. A probe never
# sits under a record, so /datasets/<id>/health is the dataset's own route and is recorded;
# /api/v2/health is a probe.
_RECORD_SEGMENT_RES = tuple(re.compile(p) for p in (
    r"^\d+$",
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    r"^(?=[^\d]*\d)[0-9a-f]{12,}$",
    r"^(?=(?:[^\d]*\d){2})[a-z0-9_-]{16,}$",
    r"^(?::.+|\{.+\})$",
))


def _names_record(segment: str) -> bool:
    return any(pattern.match(segment) for pattern in _RECORD_SEGMENT_RES)


def _path_segments(url: str) -> list:
    try:
        path = urlsplit(url).path if "://" in url else url.split("?", 1)[0].split("#", 1)[0]
    except Exception:
        return []
    return [segment.lower() for segment in path.split("/") if segment]


def is_health_check(url: str) -> bool:
    """Whether a URL is a health or readiness probe: its last path segment names one, or a health
    segment is followed only by a liveness or readiness qualifier, and no segment before it names a
    record."""
    segments = _path_segments(url)
    if not segments:
        return False
    if segments[-1] in _PROBE_SEGMENTS:
        probe_at = len(segments) - 1
    elif len(segments) >= 2 and segments[-1] in _PROBE_QUALIFIERS and segments[-2] in _PROBE_SEGMENTS:
        probe_at = len(segments) - 2
    else:
        return False
    return not any(_names_record(segment) for segment in segments[:probe_at])


def ignored_outgoing(url: str, method: object = None) -> bool:
    """Whether an outgoing call is left out of tracing by rule: a health probe, unless
    trace_health_checks is on, or a match for ignore_urls."""
    return (not traces_health_checks() and is_health_check(url)) or ignored_url(url, method)


def should_skip(url: object, method: object = None) -> bool:
    """Whether an outgoing call is made without a span and without a traceparent: the export itself, a
    call made while tracing is held off (a probe, an ignored route, an untraced block), or one left out
    by rule. A traceparent naming a span that is never exported would point the next service at a
    parent that never arrives."""
    try:
        text = url.decode("utf-8", "replace") if isinstance(url, (bytes, bytearray)) else str(url)
        return _INGEST_PATH in text or tracing_suppressed() or ignored_outgoing(text, method)
    except Exception:
        return False
