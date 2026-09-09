from __future__ import annotations

from hashlib import sha256

TRACE_ID_HEX = 32
SPAN_ID_HEX = 16


def _digest(parts: tuple) -> str:
    return sha256("".join(str(part or "") for part in parts).encode()).hexdigest()


def trace_id(*parts) -> str:
    return _digest(parts)[:TRACE_ID_HEX]


def span_id(*parts) -> str:
    return _digest(parts)[:SPAN_ID_HEX]
