"""Request and response bodies of a server span, kept as the application reads and writes them.

Nothing is read ahead of the application: the request body is copied from the messages the app itself
receives, up to ``max_body_bytes``, so an upload is never held in memory twice and a streamed request is
never buffered before the app sees its first chunk. Uploads (multipart, octet-stream, media) are not kept
at all, nor are server-sent event streams and compressed responses, which are not text a reader can use.
Each body is gathered as a list of chunks with a running length, so keeping it costs one copy, not one
per chunk.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional

_BINARY_PREFIXES = (
    "multipart/",
    "application/octet-stream",
    "image/",
    "audio/",
    "video/",
    "application/zip",
    "application/gzip",
    "application/pdf",
)


def _media_type(content_type: Any) -> str:
    if isinstance(content_type, bytes):
        content_type = content_type.decode("latin-1", "replace")
    return str(content_type or "").split(";", 1)[0].strip().lower()


def is_binary(content_type: Any) -> bool:
    media = _media_type(content_type)
    return any(media.startswith(prefix) for prefix in _BINARY_PREFIXES)


def is_event_stream(content_type: Any) -> bool:
    return _media_type(content_type) == "text/event-stream"


def decode_body(raw: bytes) -> Any:
    """A body as JSON when it parses, else as text; None when empty."""
    if not raw:
        return None
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except Exception:
        return text or None


class _Chunks:
    __slots__ = ("limit", "parts", "kept", "total", "truncated", "skipped")

    def __init__(self, limit: int, skipped: bool = False) -> None:
        self.limit = max(0, int(limit or 0))
        self.parts: List[bytes] = []
        self.kept = 0
        self.total = 0
        self.truncated = False
        self.skipped = skipped or self.limit <= 0

    def add(self, chunk: Any) -> None:
        if self.skipped or not chunk:
            return
        try:
            size = len(chunk)
        except Exception:
            return
        self.total += size
        room = self.limit - self.kept
        if room <= 0:
            self.truncated = True
            return
        piece = bytes(chunk[:room]) if size > room else bytes(chunk)
        if size > room:
            self.truncated = True
        self.parts.append(piece)
        self.kept += len(piece)

    def raw(self) -> bytes:
        return b"".join(self.parts)

    def value(self) -> Any:
        if self.skipped:
            return None
        return decode_body(self.raw())

    def attributes(self, prefix: str) -> Dict[str, Any]:
        if not self.truncated:
            return {}
        return {f"{prefix}.truncated": True, f"{prefix}.bytes": self.total}


def _header(headers: Mapping[str, str], name: str) -> str:
    for key, value in headers.items():
        if str(key).lower() == name:
            return str(value)
    return ""


class RequestBodyTap:
    """Wraps an ASGI ``receive``: hands every message on unchanged and keeps the body it carries."""

    def __init__(self, receive: Callable[[], Any], limit: int, headers: Mapping[str, str]) -> None:
        self._receive = receive
        self._chunks = _Chunks(limit, skipped=is_binary(_header(headers, "content-type")))

    async def receive(self) -> Any:
        message = await self._receive()
        try:
            if message.get("type") == "http.request":
                self._chunks.add(message.get("body", b""))
        except Exception:
            pass
        return message

    def value(self) -> Any:
        try:
            return self._chunks.value()
        except Exception:
            return None

    def attributes(self) -> Dict[str, Any]:
        return self._chunks.attributes("http.request.body")


class ResponseBodyTap:
    """Keeps a response body from the ASGI messages the app sends, up to the limit."""

    def __init__(self, limit: int) -> None:
        self._chunks = _Chunks(limit)

    def start(self, headers: Mapping[str, str]) -> None:
        encoding = _header(headers, "content-encoding").strip().lower()
        content_type = _header(headers, "content-type")
        if (is_event_stream(content_type) or is_binary(content_type)
                or (encoding and encoding != "identity")):
            self._chunks.skipped = True
            self._chunks.parts = []

    def add(self, chunk: Any) -> None:
        self._chunks.add(chunk)

    def value(self) -> Any:
        try:
            return self._chunks.value()
        except Exception:
            return None

    def attributes(self) -> Dict[str, Any]:
        return self._chunks.attributes("http.response.body")


def capped_prefix(chunks: Any, limit: int) -> Optional[bytes]:
    """The first ``limit`` bytes of a response already held as a list of chunks, without joining all of it."""
    if not isinstance(chunks, (list, tuple)):
        return None
    out = _Chunks(limit)
    for chunk in chunks:
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        if not isinstance(chunk, (bytes, bytearray)):
            return None
        out.add(chunk)
        if out.kept >= out.limit:
            break
    return out.raw()
