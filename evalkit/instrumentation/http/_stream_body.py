from __future__ import annotations

import os
import threading
import time
import weakref
from typing import Any, Callable, Dict, List, Optional

_MAX_PENDING = 1024

# Re-entrant, as a body finalised by the garbage collector can finish while this thread holds it.
_pending_lock = threading.RLock()
_pending: Dict[int, "_BodyCapture"] = {}


def _reset_lock_after_fork() -> None:
    global _pending_lock
    _pending_lock = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_lock_after_fork)


def _as_bytes(chunk: Any) -> bytes:
    if isinstance(chunk, bytes):
        return chunk
    if isinstance(chunk, (bytearray, memoryview)):
        try:
            return bytes(chunk)
        except Exception:
            return b""
    if isinstance(chunk, str):
        return chunk.encode("utf-8", errors="replace")
    return b""


class _BodyCapture:

    def __init__(self, limit: int, start_ms: float, emit: Callable[..., None]) -> None:
        self.limit = max(0, int(limit or 0))
        self.start_ms = start_ms
        self.parts: List[bytes] = []
        self.size = 0
        self.total = 0
        self.chunks = 0
        self.truncated = False
        self.complete = False
        self.streamed = False
        self.ttfb_ms: Optional[int] = None
        self._emit = emit
        self._done = False
        self._active = 0
        self._lock = threading.Lock()

    def add(self, chunk: Any) -> None:
        raw = _as_bytes(chunk)
        if not raw:
            return
        with self._lock:
            if self._done:
                return
            if self.ttfb_ms is None:
                self.ttfb_ms = int(time.time() * 1000 - self.start_ms)
            self.chunks += 1
            self.total += len(raw)
            room = self.limit - self.size
            if room <= 0:
                self.truncated = True
                return
            if len(raw) > room:
                self.parts.append(raw[:room])
                self.size += room
                self.truncated = True
            else:
                self.parts.append(raw)
                self.size += len(raw)

    def add_pair(self, pair: Any) -> None:
        try:
            self.add(pair[0])
        except Exception:
            pass

    def body(self) -> str:
        try:
            return b"".join(self.parts).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def attrs(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "http.response.streamed": self.streamed,
            "http.response.stream.complete": self.complete,
            "http.response.stream.chunks": self.chunks,
        }
        body = self.body()
        if body:
            out["http.response.body"] = body
        if self.total:
            out["http.response.body.bytes"] = self.total
        if self.truncated:
            out["http.response.body.truncated"] = True
        if self.ttfb_ms is not None:
            out["http.server.time_to_first_byte_ms"] = self.ttfb_ms
        return out

    def finish(
        self,
        complete: bool = False,
        error: Optional[BaseException] = None,
        deferrable: bool = False,
    ) -> None:
        with self._lock:
            if self._done:
                return
            if deferrable and self._active > 0:
                return
            self._done = True
            if complete:
                self.complete = True
        with _pending_lock:
            _pending.pop(id(self), None)
        try:
            self._emit(self, error)
        except Exception:
            pass

    def _begin(self) -> None:
        with self._lock:
            self._active += 1

    def _end(self, complete: bool = False, error: Optional[BaseException] = None) -> None:
        with self._lock:
            if self._active > 0:
                self._active -= 1
        self.finish(complete=complete, error=error)

    def tee_sync(self, source: Any) -> Any:
        self.streamed = True
        self._begin()
        try:
            for chunk in source:
                self.add(chunk)
                yield chunk
        except GeneratorExit:
            self._end()
            raise
        except BaseException as exc:
            self._end(error=exc)
            raise
        else:
            self._end(complete=True)

    async def tee_async(self, source: Any) -> Any:
        self.streamed = True
        self._begin()
        try:
            async for chunk in source:
                self.add(chunk)
                yield chunk
        except GeneratorExit:
            self._end()
            raise
        except BaseException as exc:
            self._end(error=exc)
            raise
        else:
            self._end(complete=True)

    async def tee_async_pairs(self, source: Any) -> Any:
        self.streamed = True
        self._begin()
        try:
            async for pair in source:
                self.add_pair(pair)
                yield pair
        except GeneratorExit:
            self._end()
            raise
        except BaseException as exc:
            self._end(error=exc)
            raise
        else:
            self._end(complete=True)


def new_capture(limit: int, start_ms: float, emit: Callable[..., None]) -> _BodyCapture:
    capture = _BodyCapture(limit, start_ms, emit)
    overflow: List[_BodyCapture] = []
    with _pending_lock:
        _pending[id(capture)] = capture
        while len(_pending) > _MAX_PENDING:
            key = next(iter(_pending))
            old = _pending.pop(key)
            overflow.append(old)
    for old in overflow:
        try:
            old.finish()
        except Exception:
            pass
    return capture


def flush_pending() -> None:
    with _pending_lock:
        captures = list(_pending.values())
    for capture in captures:
        try:
            capture.finish()
        except Exception:
            pass


def pending_count() -> int:
    with _pending_lock:
        return len(_pending)


def _watch(target: Any, capture: _BodyCapture) -> None:
    try:
        weakref.finalize(target, capture.finish)
    except Exception:
        pass


def _wrap_closer(target: Any, name: str, capture: _BodyCapture) -> None:
    try:
        original = getattr(target, name)
    except Exception:
        return

    def closer(*args: Any, **kwargs: Any) -> Any:
        try:
            return original(*args, **kwargs)
        finally:
            capture.finish(deferrable=True)

    try:
        setattr(target, name, closer)
    except Exception:
        pass


def _wrap_async_closer(target: Any, name: str, capture: _BodyCapture) -> None:
    try:
        original = getattr(target, name)
    except Exception:
        return

    async def closer(*args: Any, **kwargs: Any) -> Any:
        try:
            return await original(*args, **kwargs)
        finally:
            capture.finish(deferrable=True)

    try:
        setattr(target, name, closer)
    except Exception:
        pass


def _wrap_sync_iterator(target: Any, name: str, capture: _BodyCapture) -> bool:
    try:
        original = getattr(target, name)
    except Exception:
        return False

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return capture.tee_sync(original(*args, **kwargs))

    try:
        setattr(target, name, wrapper)
        return True
    except Exception:
        return False


def _wrap_async_iterator(target: Any, name: str, capture: _BodyCapture) -> bool:
    try:
        original = getattr(target, name)
    except Exception:
        return False

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return capture.tee_async(original(*args, **kwargs))

    try:
        setattr(target, name, wrapper)
        return True
    except Exception:
        return False


def attach_requests(resp: Any, capture: _BodyCapture) -> bool:
    if not _wrap_sync_iterator(resp, "iter_content", capture):
        return False
    _wrap_closer(resp, "close", capture)
    _watch(resp, capture)
    return True


def attach_httpx_sync(resp: Any, capture: _BodyCapture) -> bool:
    if not _wrap_sync_iterator(resp, "iter_bytes", capture):
        return False
    _wrap_closer(resp, "close", capture)
    _watch(resp, capture)
    return True


def attach_httpx_async(resp: Any, capture: _BodyCapture) -> bool:
    if not _wrap_async_iterator(resp, "aiter_bytes", capture):
        return False
    _wrap_async_closer(resp, "aclose", capture)
    _watch(resp, capture)
    return True


class _ContentTee:

    def __init__(self, inner: Any, capture: _BodyCapture) -> None:
        object.__setattr__(self, "_ek_inner", inner)
        object.__setattr__(self, "_ek_capture", capture)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_ek_inner"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_ek_inner"), name, value)

    def __aiter__(self) -> Any:
        inner = object.__getattribute__(self, "_ek_inner")
        capture = object.__getattribute__(self, "_ek_capture")
        return capture.tee_async(inner.__aiter__())

    def iter_chunked(self, n: int) -> Any:
        inner = object.__getattribute__(self, "_ek_inner")
        capture = object.__getattribute__(self, "_ek_capture")
        return capture.tee_async(inner.iter_chunked(n))

    def iter_any(self) -> Any:
        inner = object.__getattribute__(self, "_ek_inner")
        capture = object.__getattribute__(self, "_ek_capture")
        return capture.tee_async(inner.iter_any())

    def iter_chunks(self) -> Any:
        inner = object.__getattribute__(self, "_ek_inner")
        capture = object.__getattribute__(self, "_ek_capture")
        return capture.tee_async_pairs(inner.iter_chunks())

    async def read(self, n: int = -1) -> Any:
        inner = object.__getattribute__(self, "_ek_inner")
        capture = object.__getattribute__(self, "_ek_capture")
        data = await inner.read(n)
        capture.add(data)
        if n < 0 or not data:
            capture.finish(complete=True)
        return data

    async def readany(self) -> Any:
        inner = object.__getattribute__(self, "_ek_inner")
        capture = object.__getattribute__(self, "_ek_capture")
        data = await inner.readany()
        capture.add(data)
        if not data:
            capture.finish(complete=True)
        return data

    async def readline(self) -> Any:
        inner = object.__getattribute__(self, "_ek_inner")
        capture = object.__getattribute__(self, "_ek_capture")
        data = await inner.readline()
        capture.add(data)
        if not data:
            capture.finish(complete=True)
        return data

    async def readuntil(self, separator: Any = b"\n") -> Any:
        inner = object.__getattribute__(self, "_ek_inner")
        capture = object.__getattribute__(self, "_ek_capture")
        data = await inner.readuntil(separator)
        capture.add(data)
        return data

    async def readexactly(self, n: int) -> Any:
        inner = object.__getattribute__(self, "_ek_inner")
        capture = object.__getattribute__(self, "_ek_capture")
        data = await inner.readexactly(n)
        capture.add(data)
        return data

    async def readchunk(self) -> Any:
        inner = object.__getattribute__(self, "_ek_inner")
        capture = object.__getattribute__(self, "_ek_capture")
        pair = await inner.readchunk()
        capture.add_pair(pair)
        try:
            if not pair[0]:
                capture.finish(complete=True)
        except Exception:
            pass
        return pair


def attach_aiohttp(resp: Any, capture: _BodyCapture) -> bool:
    try:
        inner = resp.content
    except Exception:
        return False
    if inner is None:
        return False
    try:
        resp.content = _ContentTee(inner, capture)
    except Exception:
        return False
    _wrap_closer(resp, "close", capture)
    _wrap_closer(resp, "release", capture)
    _watch(resp, capture)
    return True
