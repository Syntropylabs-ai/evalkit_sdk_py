"""JSON for a recorded value that costs a bounded amount, whatever the value is.

Recording a function's arguments or result runs inside the application's own call. ``json.dumps``
walks all of a value before anything can be cut, so a 100 MB list cost the call that time; this writer
stops once it has the bytes it may keep. For a value that fits it writes what ``json.dumps`` writes.
Secrets are masked as the value is walked: by key in a mapping, by shape in a string.
"""
from __future__ import annotations

import json
from typing import Any, Callable, List, Optional

from ..utils.headers import MASK
from .secrets import is_secret_key, scrub_string, scrub_value

# Deeper than this, a value is written as a placeholder: every level writes at least one character, so the
# cap bounds depth too, but a cap of a few KiB still allows a recursion deep enough to exhaust the stack.
_MAX_DEPTH = 100

# A string is scrubbed with this much of it past the cut, so a credential that straddles the cut is still
# recognised by its whole shape (the longest assignment the scrubber reads is 1024 characters).
_SCRUB_MARGIN = 1024

_CIRCULAR = '"[Circular]"'
_DEEP = '"[Object]"'
_UNREADABLE = '"[Unserializable]"'

_encode_str = json.encoder.encode_basestring_ascii  # type: ignore[attr-defined]


class _Full(Exception):
    """The budget is spent: nothing more of the value is read."""


class _Out:
    __slots__ = ("parts", "size", "limit", "cut")

    def __init__(self, limit: int) -> None:
        self.parts: List[str] = []
        self.size = 0
        self.limit = limit
        self.cut = False

    @property
    def room(self) -> int:
        return self.limit - self.size

    def push(self, text: str) -> None:
        room = self.limit - self.size
        if len(text) > room:
            if room > 0:
                self.parts.append(text[:room])
                self.size = self.limit
            self.cut = True
            raise _Full()
        self.parts.append(text)
        self.size += len(text)


def _float(value: float) -> str:
    return json.dumps(value)


def _key(key: Any) -> Optional[str]:
    if isinstance(key, str):
        return key
    if key is True:
        return "true"
    if key is False:
        return "false"
    if key is None:
        return "null"
    if isinstance(key, int):
        return int.__repr__(key)
    if isinstance(key, float):
        return _float(key)
    return None


class _Writer:
    def __init__(self, out: _Out, scrub: bool, fallback: Callable[[Any], str]) -> None:
        self.out = out
        self.scrub = scrub
        self.fallback = fallback
        self.path: set = set()

    def string(self, text: str) -> None:
        room = self.out.room
        if self.scrub:
            if len(text) > room + _SCRUB_MARGIN:
                # A cut string no longer parses as JSON, so it is scrubbed as text, past the cut by the margin.
                text = scrub_string(text[: room + _SCRUB_MARGIN])
            else:
                scrubbed = scrub_value(text)
                text = scrubbed if isinstance(scrubbed, str) else text
        elif len(text) > room + 8:
            text = text[: room + 8]
        self.out.push(_encode_str(text))

    def value(self, value: Any, depth: int) -> None:
        out = self.out
        if value is None:
            out.push("null")
        elif value is True:
            out.push("true")
        elif value is False:
            out.push("false")
        elif isinstance(value, str):
            self.string(value)
        elif isinstance(value, int):
            try:
                text = int.__repr__(value)
            except ValueError:
                # Past the interpreter's limit on digits converted to text.
                text = _UNREADABLE
            out.push(text)
        elif isinstance(value, float):
            out.push(_float(value))
        elif isinstance(value, (bytes, bytearray, memoryview)):
            # Only the part that can be kept is turned into text: str() of a whole upload would be
            # four times its size.
            try:
                self.string(str(bytes(value[: self.out.room + 8])) if len(value) > self.out.room + 8 else str(value))
            except _Full:
                raise
            except Exception:
                out.push(_UNREADABLE)
        elif isinstance(value, (list, tuple, dict)):
            if depth >= _MAX_DEPTH:
                out.push(_DEEP)
                return
            marker = id(value)
            if marker in self.path:
                out.push(_CIRCULAR)
                return
            self.path.add(marker)
            try:
                if isinstance(value, dict):
                    self.mapping(value, depth)
                else:
                    self.sequence(value, depth)
            finally:
                self.path.discard(marker)
        else:
            self.other(value)

    def sequence(self, value: Any, depth: int) -> None:
        out = self.out
        out.push("[")
        first = True
        for item in value:
            if not first:
                out.push(", ")
            first = False
            self.value(item, depth + 1)
        out.push("]")

    def mapping(self, value: dict, depth: int) -> None:
        out = self.out
        out.push("{")
        first = True
        for key, item in value.items():
            name = _key(key)
            if name is None:
                name = str(key)
            if not first:
                out.push(", ")
            first = False
            out.push(_encode_str(name))
            out.push(": ")
            if self.scrub and is_secret_key(name):
                out.push(_encode_str(MASK))
            else:
                self.value(item, depth + 1)
        out.push("}")

    def other(self, value: Any) -> None:
        try:
            text = self.fallback(value)
        except Exception:
            self.out.push(_UNREADABLE)
            return
        if not isinstance(text, str):
            self.out.push(_UNREADABLE)
            return
        self.string(text)


def bounded_json(value: Any, limit: int, *, scrub: bool = True, fallback: Callable[[Any], str] = str) -> str:
    """``value`` as JSON, at most ``limit`` characters of it, followed by ``…`` when it was cut.

    A value JSON has no form for is written as its ``str()``, as a string. A failure to read any part of
    the value writes a placeholder for that part rather than raising.
    """
    out = _Out(max(0, int(limit)))
    writer = _Writer(out, scrub, fallback)
    try:
        writer.value(value, 0)
    except _Full:
        pass
    except RecursionError:
        out.cut = True
    except Exception:
        return _UNREADABLE.strip('"')
    text = "".join(out.parts)
    return text + "…" if out.cut else text
