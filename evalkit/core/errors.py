"""What a span records about the exception that failed it.

The shape follows OpenTelemetry's exception conventions and is shared with the TypeScript SDK and the
receiver, so an error reads alike from either SDK:

- status ERROR, described as ``"<type>: <message>"``, and ``error.type`` holding the exception's fully
  qualified class name;
- one span event named ``exception`` with ``exception.type``, ``exception.message`` (at most 2 KiB),
  ``exception.stacktrace`` (at most 16 KiB, its start and its end kept) and ``exception.escaped``;
- ``evalkit.exception.frames``, ``evalkit.exception.source`` and ``evalkit.exception.causes`` on the
  span, for grouping and for pointing at the application's own line;
- ``evalkit.exception.fingerprint`` only when the application chose one.

Nothing of a frame's locals or of a call's arguments is read: only file names, line numbers, function
and module names, and the exception's own text.
"""
from __future__ import annotations

import json
import os
import sysconfig
import traceback
from contextvars import ContextVar, Token
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .limits import cut_end, cut_middle

MAX_MESSAGE_BYTES = 2 * 1024
MAX_STACKTRACE_BYTES = 16 * 1024
MAX_FRAMES = 64
MAX_CAUSES = 5
MAX_CAUSE_FRAMES = 16
MAX_CAUSE_MESSAGE_BYTES = 1024

EXCEPTION_EVENT = "exception"
FRAMES_ATTR = "evalkit.exception.frames"
SOURCE_ATTR = "evalkit.exception.source"
CAUSES_ATTR = "evalkit.exception.causes"
FINGERPRINT_ATTR = "evalkit.exception.fingerprint"
CANCELLED_ATTR = "evalkit.cancelled"

_EVALKIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep

_in_app_include: Tuple[str, ...] = ()
_in_app_exclude: Tuple[str, ...] = ()


def _library_dirs() -> Tuple[str, ...]:
    dirs = set()
    try:
        paths = sysconfig.get_paths()
    except Exception:
        paths = {}
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        value = paths.get(key)
        if value:
            dirs.add(os.path.abspath(value))
    try:
        import site

        for value in list(getattr(site, "getsitepackages", lambda: [])()) + [site.getusersitepackages()]:
            if value:
                dirs.add(os.path.abspath(value))
    except Exception:
        pass
    return tuple(sorted(d.rstrip(os.sep) + os.sep for d in dirs))


_LIBRARY_DIRS = _library_dirs()
_LIBRARY_MARKERS = (os.sep + "site-packages" + os.sep, os.sep + "dist-packages" + os.sep)


def _prefixes(values: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if not values:
        return ()
    if isinstance(values, str):
        values = [values]
    return tuple(str(v).strip() for v in values if str(v).strip())


def configure(in_app_include: Optional[Iterable[str]] = None, in_app_exclude: Optional[Iterable[str]] = None) -> None:
    """Module prefixes that are, or are not, the application's own code, whatever their path says."""
    global _in_app_include, _in_app_exclude
    _in_app_include = _prefixes(in_app_include)
    _in_app_exclude = _prefixes(in_app_exclude)


def _module_matches(module: Optional[str], prefixes: Tuple[str, ...]) -> bool:
    if not module:
        return False
    return any(module == p or module.startswith(p.rstrip(".") + ".") for p in prefixes)


def is_in_app(filename: Optional[str], module: Optional[str] = None) -> bool:
    """Whether a frame is the application's own code rather than the interpreter's, a library's or EvalKit's.

    An exclusion wins over an inclusion, so a vendored package inside an included one can be left out
    (in_app_include=["mycompany"], in_app_exclude=["mycompany.vendored"]), as in the TypeScript SDK."""
    if _module_matches(module, _in_app_exclude):
        return False
    if _module_matches(module, _in_app_include):
        return True
    if module and (module == "evalkit" or module.startswith("evalkit.") or module.startswith("evalkit_cli")):
        return False
    if not filename or filename.startswith("<"):
        return False
    path = os.path.abspath(filename)
    if path.startswith(_EVALKIT_DIR):
        return False
    if any(marker in path for marker in _LIBRARY_MARKERS):
        return False
    return not any(path.startswith(d) for d in _LIBRARY_DIRS)


def _public_module(module: str) -> str:
    # openai._exceptions.RateLimitError is raised and documented as openai.RateLimitError: the private
    # modules a package defines its classes in say nothing about the error and change between releases.
    parts = [p for p in module.split(".") if p and not p.startswith("_")]
    return ".".join(parts)


def type_name(exc: Any) -> str:
    """The fully qualified class name: ``openai.RateLimitError``; a builtin exception by its name alone."""
    cls = exc if isinstance(exc, type) else type(exc)
    qualname = getattr(cls, "__qualname__", None) or getattr(cls, "__name__", None) or "Exception"
    module = getattr(cls, "__module__", None) or ""
    if module in ("builtins", "exceptions", "__builtin__"):
        return qualname
    public = _public_module(module)
    return f"{public}.{qualname}" if public else qualname


def _message(exc: BaseException) -> str:
    try:
        return str(exc)
    except Exception:
        return ""


def describe(exc: BaseException) -> str:
    """``"<type>: <message>"``, or the type alone for an exception without a message."""
    name = type_name(exc)
    message = _message(exc)
    return f"{name}: {message}" if message else name


def _frames(exc: BaseException, te: Optional[traceback.TracebackException], limit: int) -> List[Dict[str, Any]]:
    """The exception's frames, innermost first."""
    modules: List[Optional[str]] = []
    try:
        for frame, _lineno in traceback.walk_tb(exc.__traceback__):
            try:
                modules.append(frame.f_globals.get("__name__"))
            except Exception:
                modules.append(None)
    except Exception:
        modules = []
    try:
        stack = list(te.stack) if te is not None else traceback.extract_tb(exc.__traceback__)
    except Exception:
        stack = []
    frames: List[Dict[str, Any]] = []
    count = len(stack)
    for index in range(count - 1, max(-1, count - 1 - limit), -1):
        summary = stack[index]
        module = modules[index] if index < len(modules) and len(modules) == count else None
        filename = getattr(summary, "filename", None) or ""
        frame: Dict[str, Any] = {
            "file": filename,
            "line": getattr(summary, "lineno", None) or 0,
        }
        colno = getattr(summary, "colno", None)
        if isinstance(colno, int):
            # Columns count from 1, as a JavaScript stack's do.
            frame["col"] = colno + 1
        frame["function"] = getattr(summary, "name", None) or ""
        if module:
            frame["module"] = module
        frame["in_app"] = is_in_app(filename, module)
        frames.append(frame)
    return frames


def _chain(exc: BaseException) -> List[BaseException]:
    """The exceptions behind this one: what it was raised from, else what was being handled."""
    causes: List[BaseException] = []
    seen = {id(exc)}
    current: Optional[BaseException] = exc
    while current is not None and len(causes) < MAX_CAUSES:
        cause = current.__cause__
        if cause is None and not current.__suppress_context__:
            cause = current.__context__
        if cause is None or id(cause) in seen:
            break
        seen.add(id(cause))
        causes.append(cause)
        current = cause
    return causes


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)


def _source(frames: Sequence[Dict[str, Any]]) -> Optional[str]:
    for frame in frames:
        if frame.get("in_app"):
            return f"{frame['file']}:{frame['line']}"
    return None


class ExceptionRecord:
    """Everything recorded about one exception, built once."""

    __slots__ = ("type", "message", "stacktrace", "frames", "causes", "status_message", "legacy_source")

    def __init__(self, exc: BaseException) -> None:
        self.type = type_name(exc)
        message = _message(exc)
        self.message = cut_end(message, MAX_MESSAGE_BYTES)
        te: Optional[traceback.TracebackException]
        try:
            te = traceback.TracebackException.from_exception(exc)
        except Exception:
            te = None
        try:
            text = "".join(te.format()) if te is not None else describe(exc)
        except Exception:
            text = describe(exc)
        self.stacktrace = cut_middle(text, MAX_STACKTRACE_BYTES)[0]
        self.frames = _frames(exc, te, MAX_FRAMES)
        causes = []
        for cause in _chain(exc):
            entry: Dict[str, Any] = {
                "type": type_name(cause),
                "message": cut_end(_message(cause), MAX_CAUSE_MESSAGE_BYTES),
            }
            cause_frames = _frames(cause, None, MAX_CAUSE_FRAMES)
            if cause_frames:
                entry["frames"] = cause_frames
            causes.append(entry)
        self.causes = causes
        self.status_message = f"{self.type}: {self.message}" if self.message else self.type
        source = _source(self.frames)
        if source is None and self.frames:
            source = f"{self.frames[0]['file']}:{self.frames[0]['line']}"
        self.legacy_source = source

    def span_attributes(self, fingerprint: Optional[str] = None) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {"error.type": self.type}
        if self.frames:
            attrs[FRAMES_ATTR] = _json(self.frames)
        source = _source(self.frames)
        if source:
            attrs[SOURCE_ATTR] = source
        if self.causes:
            attrs[CAUSES_ATTR] = _json(self.causes)
        if fingerprint:
            attrs[FINGERPRINT_ATTR] = fingerprint
        return attrs

    def legacy_attributes(self, exc: BaseException) -> Dict[str, Any]:
        # The attributes earlier releases wrote, which existing readers of a span still look for.
        attrs: Dict[str, Any] = {
            "exception.type": getattr(type(exc), "__qualname__", self.type),
            "exception.message": self.message,
            "exception.stacktrace": self.stacktrace,
        }
        if self.legacy_source:
            attrs["error.source"] = self.legacy_source
        return attrs

    def event_attributes(self, escaped: bool) -> Dict[str, Any]:
        return {
            "exception.type": self.type,
            "exception.message": self.message,
            "exception.stacktrace": self.stacktrace,
            "exception.escaped": bool(escaped),
        }


_fingerprint: ContextVar[Optional[str]] = ContextVar("evalkit_error_fingerprint", default=None)


def _normalise_fingerprint(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = "|".join(str(v) for v in value)
    text = str(value).strip()
    return cut_end(text, 512) if text else None


def set_fingerprint(value: Any) -> Token:
    return _fingerprint.set(_normalise_fingerprint(value))


def reset_fingerprint(token: Optional[Token] = None) -> None:
    if token is None:
        _fingerprint.set(None)
        return
    try:
        _fingerprint.reset(token)
    except Exception:
        _fingerprint.set(None)


def current_fingerprint() -> Optional[str]:
    return _fingerprint.get()


def exc_attributes(exc: BaseException) -> Dict[str, Any]:
    """The span attributes for ``exc``: the earlier ones and the current ones. No event: for a call site
    that can hand only attributes on."""
    try:
        record = ExceptionRecord(exc)
    except Exception:
        return {"exception.type": type(exc).__qualname__, "exception.message": _message(exc)}
    attrs = record.legacy_attributes(exc)
    attrs.update(record.span_attributes(current_fingerprint()))
    if is_cancellation(exc):
        attrs[CANCELLED_ATTR] = True
    return attrs


def is_cancellation(exc: BaseException) -> bool:
    try:
        import asyncio

        return isinstance(exc, asyncio.CancelledError)
    except Exception:
        return False


def apply_exception(
    envelope: Any,
    exc: BaseException,
    escaped: bool = True,
    timestamp: Optional[str] = None,
    mark_error: bool = True,
) -> None:
    """Record ``exc`` on a span by the shared contract; never raises.

    ``mark_error=False`` records the exception without failing the span (a warning logged with one).
    """
    try:
        record = ExceptionRecord(exc)
    except Exception:
        try:
            if mark_error:
                envelope.status = "ERROR"
                envelope.status_message = envelope.status_message or describe(exc)
        except Exception:
            pass
        return
    from .types import SpanEvent

    if mark_error:
        envelope.status = "ERROR"
        envelope.status_message = record.status_message
    attrs = dict(envelope.attributes or {})
    for key, value in record.legacy_attributes(exc).items():
        attrs.setdefault(key, value)
    fingerprint = attrs.get(FINGERPRINT_ATTR) or current_fingerprint()
    attrs.update(record.span_attributes(fingerprint if isinstance(fingerprint, str) else None))
    if is_cancellation(exc):
        attrs[CANCELLED_ATTR] = True
    envelope.attributes = attrs
    if timestamp is None:
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    events = list(envelope.events or [])
    if not any(getattr(e, "name", None) == EXCEPTION_EVENT for e in events):
        events.append(SpanEvent(name=EXCEPTION_EVENT, timestamp=timestamp, attributes=record.event_attributes(escaped)))
    envelope.events = events
