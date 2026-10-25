"""``evalkit.untraced``: a block, or a function, inside which nothing is recorded.

It holds tracing off the way a health probe is answered, through the same context variable, so every
kind of span is left out alike: model calls, HTTP calls and the traceparent they would send, function
spans, log spans and spans of other OpenTelemetry instrumentations. The context variable is per thread
and per asyncio task, so a concurrent request is traced as usual. A task or thread started inside the
block copies the context and stays untraced after the block has ended: it is work the block started.
"""
from __future__ import annotations

import functools
import inspect
import logging
from contextvars import Token
from typing import Any, Callable, List, Optional, TypeVar

from .context import _suppression, suppress_tracing

F = TypeVar("F", bound=Callable[..., Any])

_log = logging.getLogger("evalkit")

# Function tracing (trace_function, trace_tool, trace_module, trace_packages) leaves a callable with this
# marker as it is. functools.wraps copies it onto a decorator stacked on top.
UNTRACED_MARKER = "_evalkit_untraced"


def _hold() -> Optional[Token]:
    try:
        token, _held = suppress_tracing()
        return token
    except Exception:
        return None


def _release(token: Optional[Token]) -> None:
    """Restore what the context held before. The suppression itself stays on for what copied it."""
    if token is None:
        return
    try:
        _suppression.reset(token)
        return
    except Exception:
        pass
    # Left in another context than it was entered in (a scope object shared between tasks): put back
    # what this context held before instead.
    try:
        _suppression.set(None if token.old_value is Token.MISSING else token.old_value)
    except Exception:
        pass


class _UntracedScope:
    """``with evalkit.untraced():``, ``async with evalkit.untraced():`` or ``@evalkit.untraced()``."""

    __slots__ = ("_tokens",)

    def __init__(self) -> None:
        self._tokens: List[Optional[Token]] = []

    def __enter__(self) -> None:
        self._tokens.append(_hold())
        return None

    def __exit__(self, *exc_info: Any) -> bool:
        _release(self._tokens.pop() if self._tokens else None)
        return False

    async def __aenter__(self) -> None:
        return self.__enter__()

    async def __aexit__(self, *exc_info: Any) -> bool:
        return self.__exit__(*exc_info)

    def __call__(self, fn: F) -> F:
        return untraced_function(fn)


def _is_coroutine_function(fn: Any) -> bool:
    from .client import _is_coroutine_function as is_coroutine

    return is_coroutine(fn)


def untraced_function(fn: F) -> F:
    """``fn`` run with tracing held off: each call, and each step of a generator while it runs. Its
    arguments, result and exceptions pass through unchanged."""
    if getattr(fn, UNTRACED_MARKER, False):
        return fn

    if _is_coroutine_function(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            token = _hold()
            try:
                return await fn(*args, **kwargs)
            finally:
                _release(token)
        wrapper: Any = async_wrapper

    elif inspect.isasyncgenfunction(fn):
        @functools.wraps(fn)
        async def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Stepped by hand, so tracing is off while the generator runs and on again in the consumer's
            # code between its items; what the consumer sends, throws or closes reaches the generator.
            agen = fn(*args, **kwargs)
            to_send: Any = None
            to_throw: Optional[BaseException] = None
            while True:
                token = _hold()
                try:
                    if to_throw is not None:
                        thrown, to_throw = to_throw, None
                        item = await agen.athrow(thrown)
                    else:
                        item = await agen.asend(to_send)
                except StopAsyncIteration:
                    return
                finally:
                    _release(token)
                try:
                    to_send = yield item
                except GeneratorExit:
                    token = _hold()
                    try:
                        await agen.aclose()
                    finally:
                        _release(token)
                    raise
                except BaseException as exc:
                    to_throw = exc
        wrapper = async_gen_wrapper

    elif inspect.isgeneratorfunction(fn):
        @functools.wraps(fn)
        def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
            gen = fn(*args, **kwargs)
            to_send: Any = None
            to_throw: Optional[BaseException] = None
            while True:
                token = _hold()
                try:
                    if to_throw is not None:
                        thrown, to_throw = to_throw, None
                        item = gen.throw(thrown)
                    else:
                        item = gen.send(to_send)
                except StopIteration as stop:
                    return stop.value
                finally:
                    _release(token)
                try:
                    to_send = yield item
                except GeneratorExit:
                    token = _hold()
                    try:
                        gen.close()
                    finally:
                        _release(token)
                    raise
                except BaseException as exc:
                    to_throw = exc
        wrapper = gen_wrapper

    else:
        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            token = _hold()
            try:
                return fn(*args, **kwargs)
            finally:
                _release(token)
        wrapper = sync_wrapper

    try:
        setattr(wrapper, UNTRACED_MARKER, True)
    except Exception:
        pass
    return wrapper  # type: ignore[return-value]


def untraced(fn: Any = None) -> Any:
    """Record nothing inside a block or a function.

    ``@evalkit.untraced`` (or ``@evalkit.untraced()``) on a function, sync, async or a generator;
    ``with evalkit.untraced():`` and ``async with evalkit.untraced():`` around a block. Works before
    ``init()`` and never raises.
    """
    if fn is None:
        return _UntracedScope()
    if callable(fn):
        try:
            return untraced_function(fn)
        except Exception as exc:
            _log.debug("evalkit.untraced could not wrap %r: %s", fn, exc)
            return fn
    _log.debug("evalkit.untraced was given %s, not a function; it is used as a block.", type(fn).__name__)
    return _UntracedScope()
