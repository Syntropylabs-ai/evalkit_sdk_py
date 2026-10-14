from __future__ import annotations

import logging

_log = logging.getLogger("evalkit")


def hook_failed(where: str, exc: BaseException) -> None:
    """Note a failure inside EvalKit's own instrumentation.

    Tracing is never worth breaking the call it observes: the caller's code goes on as if it were
    not instrumented, and the failure is logged at debug level only, so a defect in the SDK shows
    up in its own logs instead of in the application.
    """
    try:
        _log.debug("evalkit instrumentation failed in %s; the call itself was not affected", where, exc_info=exc)
    except Exception:
        pass


def field(obj: object, key: str, default: object = None) -> object:
    """A request item's field, whether the caller passed a dict or the provider's own object (a
    pydantic message returned by an earlier call and sent back in the next request)."""
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
    except Exception:
        return default
