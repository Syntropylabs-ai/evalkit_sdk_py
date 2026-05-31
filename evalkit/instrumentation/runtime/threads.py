from __future__ import annotations

import contextvars
import threading


def patch_threads() -> None:
    """
    Patch threading.Thread.start and concurrent.futures.ThreadPoolExecutor.submit
    so every thread/executor task automatically inherits the current ContextVar
    state (including the active evalkit trace context).

    Without this:
    - threading.Thread gets an empty ContextVar copy → orphan traces
    - run_in_executor / ThreadPoolExecutor.submit → orphan traces
    - Fire-and-forget background tasks lose the request trace context
    """
    _patch_thread_start()
    _patch_thread_pool_executor()


def _patch_thread_start() -> None:
    if getattr(threading.Thread.start, "_evalkit", False):
        return

    original_start = threading.Thread.start

    def patched_start(self: threading.Thread) -> None:
        if getattr(self, "_evalkit_ctx_wrapped", False):
            return original_start(self)

        target = self.__dict__.get("_target")
        if target is None:
            return original_start(self)

        ctx = contextvars.copy_context()
        args = self.__dict__.get("_args", ())
        kwargs = self.__dict__.get("_kwargs", {}) or {}

        self.__dict__["_target"] = lambda: ctx.run(target, *args, **kwargs)
        self.__dict__["_args"] = ()
        self.__dict__["_kwargs"] = {}
        object.__setattr__(self, "_evalkit_ctx_wrapped", True)

        return original_start(self)

    patched_start._evalkit = True  # type: ignore[attr-defined]
    threading.Thread.start = patched_start  # type: ignore[method-assign]


def _patch_thread_pool_executor() -> None:
    """
    Patch ThreadPoolExecutor.submit so tasks submitted to a pool inherit the
    calling thread's ContextVars. This covers asyncio's run_in_executor() and
    any explicit ThreadPoolExecutor usage.
    """
    try:
        from concurrent.futures import ThreadPoolExecutor
    except ImportError:
        return

    if getattr(ThreadPoolExecutor.submit, "_evalkit", False):
        return

    original_submit = ThreadPoolExecutor.submit

    def patched_submit(self: ThreadPoolExecutor, fn, *args, **kwargs):  # type: ignore[override]
        ctx = contextvars.copy_context()
        return original_submit(self, ctx.run, fn, *args, **kwargs)

    patched_submit._evalkit = True  # type: ignore[attr-defined]
    ThreadPoolExecutor.submit = patched_submit  # type: ignore[method-assign]
