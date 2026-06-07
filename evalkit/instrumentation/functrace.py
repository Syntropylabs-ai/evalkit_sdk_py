"""Automatic function tracing via an import hook.

`init()` installs a `sys.meta_path` finder that, as each of your app's own modules
is imported, wraps its module-level functions with `trace_function` (one
`function_call` span per call, APM-style). This is the same strategy OpenTelemetry
uses — instrument at import time rather than asking the user to wire every module —
and it's necessary because `init()` runs *before* the app's modules are imported, so
there is nothing to wrap yet at init time.

Scope is deliberately narrow and safe:
  - only modules whose file lives under the app's own source directory (the dir of
    the file that called `init()`), or under an explicitly named `extra_packages`;
  - never third-party libraries (site-packages / dist-packages / venv) or evalkit;
  - only module-level functions (never class methods), via the existing
    `trace_module`, which uses `functools.wraps` so framework signature
    introspection (FastAPI `Depends`, etc.) keeps working.

Everything is wrapped in `try/except`: a failure here must never break an import.
"""

from __future__ import annotations

import os
import sys
import types
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from ..core.client import EvalKitClient

_EXCLUDED_PATH_MARKERS = (
    os.sep + "site-packages" + os.sep,
    os.sep + "dist-packages" + os.sep,
    os.sep + ".venv" + os.sep,
    os.sep + "venv" + os.sep,
    os.sep + "lib" + os.sep + "python",
)

# Process-global guard so we never install more than one finder.
_installed = False


def install_function_tracer(
    client: "EvalKitClient",
    root_dir: str,
    *,
    capture_io: bool = True,
    extra_packages: Iterable[str] = (),
) -> None:
    """Install the import hook. Idempotent and exception-safe."""
    global _installed
    if _installed:
        return

    try:
        root_dir = os.path.realpath(root_dir)
    except Exception:
        return
    extra = tuple(extra_packages or ())
    traced_names: set[str] = set()

    def _in_scope(module: types.ModuleType) -> bool:
        name = getattr(module, "__name__", "") or ""
        if not name or name.startswith("evalkit"):
            return False
        # Explicitly requested packages (covers sibling packages outside root_dir).
        top = name.split(".")[0]
        if top in extra or name in extra:
            return True
        path = getattr(module, "__file__", None)
        if not path:
            return False
        try:
            rp = os.path.realpath(path)
        except Exception:
            return False
        if any(marker in rp for marker in _EXCLUDED_PATH_MARKERS):
            return False
        return rp == root_dir or rp.startswith(root_dir + os.sep)

    def _trace_loaded(module: types.ModuleType) -> None:
        try:
            name = getattr(module, "__name__", "") or ""
            if name in traced_names or not _in_scope(module):
                return
            traced_names.add(name)
            from .. import trace_module  # imported lazily; uses the active client
            trace_module(module, capture_io=capture_io)
        except Exception:
            pass

    class _EvalKitTracingFinder:
        """MetaPathFinder that wraps each in-scope module's loader to trace on import."""

        def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
            try:
                for finder in list(sys.meta_path):
                    if finder is self:
                        continue
                    find = getattr(finder, "find_spec", None)
                    if find is None:
                        continue
                    try:
                        spec = find(fullname, path, target)
                    except Exception:
                        spec = None
                    if spec is None:
                        continue
                    loader = getattr(spec, "loader", None)
                    if (
                        loader is not None
                        and hasattr(loader, "exec_module")
                        and not getattr(loader, "_evalkit_wrapped", False)
                    ):
                        try:
                            real_exec = loader.exec_module

                            def exec_module(module, _real=real_exec):  # noqa: ANN001
                                _real(module)
                                _trace_loaded(module)

                            loader.exec_module = exec_module  # type: ignore[method-assign]
                            loader._evalkit_wrapped = True  # type: ignore[attr-defined]
                        except Exception:
                            pass  # C loaders / __slots__ — leave untouched
                    return spec
            except Exception:
                return None
            return None

    try:
        sys.meta_path.insert(0, _EvalKitTracingFinder())
        _installed = True
    except Exception:
        return

    # Trace anything already imported and in scope (rare at init time, but possible).
    for module in list(sys.modules.values()):
        if isinstance(module, types.ModuleType):
            _trace_loaded(module)
