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

_installed = False


def install_function_tracer(
    client: "EvalKitClient",
    root_dir: str,
    *,
    capture_io: bool = True,
    extra_packages: Iterable[str] = (),
) -> None:
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
            from .. import trace_module
            trace_module(module, capture_io=capture_io)
        except Exception:
            pass

    class _EvalKitTracingFinder:

        def find_spec(self, fullname, path=None, target=None):
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

                            def exec_module(module, _real=real_exec):
                                _real(module)
                                _trace_loaded(module)

                            loader.exec_module = exec_module
                            loader._evalkit_wrapped = True
                        except Exception:
                            pass
                    return spec
            except Exception:
                return None
            return None

    try:
        sys.meta_path.insert(0, _EvalKitTracingFinder())
        _installed = True
    except Exception:
        return

    for module in list(sys.modules.values()):
        if isinstance(module, types.ModuleType):
            _trace_loaded(module)
