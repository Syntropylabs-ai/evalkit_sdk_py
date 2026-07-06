import shutil
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

import python_minifier


class CustomBuildHook(BuildHookInterface):
    """Ships evalkit as minified source in the wheel — locals renamed,
    whitespace/docstrings/comments stripped — to keep the proprietary
    implementation out of a plain-text pip install. remove_annotations is
    forced off: @dataclass (used in core/context.py, core/types.py,
    simulation/types.py) reads __annotations__ at class-definition time, so
    stripping them would silently drop every dataclass field."""

    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return

        pkg_dir = Path(self.root) / "evalkit"
        build_root = Path(self.directory) / "_minified_build"
        if build_root.exists():
            shutil.rmtree(build_root)
        dst_pkg = build_root / "evalkit"
        shutil.copytree(pkg_dir, dst_pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

        for py_file in dst_pkg.rglob("*.py"):
            source = py_file.read_text()
            minified = python_minifier.minify(source, filename=str(py_file), remove_annotations=False)
            py_file.write_text(minified)

        for py_file in dst_pkg.rglob("*.py"):
            build_data["force_include"][str(py_file)] = str(py_file.relative_to(build_root))
