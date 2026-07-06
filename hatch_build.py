import shutil
import tempfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

import python_minifier


class CustomBuildHook(BuildHookInterface):
    """Ships evalkit as minified source in the wheel — locals renamed,
    whitespace/docstrings/comments stripped — to keep the proprietary
    implementation out of a plain-text pip install. remove_annotations is
    forced off: @dataclass (used in core/context.py, core/types.py,
    simulation/types.py) reads __annotations__ at class-definition time, so
    stripping them would silently drop every dataclass field.

    The scratch copy MUST live outside self.directory — that resolves to the
    build target's own output dir (e.g. dist/), and anything left there gets
    swept up by naive `dist/*` globs (twine, gh-action-pypi-publish) as if it
    were a distribution file."""

    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return

        pkg_dir = Path(self.root) / "evalkit"
        self._scratch = Path(tempfile.mkdtemp(prefix="evalkit_minify_"))
        dst_pkg = self._scratch / "evalkit"
        shutil.copytree(pkg_dir, dst_pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

        for py_file in dst_pkg.rglob("*.py"):
            source = py_file.read_text()
            minified = python_minifier.minify(source, filename=str(py_file), remove_annotations=False)
            py_file.write_text(minified)

        for py_file in dst_pkg.rglob("*.py"):
            build_data["force_include"][str(py_file)] = str(py_file.relative_to(self._scratch))

    def finalize(self, version, build_data, artifact_path):
        if self.target_name != "wheel":
            return
        shutil.rmtree(self._scratch, ignore_errors=True)
