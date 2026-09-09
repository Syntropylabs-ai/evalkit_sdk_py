import shutil
import tempfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

import python_minifier


class CustomBuildHook(BuildHookInterface):

    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return

        self._scratch = Path(tempfile.mkdtemp(prefix="evalkit_minify_"))
        for package in ("evalkit", "evalkit_cli"):
            dst_pkg = self._scratch / package
            shutil.copytree(Path(self.root) / package, dst_pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
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
