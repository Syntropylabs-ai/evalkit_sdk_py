"""The voice test caller ships as an optional extra: importing evalkit, or any voicetest module, must
load no optional dependency, start nothing and print nothing."""

import subprocess
import sys

from .support import SDK_ROOT, subprocess_env

MODULES = [
    "evalkit",
    "evalkit.voicetest",
    "evalkit.voicetest.audio",
    "evalkit.voicetest.audio_cache",
    "evalkit.voicetest.api",
    "evalkit.voicetest.net",
    "evalkit.voicetest.plan",
    "evalkit.voicetest.planned_call",
    "evalkit.voicetest.runner",
    "evalkit.voicetest.check",
    "evalkit.voicetest.summary",
    "evalkit.voicetest.transport",
    "evalkit.voicetest.transport.playback",
    "evalkit.voicetest.transport.twilio",
    "evalkit.voicetest.transport.pcm16",
    "evalkit.voicetest.recording",
    "evalkit.voicetest.script",
    "evalkit.voicetest.scenarios",
    "evalkit.voicetest.sources",
    "evalkit.voicetest.turns",
    "evalkit.voicetest.caller",
    "evalkit.voicetest.refbot",
    "evalkit.voicetest.analysis",
    "evalkit.voicetest.trace",
    "evalkit.voicetest.cli",
    "evalkit_cli.main",
]


def _run(code):
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=SDK_ROOT,
                         env=subprocess_env())
    assert out.returncode == 0, out.stderr
    assert out.stdout == "" and out.stderr == ""


def test_importing_loads_no_optional_dependency():
    _run(
        "import sys, threading\n"
        f"for m in {MODULES!r}: __import__(m)\n"
        "heavy = sorted(m for m in ('numpy', 'websockets', 'webrtcvad', 'yaml') if m in sys.modules)\n"
        "assert not heavy, heavy\n"
        "assert threading.active_count() == 1\n"
    )


def test_import_evalkit_leaves_voicetest_unloaded_until_used():
    _run(
        "import sys\n"
        "import evalkit\n"
        "assert 'evalkit.voicetest' not in sys.modules\n"
        "assert evalkit.voicetest.run.__module__ == 'evalkit.voicetest.runner'\n"
        "assert callable(evalkit.voicetest.run_planned_call)\n"
        "assert callable(evalkit.voicetest.script_from_plan_call)\n"
        "heavy = sorted(m for m in ('numpy', 'websockets', 'webrtcvad') if m in sys.modules)\n"
        "assert not heavy, heavy\n"
    )
