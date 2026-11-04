"""What the voice test modules share: the SDK root for subprocesses, the optional-dependency skip and
free localhost ports."""

from __future__ import annotations

import os
import socket
import sys
from typing import Dict, Iterable

import pytest

SDK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VOICE_EXTRA = ("numpy", "websockets", "webrtcvad")


def require(*modules: str) -> None:
    """Skip the calling module unless the voice extra's modules it needs are installed."""
    for name in modules or VOICE_EXTRA:
        pytest.importorskip(name)


def subprocess_env(extra: Dict[str, str] = None) -> Dict[str, str]:
    """The environment for a child Python that must import this source tree's evalkit, whatever is
    installed in site-packages."""
    env = dict(os.environ)
    env["PYTHONPATH"] = SDK_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("EVALKIT_API_KEY", None)
    env.pop("EVALKIT_SUBSCRIPTION_KEY", None)
    env.pop("EVALKIT_API_URL", None)
    env.update(extra or {})
    return env


def free_port(ports: Iterable[int]) -> int:
    for port in ports:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free localhost port in the range")


PYTHON = sys.executable
