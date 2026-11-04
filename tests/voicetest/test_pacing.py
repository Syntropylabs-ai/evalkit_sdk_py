"""The pacing budget: late-frame statistics from a 60 s call against the reference bot running in
its own process. Slow, so it runs only with EVALKIT_VOICETEST_PACING=1; the statistics are printed and
written to pacing_60s.json in the test's temporary directory."""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time

from .support import SDK_ROOT, require, subprocess_env

require()

import pytest

from evalkit.voicetest.caller import run_call
from evalkit.voicetest.script import parse_script
from evalkit.voicetest.transport import Target

pytestmark = pytest.mark.skipif(
    os.environ.get("EVALKIT_VOICETEST_PACING") != "1", reason="set EVALKIT_VOICETEST_PACING=1 to run"
)


def _port() -> int:
    for port in range(19540, 19550):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port in 19540-19549")


@pytest.mark.parametrize("fmt,rate", [("twilio", 8000), ("pcm16", 16000)])
async def test_pacing_over_a_60s_call(fmt, rate, tmp_path):
    behaviour = tmp_path / "behaviour.json"
    behaviour.write_text(json.dumps({
        "turns": [{"delay_ms": 700, "reply_ms": 1800, "mode": "burst" if i % 3 == 1 else "realtime"}
                  for i in range(40)],
    }))
    port = _port()
    bot = subprocess.Popen(
        [sys.executable, "-m", "evalkit.voicetest", "refbot", "--format", fmt, "--port", str(port),
         "--behaviour", str(behaviour), "--sample-rate", str(rate)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=SDK_ROOT, env=subprocess_env(),
    )
    try:
        for _ in range(100):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.05)
        lines = [{"text": f"line {i}", "synth": {"ms": 1500, "seed": i}} for i in range(40)]
        lines[0]["wait"] = {"mode": "fixed", "ms": 300}
        script = parse_script({"lines": lines, "max_call_ms": 60000})
        result = await run_call(Target(url=f"ws://127.0.0.1:{port}/", format=fmt, sample_rate=rate),
                                script, str(tmp_path / "call"))
    finally:
        bot.terminate()
        bot.wait(timeout=10)
    stats = dict(result.pacing, format=fmt, duration_ms=result.duration_ms, ended_by=result.ended_by)
    (tmp_path / "pacing_60s.json").write_text(json.dumps(stats, indent=2))
    print("\npacing", json.dumps(stats))
    assert result.ended_by == "max_call" and result.duration_ms == 60000
    assert stats["frames"] == 3000
    await asyncio.sleep(0)
