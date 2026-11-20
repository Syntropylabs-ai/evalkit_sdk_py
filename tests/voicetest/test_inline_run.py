"""Runs of inline scenarios (RUN_API section 3.1 ``scenarios``) end to end, against the fake RUN_API
server and the reference bot: from Python, and through ``evalkit voice-test init`` and
``run --scenarios`` with a YAML and a JSON file. Also the caller's voice from your own provider key,
which is read from the environment, sent in the create body only, and shown nowhere."""

from __future__ import annotations

import json
import logging
import subprocess
import sys

from .support import require

require()

import pytest  # noqa: E402

from evalkit.voicetest.net import RetryPolicy  # noqa: E402
from evalkit.voicetest.refbot import Behaviour, RefBotServer  # noqa: E402
from evalkit.voicetest.runner import run_async  # noqa: E402
from evalkit.voicetest.scenarios import starter_scenarios, with_max_turns  # noqa: E402

from .fake_api import KEY, NO_VOICE_KEY, FakeVoiceApi  # noqa: E402
from .support import SDK_ROOT, free_port, subprocess_env  # noqa: E402
from .test_runner import _refbot_process  # noqa: E402

PORTS = range(19720, 19760)
FAST = RetryPolicy(attempts=4, base_s=0.05, cap_s=0.2)
# Answers long enough for a line with interruptAfterBotMs to cut in on.
LONG_ANSWERS = {"default_turn": {"delay_ms": 400, "reply_ms": 1800}}
TTS_KEY = "sk-tts-test-0123456789abcdef"
CALLER_KEY = "AIza-caller-cli-test-0123456789"


def _blobs(fake):
    """Every request that reached the fake, apart from the create, as text."""
    out = []
    for r in fake.requests:
        if r["method"] == "POST" and r["path"] == "/v1/voice-tests":
            continue
        out.append(json.dumps(r["headers"]) + r["query"] + r["body"].decode("utf-8", "replace"))
    return out


def _cli(*args, env=None, timeout=180):
    return subprocess.run([sys.executable, "-m", "evalkit_cli.main", "voice-test", *args], capture_output=True,
                          text=True, cwd=SDK_ROOT, env=subprocess_env(env or {}), timeout=timeout)


def _popen(*args, env=None):
    return subprocess.Popen([sys.executable, "-m", "evalkit_cli.main", "voice-test", *args],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=SDK_ROOT,
                            env=subprocess_env(env or {}))


async def test_inline_scenarios_from_python_with_a_planned_interruption_and_your_own_tts_key(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    scenarios = [
        {"name": "duplicate charge", "successCriteria": "refunds the second charge",
         "lines": ["Hi, I was charged twice.", "Order 4471."]},
        {"name": "cuts in", "lines": ["Explain the refund policy.",
                                      {"text": "How long does it take?", "interruptAfterBotMs": 600}]},
    ]
    with FakeVoiceApi() as fake:
        port = free_port(PORTS)
        async with RefBotServer(Behaviour.from_json(LONG_ANSWERS), "pcm16", port=port):
            report = await run_async(
                f"ws://127.0.0.1:{port}/ws", format="pcm16", scenarios=scenarios,
                voice={"provider": "openai", "model": "tts-1", "voice": "alloy", "api_key": TTS_KEY},
                first_speaker="caller", subscription_key=KEY, api_url=fake.url, handle_signals=False,
                retry=FAST, wait_timeout_s=20, out_dir=str(tmp_path / "out"), cache_dir=str(tmp_path / "cache"))
        assert report.exit_code == 0, report.to_json()

        (created,) = fake.created
        assert "scenarioSetId" not in created and "scenarioIds" not in created
        assert created["scenarios"] == [
            {"name": "duplicate charge", "lines": [{"text": "Hi, I was charged twice."}, {"text": "Order 4471."}],
             "successCriteria": ["refunds the second charge"]},
            {"name": "cuts in", "lines": [{"text": "Explain the refund policy."},
                                          {"text": "How long does it take?", "interruptAfterBotMs": 600}]},
        ]
        assert created["voice"] == {"provider": "openai", "model": "tts-1", "voice": "alloy", "apiKey": TTS_KEY}
        assert created["target"]["firstSpeaker"] == "caller"
        calls = fake.plan["calls"]
        assert [c["scenarioId"] for c in calls] == ["scn_run000_0", "scn_run000_1"]
        assert calls[1]["sessionToken"] == "sdk:run000:scn_run000_1:1"
        assert calls[0]["script"]["lines"][0]["wait"] == {"rule": "fixed", "ms": 0}

        # the planned interruption was played and measured
        turns = fake.results[1]["scorecard"]["turns"]
        assert [t["interruption_planned"] for t in turns] == [False, True]
        assert turns[1]["interruption_stop_ms"] is not None and turns[1]["interruption_stop_ms"] < 700
        assert all(r["error"] is None for r in fake.results.values())

        # the provider key went in the create body and nowhere else
        assert not [b for b in _blobs(fake) if TTS_KEY in b]
        assert TTS_KEY not in json.dumps(fake.spans)
    assert TTS_KEY not in (tmp_path / "out" / report.run_id / "run.json").read_text()
    assert TTS_KEY not in caplog.text and TTS_KEY not in json.dumps(report.to_json(), default=str)


def test_cli_init_writes_a_starter_that_runs_from_yaml_with_both_keys_from_the_environment(tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "voice-tests.yaml"
    out = _cli("init", str(path))
    assert out.returncode == 0, out.stderr
    assert f"wrote {path}: 3 scenarios: 2 with a goal, 1 with 2 lines" in out.stdout and "--scenarios" in out.stdout
    assert "--about" in out.stdout
    again = _cli("init", str(path))
    assert again.returncode == 2 and "exists; pass --force to overwrite it" in again.stderr
    assert _cli("init", str(path), "--force").returncode == 0

    port = free_port(PORTS)
    bot, _ = _refbot_process("pcm16", port, tmp_path, behaviour=LONG_ANSWERS)
    try:
        with FakeVoiceApi(max_concurrent=3) as fake:
            proc = _popen("run", "--url", f"ws://127.0.0.1:{port}/", "--format", "pcm16", "--scenarios", str(path),
                          "--max-turns", "1", "--caller-provider", "google",
                          "--first-speaker", "caller", "--tts-provider", "openai", "--api-url", fake.url,
                          "--key", KEY, "--wait-timeout", "20", "--json", "--out", str(tmp_path / "keep"),
                          env={"OPENAI_API_KEY": TTS_KEY, "GOOGLE_API_KEY": CALLER_KEY, "HOME": str(tmp_path),
                               "EVALKIT_VOICETEST_CACHE": str(tmp_path / "cache")})
            stdout, stderr = proc.communicate(timeout=180)
            assert proc.returncode == 0, stdout + stderr
            body = json.loads(stdout)
            assert body["verdict"] == "passed" and len(body["calls"]) == 3

            (created,) = fake.created
            assert created["scenarios"] == with_max_turns(starter_scenarios(), 1) and "scenarioSetId" not in created
            # the provider alone: the API picks the model and voice
            assert created["voice"] == {"provider": "openai", "apiKey": TTS_KEY}
            assert created["caller"] == {"provider": "google", "apiKey": CALLER_KEY}
            calls = fake.plan["calls"]
            assert [c["script"].get("mode") for c in calls] == ["dynamic", "dynamic", None]
            # an interruption before turn 1 is written at create, as line 1
            cut_in = calls[0]["script"]["lines"][1]
            assert cut_in["interruptAfterBotMs"] == 1500
            assert [t["interruption_planned"] for t in fake.results[0]["scorecard"]["turns"]] == [False, True]
            asked = {i: [r["turn"] for r in fake.turn_requests if r["call"] == i] for i in range(3)}
            assert asked == {0: [2], 1: [1, 2], 2: []}  # turn 2 is turns.max + 1: the bot's last answer, heard
            assert all(r["keys"] == {"caller": CALLER_KEY, "voice": TTS_KEY} for r in fake.turn_requests)
            assert [c["mode"] for c in body["calls"]] == ["dynamic", "dynamic", "scripted"]
            assert [c["turnsTaken"] for c in body["calls"]] == [1, 1, None]
            assert body["calls"][1]["transcript"][-1]["role"] == "bot" and body["calls"][2]["transcript"] is None
            assert body["caller"]["keySource"] == "request"

            assert "the caller's voice is openai, synthesized with the key in OPENAI_API_KEY" in stderr
            assert "the caller's lines are decided by google, with the key in GOOGLE_API_KEY" in stderr
            for secret in (TTS_KEY, CALLER_KEY):
                assert secret not in stdout and secret not in stderr
                assert not [r["path"] for r in fake.requests if not r["path"].endswith("/turns")
                            and r["path"] != "/v1/voice-tests" and secret in r["body"].decode("utf-8", "replace")]
                assert secret not in (tmp_path / "keep" / body["runId"] / "run.json").read_text()
    finally:
        bot.terminate()
        bot.wait(timeout=10)


def test_cli_about_has_the_callers_written_and_prints_the_conversations(tmp_path):
    about = tmp_path / "bot.txt"
    about.write_text("A refund desk for an online shop. It finds orders and refunds duplicate charges.\n")
    port = free_port(PORTS)
    bot, _ = _refbot_process("pcm16", port, tmp_path)
    caller = {"provider": "google", "model": "gemini-3.8-flash", "keySource": "platform"}
    try:
        with FakeVoiceApi(caller=caller, platform_minutes_left=87) as fake:
            proc = _popen("run", "--url", f"ws://127.0.0.1:{port}/", "--format", "pcm16", "--about-file", str(about),
                          "--callers", "2", "--max-turns", "1", "--first-speaker", "caller", "--api-url", fake.url,
                          "--key", KEY, "--wait-timeout", "20",
                          env={"HOME": str(tmp_path), "EVALKIT_VOICETEST_CACHE": str(tmp_path / "cache")})
            stdout, stderr = proc.communicate(timeout=180)
            assert proc.returncode == 0, stdout + stderr
            (created,) = fake.created
            assert created["generate"] == {"about": about.read_text().strip(), "count": 2, "maxTurns": 1}
            assert "scenarios" not in created and "scenarioSetId" not in created and "caller" not in created
            assert "  conversation     1 turn after the opening" in stdout
            assert "    bot:    bot answer before turn 2" in stdout
            assert "caller model: google/gemini-3.8-flash (the SyntropyLabs model); 87 SyntropyLabs model minutes " \
                   "left this month, with this run's budget set aside" in stdout
            # Printed once, at the start: the closing block names the caller without the plan's figure.
            assert stdout.count("87 SyntropyLabs model minutes") == 1
            assert "task success:\n  call 0: 0.80 passed Met: the agent refunds the second charge." in stdout
    finally:
        bot.terminate()
        bot.wait(timeout=10)


def test_cli_runs_a_json_file_and_without_the_env_key_sends_none(tmp_path):
    path = tmp_path / "calls.json"
    path.write_text(json.dumps({"scenarios": [{"name": "order status", "lines": ["Where is my order?", "Thanks."]}]}))
    port = free_port(PORTS)
    bot, _ = _refbot_process("pcm16", port, tmp_path)
    try:
        with FakeVoiceApi() as fake:
            proc = _popen("run", "--url", f"ws://127.0.0.1:{port}/", "--format", "pcm16", "--scenarios", str(path),
                          "--tts-provider", "gemini", "--tts-voice", "Kore", "--api-url", fake.url, "--key", KEY, "--wait-timeout", "20",
                          env={"HOME": str(tmp_path), "EVALKIT_VOICETEST_CACHE": str(tmp_path / "cache"),
                               "GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""})
            stdout, stderr = proc.communicate(timeout=180)
            assert proc.returncode == 0, stdout + stderr
            assert "call 0: order status (repeat 0)" in stdout and "verdict: passed (exit 0)" in stdout
            (created,) = fake.created
            assert created["scenarios"] == [{"name": "order status",
                                             "lines": [{"text": "Where is my order?"}, {"text": "Thanks."}]}]
            # no key in the environment: the org's own stored key is asked for instead
            assert created["voice"] == {"provider": "gemini", "voice": "Kore"}
            assert ("GOOGLE_API_KEY and GEMINI_API_KEY are not set, so no key is sent; the organization's stored "
                    "gemini key will be used") in stderr
            # no --first-speaker: the plan's own default for line 0, a greeting heard out if it starts in 3 s
            assert "firstSpeaker" not in created["target"]
            assert fake.plan["calls"][0]["script"]["lines"][0]["wait"]["startWithinMs"] == 3000
    finally:
        bot.terminate()
        bot.wait(timeout=10)


def test_cli_prints_the_apis_guidance_when_it_refuses_the_run(tmp_path):
    path = tmp_path / "calls.json"
    path.write_text(json.dumps([{"name": "s", "lines": ["hello"]}]))
    with FakeVoiceApi(require_voice_key=True) as fake:
        out = _cli("run", "--url", "ws://127.0.0.1:9/", "--format", "pcm16", "--scenarios", str(path),
                   "--api-url", fake.url, "--key", KEY, env={"HOME": str(tmp_path)}, timeout=60)
    assert out.returncode == 2, out.stderr
    assert ("evalkit voice-test: the API refused to create the run (HTTP 400):\n  " + NO_VOICE_KEY) in out.stderr
    assert "Traceback" not in out.stderr and KEY not in out.stderr + out.stdout


@pytest.mark.parametrize("args, message", [
    (["--tts-provider", "openai", "--tts-key-env", TTS_KEY], "takes the name of an environment variable"),
    (["--tts-provider", "openai", "--tts-key", TTS_KEY], "takes the name of an environment variable"),
    (["--tts-api-key", TTS_KEY], "unrecognized arguments: --tts-api-key (values not shown)"),
    (["--tts-api-key=" + TTS_KEY], "unrecognized arguments: --tts-api-key (values not shown)"),
    (["--tts-model", "tts-1"], "--tts-model needs --tts-provider"),
    (["--tts-provider", "openai", "--voice", "openai:tts-1:alloy"], "either --voice or --tts-provider"),
    (["--tts-provider", TTS_KEY], "--tts-provider takes a provider name"),
    (["--scenario", "a"], "--scenario picks from a scenario set"),
    (["--callers", "3"], "--callers goes with --about or --about-file"),
    (["--caller-key-env", CALLER_KEY], "takes the name of an environment variable"),
    (["--caller-provider", CALLER_KEY], "--caller-provider takes a provider name"),
    (["--caller-api-key", CALLER_KEY], "unrecognized arguments: --caller-api-key (values not shown)"),
    (["--max-turns", "21"], "--max-turns takes 1 to 20"),
    (["--about", "a refund desk"], "not allowed with argument"),
])
def test_cli_refuses_a_key_on_the_command_line_and_bad_combinations(tmp_path, args, message):
    path = tmp_path / "calls.json"
    path.write_text(json.dumps([{"name": "s", "lines": ["hello"]}]))
    out = _cli("run", "--url", "ws://127.0.0.1:9/", "--format", "pcm16", "--scenarios", str(path),
               "--api-url", "http://127.0.0.1:9", "--key", KEY, *args, env={"HOME": str(tmp_path)}, timeout=60)
    assert out.returncode == 2, out.stderr
    assert message in out.stderr
    assert TTS_KEY not in out.stderr + out.stdout and CALLER_KEY not in out.stderr + out.stdout


@pytest.mark.parametrize("args, message", [
    ([], "say who calls your bot: --about \"what your bot does\""),
    (["--about", "  "], "--about needs a few sentences"),
    (["--about", "a refund desk", "--callers", "21"], "--callers takes 1 to 20"),
    (["--about", "a refund desk", "--scenario", "a"], "--scenario picks from a scenario set"),
    (["--about-file", "/nonexistent/bot.txt"], "--about-file /nonexistent/bot.txt: No such file"),
    (["--scenario-set", "s1", "--max-turns", "2"], "--max-turns applies to --about and --scenarios"),
])
def test_cli_needs_exactly_one_source_of_callers(tmp_path, args, message):
    out = _cli("run", "--url", "ws://127.0.0.1:9/", "--format", "pcm16", "--api-url", "http://127.0.0.1:9",
               "--key", KEY, *args, env={"HOME": str(tmp_path)}, timeout=60)
    assert out.returncode == 2, out.stderr
    assert message in out.stderr and "Traceback" not in out.stderr


def test_cli_names_the_file_line_that_breaks_a_limit(tmp_path):
    path = tmp_path / "calls.json"
    path.write_text(json.dumps([{"name": "long", "lines": ["ok", "x" * 501]}]))
    out = _cli("run", "--url", "ws://127.0.0.1:9/", "--format", "pcm16", "--scenarios", str(path),
               "--key", KEY, env={"HOME": str(tmp_path)}, timeout=60)
    assert out.returncode == 2
    assert f'{path}: scenarios[0] ("long"): lines[1]: the text is 501 characters' in out.stderr
    both = _cli("run", "--url", "ws://127.0.0.1:9/", "--format", "pcm16", "--scenarios", str(path),
                "--scenario-set", "s1", env={"HOME": str(tmp_path)}, timeout=60)
    assert both.returncode == 2 and "not allowed with argument" in both.stderr


def test_about_is_refused_past_the_servers_limit_before_any_request():
    from evalkit.voicetest.runner import MAX_ABOUT_CHARS, VoiceTestSetupError, _generate_spec

    assert MAX_ABOUT_CHARS == 4_000
    assert _generate_spec("x" * MAX_ABOUT_CHARS, None, None)["count"] == 5
    with pytest.raises(VoiceTestSetupError, match="at most 4,000"):
        _generate_spec("x" * (MAX_ABOUT_CHARS + 1), None, None)
