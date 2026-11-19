"""The SDK run end to end: the runner against the fake RUN_API server and the reference bot, over
real WebSockets on localhost, in-process and through the ``evalkit voice-test`` command."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import os
import signal
import subprocess
import sys
import time
import wave

from .support import require

require()

import pytest  # noqa: E402

from evalkit.voicetest.net import RetryPolicy  # noqa: E402
from evalkit.voicetest.planned_call import run_planned_call  # noqa: E402
from evalkit.voicetest.refbot import Behaviour, RefBotServer  # noqa: E402
from evalkit.voicetest.runner import VoiceTestSetupError, run_async  # noqa: E402
from evalkit.voicetest.transport import Target  # noqa: E402

from .fake_api import KEY, FakeVoiceApi, default_scenarios  # noqa: E402
from .support import SDK_ROOT, free_port, subprocess_env  # noqa: E402

PORTS = range(19650, 19690)
FAST = RetryPolicy(attempts=4, base_s=0.05, cap_s=0.2)
BEHAVIOUR = {"default_turn": {"delay_ms": 500, "reply_ms": 700}}
BOT_HEADER = ("X-Bot-Secret", "hunter2")
BOT_QUERY = "token=botsecret"


def _port() -> int:
    return free_port(PORTS)


async def _run(fake, tmp_path, fmt="pcm16", rate=16000, url=None, **kwargs):
    kwargs.setdefault("out_dir", str(tmp_path / "out"))
    kwargs.setdefault("cache_dir", str(tmp_path / "cache"))
    return await run_async(url, format=fmt, sample_rate=rate, headers=dict([BOT_HEADER]),
                           scenario_set="set-1", subscription_key=KEY, api_url=fake.url,
                           handle_signals=False, retry=FAST, wait_timeout_s=20, **kwargs)


async def _run_against_refbot(fake, tmp_path, fmt="pcm16", rate=16000, **kwargs):
    port = _port()
    async with RefBotServer(Behaviour.from_json(BEHAVIOUR), fmt, port=port, sample_rate=rate):
        return await _run(fake, tmp_path, fmt, rate, url=f"ws://127.0.0.1:{port}/ws?{BOT_QUERY}", **kwargs)


def _wav(data: bytes):
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()


def _api_requests(fake):
    return [r for r in fake.requests if not r["path"].startswith("/storage/")]


def _assert_bot_secrets_stayed_with_the_bot(fake):
    for r in _api_requests(fake):
        blob = json.dumps(r["headers"]) + r["query"] + r["body"].decode("utf-8", "replace")
        assert "hunter2" not in blob and "botsecret" not in blob and "x-bot-secret" not in blob.lower(), r["path"]


async def test_a_run_end_to_end(tmp_path):
    with FakeVoiceApi() as fake:
        report = await _run_against_refbot(fake, tmp_path)
        assert report.exit_code == 0 and report.verdict == "passed", report.to_json()

        # the run was created with a label for the target, and nothing of the bot's credentials
        (created,) = fake.created
        assert created["target"]["url"].startswith("ws://127.0.0.1:") and created["target"]["url"].endswith("/ws")
        assert created["target"]["format"] == "pcm16" and created["target"]["sampleRate"] == 16000
        assert created["scenarioSetId"] == "set-1" and created["repeats"] == 1
        assert created["timingOnly"] is False
        assert "firstSpeaker" not in created["target"]  # not given: the API's default applies
        _assert_bot_secrets_stayed_with_the_bot(fake)

        # each clip downloaded once, from storage, without our key (storage would have refused it)
        assert sorted(fake.downloads.values()) == [1, 1, 1, 1]
        # every call ran, was recorded, uploaded and reported
        assert sorted(fake.results) == [0, 1] and fake.result_posts == 2
        plan_calls = {c["callIndex"]: c for c in fake.plan["calls"]}
        for index, result in fake.results.items():
            assert result["error"] is None, result["error"]
            card = result["scorecard"]
            assert card["schema"] == "evalkit.voice.scorecard.v1"
            assert card["call"]["session"] == plan_calls[index]["sessionToken"]
            assert result["vadVersion"] == "webrtc-v2" and result["callerVersion"]
            assert result["callerHealthy"] in (True, False)
            for turn in card["turns"]:
                assert turn["response_delay_ms"] == pytest.approx(500, abs=60)
            upload, headers = fake.uploads[(report.run_id, index)]
            assert _wav(upload)[:3] == (2, 2, 16000)
            assert result["recording"] == {"objectPath": f"traces/tenant1/audio/{report.run_id}/{index}.wav",
                                           "bytes": len(upload), "sha256": hashlib.sha256(upload).hexdigest()}
            assert headers["x-goog-meta-run"] == report.run_id  # the API's upload headers were sent as given
        # the voice.call traces went through ingest, joined to the call's session
        roots = [s for s in fake.spans if s.get("operation") == "voice.call"]
        assert sorted(s["traceId"] for s in roots) == sorted(r["traceId"] for r in fake.results.values())
        for root in roots:
            token = root["sessionId"]
            assert root["attributes"]["evalkit.voice.v1.session"] == token
            assert root["attributes"]["evalkit.voice.v1.recording_uri"].startswith("traces/tenant1/audio/")
        turns = [s for s in fake.spans if s.get("operation") == "voice.turn"]
        assert len(turns) == 4
        assert fake.completed == 1 and fake.polls >= 1
        paths = fake.paths("/v1/voice-tests")
        assert paths.index(f"POST /v1/voice-tests/{report.run_id}/complete") > max(
            i for i, p in enumerate(paths) if p.endswith("/result"))

    # kept locally, with the run's own summary
    root = tmp_path / "out" / report.run_id
    assert (root / "call-000" / "recording.wav").exists() and (root / "call-001" / "scorecard.json").exists()
    summary = json.loads((root / "run.json").read_text())
    assert summary["verdict"] == "passed" and len(summary["calls"]) == 2
    call_json = json.loads((root / "call-000" / "call.json").read_text())
    assert "s3cr3tsig" not in json.dumps(call_json) and "hunter2" not in json.dumps(call_json)


async def test_timing_only_uploads_nothing_and_api_errors_are_retried(tmp_path):
    fails = {
        r"POST /v1/voice-tests": 503,
        r"GET /storage/audio/.*": 500,
        r"POST /v1/voice-tests/run000/calls/0/result": 502,
        r"POST /v1/voice-tests/run000/complete": 504,
        r"GET /v1/voice-tests/run000": 503,
    }
    with FakeVoiceApi(default_scenarios()[:1], fail_once=fails) as fake:
        report = await _run_against_refbot(fake, tmp_path, timing_only=True, out_dir=None)
        assert report.exit_code == 0, report.to_json()
        assert not fake.upload_urls and not fake.uploads
        # the server is told, so it refuses recordings for the run and shows it as timing-only
        assert fake.created and all(c["timingOnly"] is True for c in fake.created)
        (result,) = fake.results.values()
        assert result["recording"] is None and result["scorecard"]["turns"]
        (root,) = [s for s in fake.spans if s.get("operation") == "voice.call"]
        assert "evalkit.voice.v1.recording_uri" not in root["attributes"]
        for pattern in fails:
            assert len(fake.requests_to(pattern)) >= 2, pattern
        assert report.out_dir is None  # nothing kept without --out


async def test_calls_that_fail_are_reported_and_the_run_goes_on(tmp_path):
    def all_ok(results):
        return all(r["error"] is None for r in results.values())

    with FakeVoiceApi(passed=all_ok) as fake:
        dead = _port()  # nothing listens here
        report = await _run(fake, tmp_path, url=f"ws://127.0.0.1:{dead}/")
        assert [c.error["kind"] for c in report.calls] == ["connect_failed", "connect_failed"]
        assert sorted(fake.results) == [0, 1]
        for result in fake.results.values():
            assert result["error"]["kind"] == "connect_failed" and result["scorecard"] is None
            assert result["recording"] is None and result["traceId"] is None
        assert not fake.upload_urls and fake.completed == 1
        assert report.passed is False and report.exit_code == 1


async def test_audio_that_fails_its_sha256_fails_its_call_as_a_setup_failure(tmp_path):
    with FakeVoiceApi(corrupt=("Thanks.",)) as fake:
        report = await _run_against_refbot(fake, tmp_path)
        kinds = {c.call_index: (c.error or {}).get("kind") for c in report.calls}
        assert kinds == {0: None, 1: "audio_unavailable"}
        assert "sha256" in fake.results[1]["error"]["message"]
        assert report.exit_code == 2 and report.verdict == "error"
        assert not [f for f in os.listdir(str(tmp_path / "cache")) if f.endswith(".part")]


async def test_setup_failures_raise_before_any_call(tmp_path, monkeypatch):
    with FakeVoiceApi(create_status=404) as fake:
        with pytest.raises(VoiceTestSetupError, match="scenario set sc-missing not found") as err:
            await _run(fake, tmp_path, url="ws://127.0.0.1:9/")
        assert KEY not in str(err.value)
        with pytest.raises(VoiceTestSetupError, match="ws:// or wss://"):
            await _run(fake, tmp_path, url="http://127.0.0.1:9/")
        with pytest.raises(VoiceTestSetupError, match="first_speaker must be one of auto, bot, caller"):
            await _run(fake, tmp_path, url="ws://127.0.0.1:9/", first_speaker="nobody")
        assert len(fake.requests) == 1  # the bad options were refused before the API was asked
    for name in ("EVALKIT_API_KEY", "EVALKIT_SUBSCRIPTION_KEY"):
        monkeypatch.delenv(name, raising=False)
    import evalkit

    monkeypatch.setattr(evalkit, "_instance", None)
    with pytest.raises(VoiceTestSetupError, match="subscription key"):
        await run_async("ws://127.0.0.1:9/", format="pcm16", scenario_set="s", api_url="http://127.0.0.1:9",
                        handle_signals=False)


async def test_with_one_slot_the_calls_start_in_plan_order_whichever_audio_is_ready_first(tmp_path,
                                                                                        monkeypatch):
    import evalkit.voicetest.planned_call as planned_call

    real = planned_call.prepare_script_audio

    def first_comes_back_last(script, sources, rate):
        if script.lines[0].text == "first":
            time.sleep(0.5)  # a worker thread: call 0's audio is ready after call 1's
        return real(script, sources, rate)

    monkeypatch.setattr(planned_call, "prepare_script_audio", first_comes_back_last)
    scenarios = [{"id": f"sc-{text}", "name": text,
                  "lines": [{"text": text, "ms": 300, "seed": seed, "wait": {"rule": "fixed", "ms": 0}}]}
                 for seed, text in enumerate(("first", "second"), 1)]
    with FakeVoiceApi(scenarios, max_concurrent=1) as fake:
        report = await _run_against_refbot(fake, tmp_path)
    assert [c.error for c in report.calls] == [None, None], report.to_json()
    root = tmp_path / "out" / report.run_id
    starts = [json.loads((root / f"call-{i:03d}" / "call.json").read_text())["t0_epoch_ms"] for i in (0, 1)]
    assert starts[0] < starts[1]


# One planned call: the error kinds ---------------------------------------------------------------


def _planned(index=0, lines=None, max_call_ms=30000):
    lines = lines or [{"text": "hello", "synth": {"ms": 600, "seed": 1}, "wait": {"rule": "fixed", "ms": 200}},
                      {"text": "again", "synth": {"ms": 600, "seed": 2}}]
    return {"callIndex": index, "sessionToken": f"sdk:r:s:{index}", "scenarioName": "t",
            "script": {"maxCallMs": max_call_ms, "lines": lines,
                       "end": {"afterLastLine": "bot_done", "silenceMs": 500, "maxMs": 4000}}}


async def test_a_bot_that_hangs_up_early_is_reported_with_what_was_recorded(tmp_path):
    from websockets.asyncio.server import serve

    async def hang_up(ws):
        await asyncio.sleep(0.5)
        await ws.close(code=1011, reason="crashed")

    port = _port()
    with FakeVoiceApi() as fake:
        from evalkit.voicetest.api import VoiceTestApi

        api = VoiceTestApi.with_subscription_key(fake.url, KEY, retry=FAST)
        async with serve(hang_up, "127.0.0.1", port):
            outcome = await run_planned_call(_planned(), run_id="r", api=api, out_dir=str(tmp_path / "c"),
                                             target=Target(url=f"ws://127.0.0.1:{port}/", format="pcm16"))
        assert outcome.error["kind"] == "bot_closed_early" and "1011" in outcome.error["message"]
        assert outcome.scorecard is not None and outcome.reported
        assert fake.results[0]["error"]["kind"] == "bot_closed_early"
        assert fake.results[0]["recording"]["bytes"] > 0  # what was recorded is still uploaded


async def test_a_call_over_max_call_ms_is_an_error(tmp_path):
    port = _port()
    async with RefBotServer(Behaviour.from_json(BEHAVIOUR), "pcm16", port=port):
        outcome = await run_planned_call(_planned(), run_id="r", out_dir=str(tmp_path / "c"),
                                         target=Target(url=f"ws://127.0.0.1:{port}/", format="pcm16"),
                                         limits={"maxCallMs": 1500})
    assert outcome.error["kind"] == "max_call" and outcome.ended_by == "max_call"
    assert outcome.duration_ms == 1500 and outcome.scorecard is not None


async def test_a_connector_names_why_it_refused(tmp_path):
    class GuardRefused(Exception):
        kind = "guard_refused"

    class Odd(Exception):
        kind = "not_a_kind"

    async def guard(url, **kwargs):
        raise GuardRefused("10.0.0.5 is a private address")

    async def odd(url, **kwargs):
        raise Odd("no")

    target = Target(url="ws://bot.internal/", format="pcm16")
    outcome = await run_planned_call(_planned(), run_id="r", out_dir=str(tmp_path / "a"), target=target,
                                     connector=guard)
    assert outcome.error["kind"] == "guard_refused" and "private address" in outcome.error["message"]
    outcome = await run_planned_call(_planned(), run_id="r", out_dir=str(tmp_path / "b"), target=target,
                                     connector=odd)
    assert outcome.error["kind"] == "connect_failed"


async def test_a_passthrough_connector_carries_a_whole_call(tmp_path):
    from websockets.asyncio.client import connect

    seen = []

    async def connector(url, **kwargs):
        seen.append((url, kwargs))
        return await connect(url, **kwargs)

    port = _port()
    async with RefBotServer(Behaviour.from_json(BEHAVIOUR), "twilio", port=port):
        outcome = await run_planned_call(
            _planned(), run_id="r", out_dir=str(tmp_path / "c"), connector=connector,
            target=Target(url=f"ws://127.0.0.1:{port}/", format="twilio", headers={"A": "b"}))
    assert outcome.error is None and outcome.scorecard["call"]["transport"] == "twilio"
    ((url, kwargs),) = seen
    assert url.endswith("evalkit_session=sdk%3Ar%3As%3A0") and kwargs["additional_headers"] == {"A": "b"}


# The command line --------------------------------------------------------------------------------


def _refbot_process(fmt, port, tmp_path, behaviour=BEHAVIOUR):
    path = tmp_path / f"behaviour-{port}.json"
    path.write_text(json.dumps(behaviour))
    truth = tmp_path / f"truth-{port}.jsonl"
    proc = subprocess.Popen(
        [sys.executable, "-m", "evalkit.voicetest", "refbot", "--format", fmt, "--port", str(port),
         "--behaviour", str(path), "--truth-out", str(truth)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=SDK_ROOT, env=subprocess_env())
    import socket

    for _ in range(200):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.05)
    return proc, truth


def _cli(fake, url, fmt, tmp_path, *extra):
    cmd = [sys.executable, "-m", "evalkit_cli.main", "voice-test", "-v", "run", "--url", url,
           "--format", fmt, "--header", "=".join(BOT_HEADER), "--scenario-set", "set-1",
           "--api-url", fake.url, "--key", KEY, "--wait-timeout", "20", *extra]
    env = subprocess_env({"EVALKIT_VOICETEST_CACHE": str(tmp_path / "cache"), "HOME": str(tmp_path)})
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=SDK_ROOT,
                            env=env)


def test_cli_run_over_twilio_exits_1_when_the_run_fails(tmp_path):
    port = _port()
    bot, _ = _refbot_process("twilio", port, tmp_path)
    try:
        with FakeVoiceApi(encoding="wav", passed=False) as fake:
            proc = _cli(fake, f"ws://127.0.0.1:{port}/?{BOT_QUERY}", "twilio", tmp_path, "--json",
                        "--first-speaker", "bot")
            out, err = proc.communicate(timeout=120)
            assert proc.returncode == 1, err
            body = json.loads(out)
            assert body["verdict"] == "failed" and body["exitCode"] == 1 and body["passed"] is False
            assert [c["error"] for c in body["calls"]] == [None, None]
            assert all(c["figures"]["response_delay_p50_ms"] is not None for c in body["calls"])
            assert fake.created[0]["target"] == {"url": f"ws://127.0.0.1:{port}/", "format": "twilio",
                                                 "sampleRate": 8000, "firstSpeaker": "bot"}
            assert len(fake.uploads) == 2 and fake.completed == 1
            assert all(r["scorecard"]["call"]["transport"] == "twilio" for r in fake.results.values())
            _assert_bot_secrets_stayed_with_the_bot(fake)
            assert KEY not in out and KEY not in err  # not even with -v
    finally:
        bot.terminate()
        bot.wait(timeout=10)


def test_cli_run_prints_a_summary_and_exits_0_when_it_passes(tmp_path):
    port = _port()
    bot, _ = _refbot_process("pcm16", port, tmp_path)
    try:
        with FakeVoiceApi(default_scenarios()[:1]) as fake:
            proc = _cli(fake, f"ws://127.0.0.1:{port}/", "pcm16", tmp_path, "--out", str(tmp_path / "keep"))
            out, err = proc.communicate(timeout=120)
            assert proc.returncode == 0, out + err
            assert "call 0: refund for a duplicate charge (repeat 0)" in out
            # The figure is measured live against the refbot, so a busy machine moves it; its accuracy
            # is the calibration's to prove, and this test checks the summary's shape.
            assert re.search(r"response delay   p50 \d[\d,]* ms", out) and "talk-over        none" in out
            assert "caller           healthy" in out and "recording        uploaded" in out
            assert "voice_response_delay  1.00  passed  p95 840 ms, limit 1,200 ms, 0 of 2 turns over" in out
            assert "console: https://console.example/voice/run000" in out
            assert out.rstrip().endswith("verdict: passed (exit 0)")
            assert (tmp_path / "keep" / "run000" / "call-000" / "recording.wav").exists()
            assert KEY not in out and KEY not in err
    finally:
        bot.terminate()
        bot.wait(timeout=10)


@pytest.mark.parametrize("args, message", [
    (["--key", KEY], "scenario set sc-missing not found"),
    (["--key", "tk_wrong"], "Invalid subscription key"),
    ([], "a subscription key is required"),
])
def test_cli_setup_failures_exit_2(tmp_path, args, message):
    with FakeVoiceApi(create_status=404) as fake:
        env = subprocess_env({"HOME": str(tmp_path)})
        out = subprocess.run(
            [sys.executable, "-m", "evalkit_cli.main", "voice-test", "run", "--url", "ws://127.0.0.1:9/",
             "--format", "pcm16", "--scenario-set", "s", "--api-url", fake.url, *args],
            capture_output=True, text=True, cwd=SDK_ROOT, env=env, timeout=60)
    assert out.returncode == 2, out.stderr
    assert message in out.stderr and KEY not in out.stderr + out.stdout


def _wait_for_connect(truth, timeout=30.0):
    """Until the reference bot's truth log shows the first call connected."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if truth.exists():
            for line in truth.read_text().splitlines():
                try:
                    if json.loads(line).get("type") == "connect":
                        return
                except ValueError:
                    pass  # a line still being written
        time.sleep(0.05)
    raise AssertionError("the caller never connected")


@pytest.mark.parametrize("presses", [1, 2])
def test_ctrl_c_stops_new_calls_and_still_reports(tmp_path, presses):
    port = _port()
    bot, truth = _refbot_process("pcm16", port, tmp_path)
    scenarios = [dict(s, id=f"{s['id']}-{k}") for k, s in enumerate(default_scenarios() + default_scenarios()[:1])]
    try:
        with FakeVoiceApi(scenarios, max_concurrent=1) as fake:
            proc = _cli(fake, f"ws://127.0.0.1:{port}/", "pcm16", tmp_path, "--json")
            _wait_for_connect(truth)
            time.sleep(0.8)
            proc.send_signal(signal.SIGINT)
            if presses == 2:
                time.sleep(0.5)
                proc.send_signal(signal.SIGINT)
            out, err = proc.communicate(timeout=60)
            assert proc.returncode == 130, err
            body = json.loads(out)
            assert body["interrupted"] is True and body["verdict"] == "interrupted"
            kinds = [(fake.results[i]["error"] or {}).get("kind") for i in range(3)]
            first = None if presses == 1 else "aborted"
            assert kinds == [first, "not_run", "not_run"], kinds
            # the call that ran is reported with its recording, finished or cut short
            assert fake.results[0]["scorecard"]["turns"] and (0 in [i for _, i in fake.uploads])
            if presses == 2:
                assert fake.results[0]["scorecard"]["call"]["duration_ms"] < 4000
            assert fake.completed == 1 and fake.polls == 0
    finally:
        bot.terminate()
        bot.wait(timeout=10)


# check -------------------------------------------------------------------------------------------


def test_check_against_the_reference_bot(tmp_path):
    from evalkit.voicetest.cli import main

    port = _port()
    bot, _ = _refbot_process("pcm16", port, tmp_path)
    try:
        buf = io.StringIO()
        assert main(["check", "--url", f"ws://127.0.0.1:{port}/", "--format", "pcm16", "--json"], out=buf) == 0
        found = json.loads(buf.getvalue())
        assert found["connected"] and found["answered"] and found["formatErrors"] == []
        assert found["responseDelayMs"] == pytest.approx(500, abs=60)
    finally:
        bot.terminate()
        bot.wait(timeout=10)


def test_check_waits_before_its_line_as_the_first_speaker_says():
    from evalkit.voicetest.check import LINE_MS, check_script
    from evalkit.voicetest.cli import build_parser

    auto, caller = check_script(8000), check_script(8000, "caller")
    assert auto["lines"][0]["wait"] == {"mode": "bot_done", "start_within_ms": 3000, "silence_ms": 700,
                                        "max_ms": 15000}
    assert caller["lines"][0]["wait"] == {"mode": "fixed", "ms": 0}
    assert caller["max_call_ms"] == LINE_MS + 8000 + 2000
    assert auto["max_call_ms"] == 15000 + LINE_MS + 8000 + 2000
    parser = build_parser()
    commands = {"check": ["check", "--url", "ws://x/"],
                "run": ["run", "--url", "ws://x/", "--format", "pcm16", "--scenario-set", "s"]}
    for base in commands.values():
        assert parser.parse_args(base + ["--first-speaker", "caller"]).first_speaker == "caller"
        with pytest.raises(SystemExit):
            parser.parse_args(base + ["--first-speaker", "nobody"])
    assert parser.parse_args(commands["check"]).first_speaker == "auto"
    assert parser.parse_args(commands["run"]).first_speaker is None  # left to the API's default


@pytest.mark.parametrize("bot_format,check_format,first_hint", [
    ("twilio", "pcm16", "nothing usable came back in pcm16 at 16000 Hz: try --format twilio, or another "
                        "--sample-rate (8000 or 24000); then check the endpoint path (/media)"),
    ("pcm16", "twilio", "nothing usable came back in the twilio format: try --format pcm16 with "
                        "--sample-rate 8000, 16000 or 24000; then check the endpoint path (/media)"),
])
async def test_check_in_the_wrong_format_suggests_the_other_format_first(bot_format, check_format, first_hint):
    from evalkit.voicetest.check import check_async

    # The reference bot, like many bots, waits for a start message of its own format and sends nothing.
    port = _port()
    async with RefBotServer(Behaviour.from_json(BEHAVIOUR), bot_format, port=port, sample_rate=16000):
        found = await check_async(Target(url=f"ws://127.0.0.1:{port}/media?{BOT_QUERY}", format=check_format),
                                  answer_timeout_ms=1000, first_speaker="caller")
    assert found["connected"] and not found["ok"] and found["botChunks"] == 0
    assert found["hints"][0] == first_hint
    assert found["hints"][1].startswith("the bot sent no audio: if the format and path are right")
    assert all("botsecret" not in hint for hint in found["hints"])


async def test_check_names_a_format_mix_up_and_an_unreachable_bot(tmp_path):
    from websockets.asyncio.server import serve

    from evalkit.voicetest.check import check_async

    async def twilio_like(ws):
        await ws.send(json.dumps({"event": "media", "media": {"payload": "AAAA"}}))
        async for _ in ws:
            pass

    port = _port()
    async with serve(twilio_like, "127.0.0.1", port):
        found = await check_async(Target(url=f"ws://127.0.0.1:{port}/", format="pcm16"), answer_timeout_ms=1000)
    assert found["connected"] and not found["ok"] and not found["answered"]
    assert found["formatErrors"] == [{"detail": "event 'media'", "count": 1}]
    assert any("--format twilio" in h for h in found["hints"])
    found = await check_async(Target(url=f"ws://127.0.0.1:{_port()}/", format="pcm16", open_timeout_s=2))
    assert not found["connected"] and not found["ok"] and found["connectError"]
