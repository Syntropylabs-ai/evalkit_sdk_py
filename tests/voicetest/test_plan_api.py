"""The plan (RUN_API section 3.3) as the caller's script, the audio cache, the API client and the
Target connector, without a live call."""

from __future__ import annotations

import asyncio
import hashlib
import os

import pytest

from evalkit.voicetest.api import VoiceTestApi, VoiceTestApiError
from evalkit.voicetest.net import HttpError, RetryPolicy
from evalkit.voicetest.plan import (
    PlanError,
    parse_plan,
    script_from_plan_call,
    target_from_settings,
    target_label,
)
from evalkit.voicetest.script import ScriptError

from .fake_api import KEY, FakeVoiceApi

FAST = RetryPolicy(attempts=3, base_s=0.01, cap_s=0.02)

# RUN_API section 3.3, as written there.
PLAN = {
    "runId": "r1", "consoleUrl": "https://syntropylabs.ai/x",
    "limits": {"maxCallMs": 600000, "maxConcurrentCalls": 2},
    "calls": [{
        "callIndex": 0, "scenarioId": "s1", "scenarioName": "refund for a duplicate charge", "repeat": 0,
        "sessionToken": "sdk:r1:s1:0",
        "script": {
            "maxCallMs": 300000,
            "lines": [{
                "text": "Hi, I was charged twice for my order.",
                "audio": {"url": "https://storage.example/a.raw?X-Goog-Signature=abc", "encoding": "pcm16",
                          "sampleRate": 16000, "sha256": "ab" * 32, "durationMs": 2380},
                "wait": {"rule": "bot_done", "silenceMs": 700, "maxMs": 15000},
                "interruptAfterBotMs": None}],
            "end": {"afterLastLine": "bot_done"}}}]}


def test_the_run_api_plan_becomes_a_script():
    plan = parse_plan(PLAN)
    assert plan.run_id == "r1" and plan.max_concurrent_calls == 2 and plan.max_call_ms == 600000
    (call,) = plan.calls
    assert call.session_token == "sdk:r1:s1:0" and call.scenario_name.startswith("refund")
    script, sources = script_from_plan_call(call, limits=plan.limits)
    assert {"wav", "synth", "audio"} <= set(sources)
    (line,) = script.lines
    assert line.source == "audio" and line.spec["sha256"] == "ab" * 32
    assert (line.wait.mode, line.wait.silence_ms, line.wait.max_ms) == ("bot_done", 700, 15000)
    assert line.interrupt_after_bot_ms is None
    assert script.end.mode == "bot_done" and script.max_call_ms == 300000
    assert script.name == "refund for a duplicate charge"
    # call.json keeps the script, but never the signed URL's credentials
    assert script.raw["lines"][0]["audio"]["url"] == "https://storage.example/a.raw"


def test_the_run_limit_caps_the_script_and_unknown_fields_are_ignored():
    call = dict(PLAN["calls"][0])
    call["script"] = dict(call["script"], newThing=1)
    call["script"]["lines"] = [dict(call["script"]["lines"][0], emotion="calm",
                                    wait={"rule": "fixed", "ms": 250}, interruptAfterBotMs=None)]
    script, _ = script_from_plan_call(call, limits={"maxCallMs": 1500})
    assert script.max_call_ms == 1500
    assert script.lines[0].wait.mode == "fixed" and script.lines[0].wait.ms == 250


def test_the_first_lines_wait_keeps_start_within_ms_and_a_fixed_zero():
    call = dict(PLAN["calls"][0])
    line = call["script"]["lines"][0]
    auto = {"rule": "bot_done", "startWithinMs": 3000, "silenceMs": 700, "maxMs": 15000}
    script, _ = script_from_plan_call(dict(call, script=dict(call["script"], lines=[dict(line, wait=auto)])))
    wait = script.lines[0].wait
    assert (wait.mode, wait.start_within_ms, wait.silence_ms, wait.max_ms) == ("bot_done", 3000, 700, 15000)
    caller = {"rule": "fixed", "ms": 0}
    script, _ = script_from_plan_call(dict(call, script=dict(call["script"], lines=[dict(line, wait=caller)])))
    assert (script.lines[0].wait.mode, script.lines[0].wait.ms) == ("fixed", 0)
    assert script.raw["lines"][0]["wait"] == {"mode": "fixed", "ms": 0}


def test_snake_case_scripts_and_interruptions_are_accepted():
    call = {"call_index": 3, "session_token": "t", "script": {
        "max_call_ms": 9000,
        "lines": [{"text": "a", "synth": {"ms": 300}},
                  {"text": "b", "synth": {"ms": 300}, "interrupt_after_bot_ms": 400, "wait": {"max_ms": 5000}}],
        "end": {"mode": "fixed", "ms": 100}}}
    script, _ = script_from_plan_call(call)
    assert script.lines[1].interrupt_after_bot_ms == 400 and script.end.mode == "fixed"


@pytest.mark.parametrize("bad, where", [
    ({"runId": "r", "calls": [{"callIndex": 0, "script": {"lines": []}}]}, "sessionToken"),
    ({"calls": []}, "runId"),
    ({"runId": "r", "calls": [{"callIndex": 0, "sessionToken": "t", "script": {}},
                              {"callIndex": 0, "sessionToken": "u", "script": {}}]}, "repeats a callIndex"),
])
def test_a_plan_that_cannot_run_is_refused(bad, where):
    with pytest.raises(PlanError, match=where):
        parse_plan(bad)


def test_a_call_with_no_lines_is_a_script_error():
    with pytest.raises(ScriptError):
        script_from_plan_call({"callIndex": 0, "sessionToken": "t", "script": {"lines": []}})


def test_target_settings_and_label():
    target = target_from_settings({"url": "wss://u:p@bot.example/media?key=secret", "format": "twilio",
                                   "headers": [{"name": "Authorization", "value": "Bearer x"}],
                                   "sessionQueryParam": "s"})
    assert target.sample_rate == 8000 and target.headers == {"Authorization": "Bearer x"}
    assert target.session_query_param == "s"
    assert "Bearer x" not in repr(target) and "connector" not in target.describe()
    assert target_label(target.url) == "wss://bot.example/media"


def test_the_connector_replaces_websockets_connect_with_the_same_arguments():
    pytest.importorskip("numpy")
    from evalkit.voicetest.transport import Target, make_transport
    from evalkit.voicetest.transport.base import MAX_MESSAGE_BYTES

    seen = {}

    class Refused(Exception):
        kind = "guard_refused"

    async def connector(url, **kwargs):
        seen["url"], seen["kwargs"] = url, kwargs
        raise Refused("private address")

    target = Target(url="ws://bot.internal/ws", format="pcm16", headers={"A": "b"}, connector=connector,
                    open_timeout_s=3.0, close_timeout_s=1.5)
    transport = make_transport(target, "tok")
    with pytest.raises(Refused):
        asyncio.run(transport.connect())
    assert seen["url"] == "ws://bot.internal/ws?evalkit_session=tok"
    assert seen["kwargs"] == {"additional_headers": {"A": "b"}, "compression": None,
                              "max_size": MAX_MESSAGE_BYTES, "open_timeout": 3.0, "close_timeout": 1.5,
                              "ping_interval": None}
    assert "connector" not in repr(target)


# The audio cache ------------------------------------------------------------------------------


def _served(tmp_path, data: bytes):
    """A fake API serving one clip, and the plan audio spec that names it."""
    fake = FakeVoiceApi().__enter__()
    sha = hashlib.sha256(data).hexdigest()
    fake.audio[sha] = data
    return fake, {"url": f"{fake.url}/storage/audio/{sha}?sig=x", "encoding": "pcm16", "sampleRate": 16000,
                  "sha256": sha}


def test_audio_is_downloaded_once_verified_and_cached(tmp_path):
    pytest.importorskip("numpy")
    from evalkit.voicetest.audio_cache import AudioCache
    from evalkit.voicetest.sources import PlanAudioSource

    from .fake_api import synth_clip

    fake, spec = _served(tmp_path, synth_clip(400, 1))
    try:
        cache = AudioCache(str(tmp_path / "cache"), retry=FAST)
        pcm = PlanAudioSource(cache).prepare(spec, 8000)
        # resampled to 8 kHz, the silent padding and the quietest edge samples trimmed
        assert 400 * 8 - 80 <= len(pcm) <= 400 * 8
        assert fake.downloads[spec["sha256"]] == 1
        # a second source over the same directory (a later run) reads the file instead
        again = AudioCache(str(tmp_path / "cache"), retry=FAST)
        PlanAudioSource(again).prepare(spec, 16000)
        assert fake.downloads[spec["sha256"]] == 1 and again.hits == 1
        # a cached file that no longer matches its hash is fetched again
        with open(cache.path_for(spec["sha256"]), "r+b") as fh:
            fh.write(b"\1\1\1\1")
        PlanAudioSource(AudioCache(str(tmp_path / "cache"), retry=FAST)).prepare(spec, 16000)
        assert fake.downloads[spec["sha256"]] == 2
        assert all(r["headers"].get("x-subscription-key") is None for r in fake.requests)
    finally:
        fake.__exit__()


def test_audio_that_does_not_match_its_sha256_is_refused(tmp_path):
    pytest.importorskip("numpy")
    from evalkit.voicetest.audio_cache import AudioCache, AudioError
    from evalkit.voicetest.sources import PlanAudioSource

    from .fake_api import synth_clip

    fake, spec = _served(tmp_path, synth_clip(300, 2))
    good = fake.audio[spec["sha256"]]
    fake.audio[spec["sha256"]] = good[:4000] + bytes([good[4000] ^ 0xFF]) + good[4001:]
    try:
        source = PlanAudioSource(AudioCache(str(tmp_path / "cache"), retry=FAST))
        with pytest.raises(AudioError, match="sha256"):
            source.prepare(spec, 16000)
        assert os.listdir(str(tmp_path / "cache")) == []  # nothing partial or wrong is kept
        with pytest.raises(AudioError):  # and the second line needing it fails at once
            source.prepare(spec, 16000)
        assert fake.downloads[spec["sha256"]] == 1
    finally:
        fake.__exit__()


@pytest.mark.parametrize("encoding", ["pcm16", "wav", "mulaw"])
def test_audio_encodings(encoding):
    np = pytest.importorskip("numpy")
    from evalkit.voicetest.audio import SynthSpec, mulaw_encode, synth_speech
    from evalkit.voicetest.audio_cache import decode_audio

    from .fake_api import synth_clip

    if encoding == "mulaw":
        data = mulaw_encode(synth_speech(SynthSpec.from_json({"ms": 500, "seed": 3}), 8000))
        spec = {"encoding": "mulaw", "sampleRate": 8000}
    else:
        data = synth_clip(500, 3, encoding)
        spec = {"encoding": "pcm16", "sampleRate": 16000}
    pcm = decode_audio(data, spec, 16000)
    assert pcm.dtype == np.int16 and 500 * 16 - 160 <= len(pcm) <= 500 * 16


# The API client --------------------------------------------------------------------------------


def test_the_api_client_hides_its_credential():
    api = VoiceTestApi.with_subscription_key("http://127.0.0.1:1", KEY, retry=FAST)
    assert KEY not in repr(api)
    with pytest.raises(VoiceTestApiError) as err:
        api.get_run("r1")
    assert KEY not in str(err.value) and err.value.status is None
    token = VoiceTestApi.with_run_token("http://x", "rt_123")
    assert token._auth == {"X-Voice-Run-Token": "rt_123"}


def test_creating_a_run_sends_one_idempotency_key_and_resends_a_lost_answer_with_it():
    body = {"name": "x", "scenarioSetId": "s", "repeats": 1}
    # the run was made but its answer lost: the resend carries the same key and gets the same run
    with FakeVoiceApi(lose_create_answer=True) as fake:
        api = VoiceTestApi.with_subscription_key(fake.url, KEY, retry=FAST)
        assert api.create_run(body)["runId"] == "run000"
        sent = fake.requests_to("POST /v1/voice-tests")
        keys = {r["headers"].get("idempotency-key") for r in sent}
        assert len(sent) == 2 and len(fake.created) == 1 and len(keys) == 1 and None not in keys
    # a 500 before anything was made is resent too; each run has a key of its own
    with FakeVoiceApi(fail_once={"POST /v1/voice-tests": 500}) as fake:
        api = VoiceTestApi.with_subscription_key(fake.url, KEY, retry=FAST)
        assert api.create_run(body)["runId"] == "run000"
        assert api.create_run(body)["runId"] == "run001"
        keys = [r["headers"]["idempotency-key"] for r in fake.requests_to("POST /v1/voice-tests")]
        assert len(keys) == 3 and keys[0] == keys[1] != keys[2]
    # a refusal is final
    with FakeVoiceApi(create_status=404) as fake:
        api = VoiceTestApi.with_subscription_key(fake.url, KEY, retry=FAST)
        with pytest.raises(VoiceTestApiError) as err:
            api.create_run(body)
        assert err.value.status == 404 and len(fake.requests_to("POST /v1/voice-tests")) == 1
    calls = []  # a refused connection: nothing was created, so it is sent again

    def counting():
        calls.append(1)
        raise HttpError("POST", "http://x", None, "refused", unsent=True)

    from evalkit.voicetest.net import with_retries

    with pytest.raises(HttpError):
        with_retries(counting, FAST, "x", idempotent=False)
    assert len(calls) == FAST.attempts


def test_results_and_reads_are_retried():
    with FakeVoiceApi(fail_once={r"POST /v1/voice-tests/r9/calls/0/result": 502,
                                 r"GET /v1/voice-tests/r9": 503}) as fake:
        api = VoiceTestApi.with_subscription_key(fake.url, KEY, retry=FAST)
        api.post_result("r9", 0, {"scorecard": None, "error": {"kind": "not_run", "message": "x"}})
        api.get_run("r9")
        assert len(fake.requests_to(r"POST /v1/voice-tests/r9/calls/0/result")) == 2
        assert len(fake.requests_to(r"GET /v1/voice-tests/r9")) == 2


def test_a_quota_refusal_is_not_waited_out():
    error = HttpError("POST", "http://x", 429, "daily voice test quota reached")
    assert not error.retryable
    assert HttpError("POST", "http://x", 429, "slow down", retry_after_s=2).retryable


# A dynamic call (RUN_API section 3.3): the opening as line 0, and the loop's rules.
DYNAMIC = dict(PLAN, caller={"provider": "google", "model": "gemini-2.5-flash", "keySource": "platform"},
               platformMinutesLeft=87, calls=[dict(PLAN["calls"][0], script={
                   "mode": "dynamic", "maxCallMs": 300000,
                   "lines": [dict(PLAN["calls"][0]["script"]["lines"][0],
                                  wait={"rule": "bot_done", "startWithinMs": 3000, "silenceMs": 700, "maxMs": 15000})],
                   "turns": {"max": 8, "wait": {"rule": "bot_done", "silenceMs": 700, "maxMs": 15000},
                             "interrupt": {"turn": 2, "afterBotMs": 1500}},
                   "end": {"afterLastLine": "bot_done"}})])


def test_a_dynamic_plan_becomes_a_dynamic_script():
    plan = parse_plan(DYNAMIC)
    assert plan.caller == {"provider": "google", "model": "gemini-2.5-flash", "keySource": "platform"}
    assert plan.platform_minutes_left == 87
    (call,) = plan.calls
    assert call.mode == "dynamic"
    script, _ = script_from_plan_call(call)
    assert script.dynamic and len(script.lines) == 1
    turns = script.turns
    assert (turns.max, turns.wait.mode, turns.wait.silence_ms, turns.wait.max_ms) == (8, "bot_done", 700, 15000)
    assert (turns.interrupt_turn, turns.interrupt_after_bot_ms) == (2, 1500)
    assert script.raw["mode"] == "dynamic" and script.raw["turns"]["interrupt"] == {"turn": 2, "after_bot_ms": 1500}


def test_fields_a_newer_server_adds_to_turns_are_left_out():
    call = dict(PLAN["calls"][0], script=dict(PLAN["calls"][0]["script"], mode="dynamic",
                                              turns={"max": 2, "voiceStyle": "warm"}))
    script, _ = script_from_plan_call(call)
    assert script.turns.max == 2 and script.raw["turns"] == {"max": 2}


def test_a_scripted_plan_with_or_without_mode_keeps_the_script_as_before():
    plain, _ = script_from_plan_call(PLAN["calls"][0])
    marked, _ = script_from_plan_call(dict(PLAN["calls"][0], script=dict(PLAN["calls"][0]["script"], mode="scripted")))
    assert not plain.dynamic and plain.turns is None and "mode" not in plain.raw
    assert marked.raw == plain.raw
    assert parse_plan(PLAN).caller is None and parse_plan(PLAN).platform_minutes_left is None
    assert parse_plan(dict(PLAN, caller="x", platformMinutesLeft=True)).caller is None


@pytest.mark.parametrize("script, message", [
    ({"mode": "live"}, "not one this caller plays"),
    ({"mode": "dynamic", "turns": {"max": 21}}, "turns.max must be a whole number from 1 to 20"),
    ({"mode": "dynamic", "turns": {"max": 2, "interrupt": {"turn": 3, "afterBotMs": 5}}}, "from 1 to turns.max"),
])
def test_a_dynamic_script_that_cannot_run_is_refused(script, message):
    call = dict(PLAN["calls"][0], script=dict(PLAN["calls"][0]["script"], **script))
    with pytest.raises(ScriptError, match=message):
        script_from_plan_call(call)


def test_the_turn_request_scrubs_the_keys_it_carries():
    with FakeVoiceApi(answer=lambda *a: {"status": 400, "message": "refused AIza-caller and sk-voice"}) as fake:
        api = VoiceTestApi.with_subscription_key(fake.url, KEY, retry=FAST)
        fake.plan = {"calls": [{"callIndex": 0, "script": {"turns": {"max": 2}}}]}
        fake.next_turn[0] = 1
        import base64
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\0\0" * 1600)
        with pytest.raises(VoiceTestApiError) as info:
            api.next_turn("r1", 0, 1, 0, {"data": base64.b64encode(buf.getvalue()).decode(), "durationMs": 100},
                          {"caller": "AIza-caller", "voice": "sk-voice", "unused": None})
        assert info.value.status == 400
        assert "AIza-caller" not in str(info.value) and "sk-voice" not in str(info.value)
        assert info.value.server_message == "refused *** and ***"
        (sent,) = fake.turn_requests
        assert sent["keys"] == {"caller": "AIza-caller", "voice": "sk-voice"}
        assert len(fake.requests_to(r"POST /v1/voice-tests/r1/calls/0/turns")) == 1  # one attempt: the caller retries
