"""Dynamic calls end to end (RUN_API sections 3.4 and 4.1, CONTRACT section 4.1): the caller against the
reference bot over real WebSockets, asking the fake API's turn endpoint for each line after the opening.

The fake hears nothing: its ``heard`` names the turn and the bot audio's length, and its lines are
synthetic speech, so what is checked is the loop itself (when each request is made and with what audio,
think time, retakes, cut-ins, the end, failures) and that the request keys stay out of everything kept."""

from __future__ import annotations

import json
import logging

from .support import require

require()

import pytest  # noqa: E402

from evalkit.voicetest.net import RetryPolicy  # noqa: E402
from evalkit.voicetest.refbot import Behaviour, RefBotServer  # noqa: E402
from evalkit.voicetest.runner import run_async  # noqa: E402
from evalkit.voicetest.trace import NS  # noqa: E402

from .fake_api import KEY, FakeVoiceApi, default_answer  # noqa: E402
from .support import free_port  # noqa: E402

PORTS = range(19800, 19840)
FAST = RetryPolicy(attempts=4, base_s=0.05, cap_s=0.2)
CALLER_KEY = "AIza-caller-test-key-0123456789"
TTS_KEY = "sk-tts-dynamic-test-0123456789"
OPENING = "Hi, I was charged twice for my order."


def _events(report, index=0):
    path = f"{report.out_dir}/call-{index:03d}/events.jsonl"
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _of(events, kind):
    return [e for e in events if e["type"] == kind]


async def _run(fake, tmp_path, behaviour, scenarios, **kwargs):
    port = free_port(PORTS)
    async with RefBotServer(Behaviour.from_json(behaviour), "pcm16", port=port):
        return await run_async(
            f"ws://127.0.0.1:{port}/ws", format="pcm16", scenarios=scenarios, first_speaker="caller",
            subscription_key=KEY, api_url=fake.url, handle_signals=False, retry=FAST, wait_timeout_s=20,
            out_dir=str(tmp_path / "out"), cache_dir=str(tmp_path / "cache"), **kwargs)


def _kept_files(report):
    import os

    out = []
    for root, _, files in os.walk(report.out_dir):
        for name in files:
            if name.endswith((".json", ".jsonl")):
                with open(os.path.join(root, name), encoding="utf-8") as fh:
                    out.append(fh.read())
    return out


async def test_a_dynamic_call_asks_for_each_line_and_keeps_the_conversation(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="evalkit")
    scenario = {"name": "duplicate charge", "goal": "Get the second charge refunded", "opening": OPENING,
                "maxTurns": 2, "successCriteria": ["The agent refunds the second charge"]}
    with FakeVoiceApi(think_s=0.4) as fake:
        report = await _run(fake, tmp_path, {"default_turn": {"delay_ms": 600, "reply_ms": 900}}, [scenario],
                            caller={"provider": "google", "api_key": CALLER_KEY},
                            voice={"provider": "openai", "api_key": TTS_KEY})
        assert report.exit_code == 0, report.to_json()
        (created,) = fake.created
        assert created["scenarios"] == [scenario]
        assert created["caller"] == {"provider": "google", "apiKey": CALLER_KEY}
        assert fake.plan["calls"][0]["script"]["mode"] == "dynamic"
        assert report.caller == {"provider": "google", "model": "gemini-3.8-flash", "keySource": "request"}

        # one request per line after the opening, then the heard-only one after the last (turns.max + 1)
        requests = fake.turn_requests
        assert [(r["turn"], r["retake"]) for r in requests] == [(1, 0), (2, 0), (3, 0)]
        for r in requests:
            assert r["format"] == (1, 2, 16000)  # a 16 kHz mono PCM16 WAV
            assert r["keys"] == {"caller": CALLER_KEY, "voice": TTS_KEY}  # re-sent every turn, never stored
            assert r["headers"]["x-subscription-key"] == KEY
            # the bot's answer and the quiet that ended it, from the end of the caller's line
            assert 1300 <= r["audioMs"] <= 2500 and abs(r["audioMs"] - r["durationMs"]) <= 1
            assert r["cutInAtMs"] is None  # no line cut into the bot's answer

        result = fake.results[0]
        assert result["error"] is None
        card = result["scorecard"]
        call = card["call"]
        assert call["mode"] == "dynamic"
        assert [(e["turn"], e["role"]) for e in call["transcript"]] == [
            (0, "caller"), (0, "bot"), (1, "caller"), (1, "bot"), (2, "caller"), (2, "bot")]
        assert call["transcript"][0] == {"turn": 0, "role": "caller", "text": OPENING}
        assert call["transcript"][2]["text"] == "Here is my reply number 1."
        assert call["transcript"][5]["text"].startswith("bot answer before turn 3")
        turns = card["turns"]
        assert [t["heard"].split(" (")[0] for t in turns] == [
            "bot answer before turn 1", "bot answer before turn 2", "bot answer before turn 3"]
        assert [t["caller_text"] for t in turns] == [OPENING, "Here is my reply number 1.", "Here is my reply number 2."]
        # think time: the round trip to the turn endpoint (0.4 s of fake model time), never the bot's
        thinks = [t["caller"]["think_ms"] for t in turns]
        assert thinks[0] == 0 and all(400 <= ms <= 1500 for ms in thinks[1:]), thinks
        assert all(t["response_delay_ms"] == pytest.approx(600, abs=60) for t in turns), [
            t["response_delay_ms"] for t in turns]
        assert all(t["silence_ms"] == 0 for t in turns)
        health = card["caller_health"]
        assert health["think_ms_p50"] is not None and health["turn_requests"] == 3 and health["retakes"] == 0
        assert card["prosody"]["stock_phrases"]["status"] == "measured"

        events = _events(report)
        assert [e["ms"] for e in _of(events, "think")] == thinks
        answers = _of(events, "turn_answer")
        assert [(a["index"], a["done"]) for a in answers] == [(1, False), (2, False), (3, True)]
        assert answers[-1].get("final") is True
        # the last request is made after the hang-up, so it adds nothing to the call
        closed = _of(events, "closed")[-1]
        assert answers[-1]["t"] == closed["t"] and call["duration_ms"] == closed["t"]

        outcome = report.calls[0]
        assert outcome.mode == "dynamic" and outcome.turns_taken == 2 and len(outcome.transcript) == 6
        assert report.to_json()["calls"][0]["transcript"] == call["transcript"]

        root = next(s for s in fake.spans if s.get("operation") == "voice.call")
        assert root["attributes"][NS + "call.mode"] == "dynamic"
        spans = sorted((s for s in fake.spans if s.get("operation") == "voice.turn"),
                       key=lambda s: s["attributes"][NS + "turn.index"])
        assert [s["attributes"][NS + "transcript.bot"] for s in spans] == [t["heard"] for t in turns]
        assert [s["attributes"][NS + "turn.think_ms"] for s in spans] == thinks

        # the request keys reached the create and the turn requests, and nothing else
        for r in fake.requests:
            if r["path"].endswith("/turns") or r["path"] == "/v1/voice-tests":
                continue
            blob = json.dumps(r["headers"]) + r["query"] + r["body"].decode("utf-8", "replace")
            assert CALLER_KEY not in blob and TTS_KEY not in blob, r["path"]
        kept = "".join(_kept_files(report)) + json.dumps(report.to_json(), default=str) + caplog.text
        assert CALLER_KEY not in kept and TTS_KEY not in kept


@pytest.mark.parametrize("talk_ms, retake", [(1400, True), (500, False)])
async def test_the_bot_speaking_again_while_a_line_is_decided(tmp_path, talk_ms, retake):
    # The bot says more 3.2 s into the call, while turn 1 is being decided: the answer arrives with the bot
    # speaking, so the caller waits for it to finish, then asks again if it spoke 1,000 ms or more.
    behaviour = {"turns": [{"delay_ms": 300, "reply_ms": 600, "talk_over": {"at_ms": 3200, "ms": talk_ms}}],
                 "default_turn": {"delay_ms": 400, "reply_ms": 700}}

    def answer(call, turn, retake_, request):
        return dict(default_answer(call, turn, retake_, request), delay=1.5 if retake_ == 0 else 0.3)

    scenario = {"name": "talks on", "goal": "Get a refund", "opening": OPENING, "maxTurns": 1}
    with FakeVoiceApi(answer=answer) as fake:
        report = await _run(fake, tmp_path, behaviour, [scenario])
        assert report.calls[0].error is None, report.to_json()
        events = _events(report)
        again = [e for e in _of(events, "wait_start") if e.get("again")]
        assert [e["index"] for e in again] == [1]
        asked = [(e["index"], e["retake"]) for e in _of(events, "turn_request")]
        if retake:
            (taken,) = _of(events, "retake")
            assert taken["index"] == 1 and taken["bot_spoke_ms"] >= 1000
            assert asked == [(1, 0), (1, 1), (2, 0)]
            first, second = [r for r in fake.turn_requests if r["turn"] == 1]
            assert second["audioMs"] > first["audioMs"] + 1000  # the longer audio, from the same line end
        else:
            assert not _of(events, "retake") and asked == [(1, 0), (2, 0)]
        card = fake.results[0]["scorecard"]
        assert [e["role"] for e in card["call"]["transcript"]] == ["caller", "bot", "caller", "bot"]
        assert card["turns"][0]["heard"].startswith("bot answer before turn 1")
        # the deciding time runs from the request that wrote the line, the wait for the bot included
        think = card["turns"][1]["caller"]["think_ms"]
        assert think >= (300 if retake else 1500)
        assert card["turns"][0]["silence_ms"] == 0


@pytest.mark.parametrize("reply_ms, after_ms, status", [(2400, 600, "reached"), (500, 1500, "not_reached")])
async def test_a_cut_in_written_with_the_line_before(tmp_path, reply_ms, after_ms, status):
    behaviour = {"turns": [{"delay_ms": 300, "reply_ms": 700}],
                 "default_turn": {"delay_ms": 300, "reply_ms": reply_ms, "on_interrupt": "stop", "stop_delay_ms": 200}}
    scenario = {"name": "cuts in", "goal": "Find out how long a refund takes", "opening": OPENING, "maxTurns": 2,
                "interrupt": {"turn": 2, "afterBotMs": after_ms}}
    with FakeVoiceApi() as fake:
        report = await _run(fake, tmp_path, behaviour, [scenario])
        assert report.calls[0].error is None, report.to_json()
        # the cut-in came with turn 1's answer, so turn 2 is never asked for; then the heard-only turn 3
        assert [(r["turn"], r["retake"]) for r in fake.turn_requests] == [(1, 0), (3, 0)]
        events = _events(report)
        (planned,) = _of(events, "interrupt_planned")
        assert planned == {**planned, "index": 2, "after_bot_ms": after_ms}
        assert bool(_of(events, "interrupt_not_reached")) == (status == "not_reached")
        turns = fake.results[0]["scorecard"]["turns"]
        assert [t["caller_text"] for t in turns] == [OPENING, "Here is my reply number 1.", "Sorry, quick question."]
        assert turns[2]["interruption_planned"] and turns[2]["interruption_status"] == status
        assert (turns[2]["interruption_stop_ms"] is not None) == (status == "reached")
        assert turns[2]["caller"]["think_ms"] == 0  # written with the line before, so nothing to decide
        # The request after the cut-in sends the bot's audio from the end of line 1, the line whose answer
        # was cut, with the point the cut-in began (RUN_API 4.1, cutInAtMs), so the API hears that answer
        # as well as the reply to the cut-in. Reached: 300 ms to the answer, then after_ms of it; not
        # reached: the 500 ms answer, then the turn wait's 600 ms of quiet.
        after_cut = fake.turn_requests[1]
        low, high = (700, 1300) if status == "reached" else (1200, 2200)
        assert low <= after_cut["cutInAtMs"] <= high, after_cut["cutInAtMs"]
        assert after_cut["audioMs"] > after_cut["cutInAtMs"] + 500  # the cut-in line, then the reply to it
        line_1_end = next(e for e in _of(events, "line_end") if e["index"] == 1)["t"]
        cut_start = next(e for e in _of(events, "line_start") if e["index"] == 2)["t"]
        assert abs(after_cut["cutInAtMs"] - (cut_start - line_1_end)) <= 2
        requested = [e for e in _of(events, "turn_request") if e["index"] == 3]
        assert requested and all(e["cut_in_at_ms"] == after_cut["cutInAtMs"] for e in requested)
        assert fake.turn_requests[0]["cutInAtMs"] is None
        # an API that does not send heardBeforeCutIn leaves no text for the cut answer; the reply's is there
        assert turns[1]["heard"] is None and turns[2]["heard"].startswith("bot answer before turn 3")


async def test_what_the_bot_said_before_a_cut_in_is_its_turn_in_the_transcript(tmp_path):
    behaviour = {"turns": [{"delay_ms": 300, "reply_ms": 700}],
                 "default_turn": {"delay_ms": 300, "reply_ms": 2400, "on_interrupt": "stop", "stop_delay_ms": 200}}
    scenario = {"name": "cuts in", "goal": "Find out how long a refund takes", "opening": OPENING, "maxTurns": 2,
                "interrupt": {"turn": 2, "afterBotMs": 600}}
    with FakeVoiceApi(heard_before_cut_in=True) as fake:
        report = await _run(fake, tmp_path, behaviour, [scenario])
        assert report.calls[0].error is None, report.to_json()
        cut = fake.turn_requests[1]["cutInAtMs"]
        card = fake.results[0]["scorecard"]
        turns = card["turns"]
        # The answer the cut-in interrupted (line 1's) is what turn 3's answer heard before the cut-in.
        assert turns[1]["heard"] == f"bot answer cut in at {cut} ms"
        assert turns[2]["heard"].startswith("bot answer before turn 3")
        # In order, as the API's call read has it: line 1, the bot's words before the cut-in, the cut-in.
        entries = [(e["turn"], e["role"], e["text"]) for e in card["call"]["transcript"]][2:5]
        assert entries == [(1, "caller", "Here is my reply number 1."), (1, "bot", f"bot answer cut in at {cut} ms"),
                           (2, "caller", "Sorry, quick question.")]
        assert card["call"]["transcript"][5]["text"].startswith("bot answer before turn 3")


async def test_a_cut_in_planned_before_the_first_answer_sends_that_answer_too(tmp_path):
    # interrupt.turn 1: the cut-in is planned as line 1, said 600 ms into the bot's answer to the opening.
    behaviour = {"turns": [{"delay_ms": 300, "reply_ms": 2400, "on_interrupt": "stop", "stop_delay_ms": 200}],
                 "default_turn": {"delay_ms": 300, "reply_ms": 700}}
    scenario = {"name": "cuts in early", "goal": "Find out how long a refund takes", "opening": OPENING,
                "maxTurns": 2, "interrupt": {"turn": 1, "afterBotMs": 600}}
    with FakeVoiceApi() as fake:
        report = await _run(fake, tmp_path, behaviour, [scenario])
        assert report.calls[0].error is None, report.to_json()
        assert [(r["turn"], r["retake"]) for r in fake.turn_requests] == [(2, 0), (3, 0)]
        events = _events(report)
        opening_end = next(e for e in _of(events, "line_end") if e["index"] == 0)["t"]
        cut_start = next(e for e in _of(events, "line_start") if e["index"] == 1)["t"]
        first, final = fake.turn_requests
        assert 700 <= first["cutInAtMs"] <= 1300 and abs(first["cutInAtMs"] - (cut_start - opening_end)) <= 2
        assert first["audioMs"] > first["cutInAtMs"] + 500
        assert final["cutInAtMs"] is None  # line 2 was a plain line
        turns = fake.results[0]["scorecard"]["turns"]
        assert turns[1]["interruption_planned"] and turns[1]["interruption_status"] == "reached"


async def test_done_ends_the_call_with_or_without_a_last_line(tmp_path):
    def answer(call, turn, retake, request):
        if call["callIndex"] == 0:
            return {"line": "Thanks, that is all I needed.", "done": True}
        return {"done": True}

    scenarios = [{"name": "last words", "goal": "Say thanks", "opening": OPENING},
                 {"name": "no last words", "goal": "Hang up", "opening": OPENING}]
    with FakeVoiceApi(answer=answer, max_concurrent=2) as fake:
        report = await _run(fake, tmp_path, {"default_turn": {"delay_ms": 300, "reply_ms": 700}}, scenarios)
        assert [c.error for c in report.calls] == [None, None], report.to_json()
        asked = {i: [r["turn"] for r in fake.turn_requests if r["call"] == i] for i in (0, 1)}
        # a done line is said, the end rule waited, and the bot's answer to it still sent; with no line the
        # call ends at once, the bot's last answer already heard
        assert asked == {0: [1, 2], 1: [1]}
        first, second = (fake.results[i]["scorecard"]["call"]["transcript"] for i in (0, 1))
        assert [e["role"] for e in first] == ["caller", "bot", "caller", "bot"]
        assert first[2]["text"] == "Thanks, that is all I needed."
        assert [e["role"] for e in second] == ["caller", "bot"]
        assert report.calls[0].turns_taken == 1 and report.calls[1].turns_taken == 0
        assert report.calls[1].ended_by == "script"


async def test_turn_failures_retries_and_a_lost_answer(tmp_path):
    asked = {}

    def answer(call, turn, retake, request):
        index = call["callIndex"]
        asked[index] = asked.get(index, 0) + 1
        if index == 0:  # a key the provider refused: the call ends at once, in its words, without the key
            return {"status": 400, "message": f"API key not valid: {request['keys']['caller']}"}
        if index == 1 and asked[index] == 1:  # a rate limit that names its wait
            return {"status": 429, "retryAfter": 1, "message": "rate limited"}
        if index == 2 and asked[index] == 1:  # answered and kept, but the answer lost on the way back
            return dict(default_answer(call, turn, retake, request), lose=True)
        if index == 3:  # overloaded for longer than the call's budget
            return {"status": 503, "retryAfter": 1, "message": "overloaded"}
        return default_answer(call, turn, retake, request)

    scenarios = [{"name": f"call {i}", "goal": "Get a refund", "opening": OPENING, "maxTurns": 1} for i in range(4)]
    with FakeVoiceApi(answer=answer, max_concurrent=4, call_max_ms=9000) as fake:
        report = await _run(fake, tmp_path, {"default_turn": {"delay_ms": 300, "reply_ms": 700}}, scenarios,
                            caller={"api_key": CALLER_KEY})
        errors = [c.error for c in report.calls]
        assert errors[0]["kind"] == "caller_turn_failed"
        assert "API key not valid: ***" in errors[0]["message"] and CALLER_KEY not in errors[0]["message"]
        assert errors[1] is None and errors[2] is None
        assert errors[3]["kind"] == "caller_turn_failed" and "budget" in errors[3]["message"]
        assert report.exit_code == 2  # the testing side failed, which says nothing about the bot

        def turns_of(i):
            return [(r["turn"], r["retake"]) for r in fake.turn_requests if r["call"] == i]

        assert turns_of(0) == [(1, 0)]
        assert turns_of(1)[:2] == [(1, 0), (1, 0)]
        first, again = [r["at"] for r in fake.turn_requests if r["call"] == 1][:2]
        assert again - first >= 0.95  # Retry-After was honoured
        assert turns_of(2)[:2] == [(1, 0), (1, 0)]
        assert asked[2] == 1  # the repeat got the stored answer: the model was asked once
        assert len(turns_of(3)) >= 3
        failed = _of(_events(report, 1), "turn_failed")
        assert failed[0]["status"] == 429 and _of(_events(report, 1), "turn_answer")
        assert fake.results[0]["error"]["kind"] == "caller_turn_failed"
        assert CALLER_KEY not in "".join(_kept_files(report)) + json.dumps(fake.results)


async def test_the_request_for_the_bots_last_answer_gives_way_to_stop(tmp_path):
    """A drain or cancel never waits on the heard-only request after the hang-up."""
    import asyncio
    import threading
    import time

    from evalkit.voicetest.audio import SpeechTracker
    from evalkit.voicetest.caller import _BotTail, _DynamicRunner
    from evalkit.voicetest.recording import EventLog
    from evalkit.voicetest.script import parse_script
    from evalkit.voicetest.sources import render_lines

    released = threading.Event()

    class Slow:
        api = type("Api", (), {"timeout_s": 30.0})()
        executor = None

        def ask(self, *args):
            released.wait(10)
            raise AssertionError("not reached")

        def scrub(self, text):
            return text

    script = parse_script({"mode": "dynamic", "lines": [{"text": "hi", "synth": {"ms": 300}}], "turns": {"max": 1}})
    events = EventLog(str(tmp_path / "events.jsonl"), 16000)
    runner = _DynamicRunner(script, render_lines(script, 16000), 16000, events, SpeechTracker(16000), Slow(),
                            _BotTail(16000, 60000), asyncio.get_running_loop())
    stop = asyncio.Event()
    asyncio.get_running_loop().call_later(0.2, stop.set)
    started = time.monotonic()
    try:
        warning = await runner.final_request(2, 16000, time.monotonic() + 60, stop)
    finally:
        released.set()
        events.close()
    assert time.monotonic() - started < 2.0
    assert warning == "the bot's last answer was not transcribed: turn 2: the run was stopped"
    assert [e["type"] for e in events.events][-2:] == ["turn_request", "turn_failed"]
