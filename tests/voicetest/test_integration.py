"""The caller against the reference bot over real WebSockets on localhost, for both formats."""

import asyncio
import json
import socket
import time
import wave

from .support import require

require()

import numpy as np
import pytest

from evalkit.voicetest.caller import VAD_VERSION, run_call
from evalkit.voicetest.refbot import Behaviour, RefBotServer
from evalkit.voicetest.script import parse_script
from evalkit.voicetest.transport import Target

PORTS = range(19500, 19550)
_rotation = {"i": 0}
# In-process the bot and caller share one event loop, so the bot's sends can wait behind the
# caller's per-frame work; 8 ms covers that plus the round trip.
DELTA_MS = 8


def free_port() -> int:
    for _ in range(len(PORTS)):
        port = PORTS[_rotation["i"] % len(PORTS)]
        _rotation["i"] += 1
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port in 19500-19549")


BEHAVIOUR = {
    "greeting": {"after_ms": 200, "ms": 700},
    "send_unknown_frame": True,
    "turns": [
        {"delay_ms": 650, "reply_ms": 900},
        {"delay_ms": 600, "reply_ms": 2500, "mode": "burst", "on_interrupt": "clear", "stop_delay_ms": 200},
        {"delay_ms": 700, "reply_ms": 600, "talk_over": {"at_ms": 300, "ms": 400}},
        {"silent": True},
    ],
    "default_turn": {"delay_ms": 600, "reply_ms": 500},
}

SCRIPT = {
    "name": "integration",
    "lines": [
        {"text": "first", "synth": {"ms": 800, "seed": 1}},
        {"text": "second", "synth": {"ms": 700, "seed": 2}},
        {"text": "barge in", "synth": {"ms": 900, "seed": 3}, "interrupt_after_bot_ms": 700,
         "wait": {"max_ms": 6000}},
        {"text": "after talk-over", "synth": {"ms": 500, "seed": 4},
         "wait": {"mode": "bot_done", "silence_ms": 500, "max_ms": 6000}},
        {"text": "after silence", "synth": {"ms": 500, "seed": 5}, "wait": {"mode": "fixed", "ms": 1200}},
    ],
    "end": {"mode": "bot_done", "silence_ms": 600, "max_ms": 5000},
}

STALL_FRAME = 30


def stall(frame):
    if frame == STALL_FRAME:
        time.sleep(0.032)  # runs ~20 ms before the deadline, so the frame leaves ~12 ms late


async def call(fmt, rate, out_dir, behaviour=BEHAVIOUR, script=SCRIPT, send_start=True, hook=stall):
    port = free_port()
    server = RefBotServer(Behaviour.from_json(behaviour), fmt, port=port, sample_rate=rate)
    async with server:
        target = Target(url=server.url, format=fmt, sample_rate=rate, send_start=send_start)
        result = await run_call(target, parse_script(script), str(out_dir), session="sim:run:sc:7",
                                tick_hook=hook)
        await asyncio.sleep(0.2)  # let the bot see the close
    return result, server.truth.records


def by_type(events, type_):
    return [e for e in events if e["type"] == type_]


def read_recording(path):
    with wave.open(str(path), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (2, 2, 16000)
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, 2)
    return data[:, 0], data[:, 1]


def first_sound(channel, from_ms):
    nz = np.flatnonzero(np.abs(channel[int(from_ms * 16) :].astype(np.int32)) > 8)
    return None if not len(nz) else from_ms + nz[0] / 16.0


@pytest.mark.parametrize("fmt,rate", [("twilio", 8000), ("pcm16", 16000)])
async def test_call_against_the_reference_bot(fmt, rate, tmp_path):
    result, truth = await call(fmt, rate, tmp_path)
    events = result.events
    assert result.ended_by == "script", result.errors

    # every event type the contract defines shows up
    expected = {"connect", "start_sent", "closed", "line_start", "line_end", "wait_start", "wait_end",
                "think", "interrupt_planned", "bot_chunk", "clear", "pacing", "error"}
    if fmt == "twilio":
        expected |= {"stop_sent", "mark_sent", "mark_echoed"}
    assert expected <= {e["type"] for e in events}
    for e in events:
        assert isinstance(e["t"], int) and isinstance(e["wall"], float) and isinstance(e["type"], str)
    with open(result.events_path) as f:
        assert [json.loads(line) for line in f] == events

    # pacing: the stalled frame is logged at its sample position, and nothing else is late
    (late,) = [p for p in by_type(events, "pacing") if p["frame"] == STALL_FRAME]
    assert late["t"] == STALL_FRAME * 20 and late["late_ms"] >= 10
    (err,) = by_type(events, "error")
    assert err["kind"] == "unknown_frame"

    # call.json
    call_json = json.loads(open(result.call_path).read())
    assert call_json["duration_ms"] == result.duration_ms == result.frames_sent * 20
    assert call_json["session"] == "sim:run:sc:7" and call_json["vad"]["version"] == VAD_VERSION
    assert call_json["format"] == fmt and call_json["sample_rate"] == rate
    assert call_json["script"]["name"] == "integration" and "t0_epoch_ms" in call_json

    # the recording: 2 channels, 16 kHz, exactly the call length
    caller, bot = read_recording(result.recording_path)
    assert len(caller) == result.duration_ms * 16

    # caller channel: sound exactly inside the line spans
    starts = {e["index"]: e["t"] for e in by_type(events, "line_start")}
    ends = {e["index"]: e["t"] for e in by_type(events, "line_end")}
    assert sorted(starts) == sorted(ends) == [0, 1, 2, 3, 4]
    mask = np.zeros(len(caller), bool)
    for i in starts:
        assert ends[i] - starts[i] == SCRIPT["lines"][i]["synth"]["ms"]
        mask[(starts[i] - 2) * 16 : (ends[i] + 2) * 16] = True
        assert abs(first_sound(caller, starts[i] - 1) - starts[i]) <= 2
    assert not np.abs(caller[~mask].astype(np.int32)).max() > 8

    # the bot heard the caller where the caller spoke
    t_start = {r["turn"]: r["ms"] for r in truth if r["type"] == "caller_speech_start"}
    t_end = {r["turn"]: r["ms"] for r in truth if r["type"] == "caller_speech_end"}
    for i in starts:
        assert abs(t_start[i] - starts[i]) <= 10 and abs(t_end[i] - ends[i]) <= 10

    # the session token arrived in the query and in-band
    (start,) = [r for r in truth if r["type"] == "start_received"]
    assert start["query_session"] == start["inband_session"] == "sim:run:sc:7"

    # bot channel: each truth speech start is heard within DELTA_MS of where the bot placed it
    speech = [r for r in truth if r["type"] == "speech_start"]
    assert {(r["kind"], r["turn"]) for r in speech} == {
        ("greeting", -1), ("reply", 0), ("reply", 1), ("talk_over", 2), ("reply", 2), ("reply", 4)}
    for r in speech:
        heard = first_sound(bot, r["ms"] - 2)
        assert r["ms"] - 1 <= heard <= r["ms"] + DELTA_MS, (r, heard)
    for r in [r for r in truth if r["type"] == "speech_start" and r["kind"] == "reply"]:
        assert r["ms"] == pytest.approx(r["planned_ms"])  # every reply went out exactly on plan

    # the planned interruption: line 2 began after 700 ms of the burst reply, and the clear cut it
    (burst,) = [r for r in speech if r["kind"] == "reply" and r["turn"] == 1]
    assert burst["mode"] == "burst"
    assert 700 <= starts[2] - burst["ms"] <= 700 + 60
    (cleared,) = [r for r in truth if r["type"] == "speech_end" and r["reason"] == "cleared"]
    (clear_ev,) = by_type(events, "clear")
    assert cleared["ms"] - 1 <= clear_ev["t"] <= cleared["ms"] + DELTA_MS
    assert clear_ev["dropped_ms"] == pytest.approx(2500 - (clear_ev["t"] - burst["ms"]), abs=DELTA_MS + 2)
    (talk_over,) = [r for r in speech if r["kind"] == "talk_over"]
    gap = bot[int((clear_ev["t"] + 2) * 16) : int((talk_over["ms"] - 2) * 16)]
    assert not np.abs(gap.astype(np.int32)).max() > 8  # nothing of the dropped audio was heard
    burst_chunks = [e for e in by_type(events, "bot_chunk") if burst["now_ms"] - 1 <= e["t"] <= burst["now_ms"] + 10]
    assert max(e["queued_ms"] for e in burst_chunks) >= 2400  # the buffer saw the whole burst at once

    # the talk-over starts while the caller is speaking
    assert starts[2] < talk_over["ms"] < ends[2]

    # waits: line 3 waited for reply 2, line 4 waited a fixed 1200 ms after the silent turn
    reasons = {e["index"]: e["reason"] for e in by_type(events, "wait_end")}
    assert reasons == {0: "bot_done", 1: "bot_done", 2: "interrupt", 3: "bot_done", 4: "fixed", 5: "bot_done"}
    assert starts[4] - ends[3] in (1200, 1220)
    assert [r["turn"] for r in truth if r["type"] == "silent_turn"] == [3]

    if fmt == "twilio":
        sent = {e["name"]: e for e in by_type(events, "mark_sent")}
        echoed = {e["name"]: e for e in by_type(events, "mark_echoed")}
        assert set(sent) == set(echoed) == {"greeting", "reply-0", "reply-1", "talk_over-2", "reply-2", "reply-4"}
        assert echoed["reply-1"]["cleared"] is True
        for name, e in echoed.items():
            if not e["cleared"]:
                assert e["t"] == sent[name]["heard_ms"]
        received = {r["name"] for r in truth if r["type"] == "mark_received"}
        assert received == set(sent)


async def test_pcm16_24k_without_a_start_message(tmp_path):
    behaviour = {"sample_rate": 24000, "turns": [{"delay_ms": 700, "reply_ms": 800}]}
    script = {"lines": [{"text": "hi", "synth": {"ms": 600}, "wait": {"mode": "fixed", "ms": 200}}],
              "end": {"mode": "bot_done", "silence_ms": 400, "max_ms": 4000}}
    result, truth = await call("pcm16", 24000, tmp_path, behaviour, script, send_start=False, hook=None)
    assert result.ended_by == "script"
    assert "start_sent" not in {e["type"] for e in result.events}
    assert not by_type(result.events, "pacing") or max(p["late_ms"] for p in by_type(result.events, "pacing")) < 50
    caller, bot = read_recording(result.recording_path)
    assert len(caller) == len(bot) == result.duration_ms * 16
    (start,) = [r for r in truth if r["type"] == "start_received"]
    assert start["sample_rate"] == 24000 and start["inband_session"] is None
    assert start["query_session"] == "sim:run:sc:7"
    (reply,) = [r for r in truth if r["type"] == "speech_start"]
    (line_end,) = by_type(result.events, "line_end")
    heard = first_sound(bot, reply["ms"] - 2)
    assert reply["ms"] - 1 <= heard <= reply["ms"] + DELTA_MS
    assert abs((reply["ms"] - line_end["t"]) - 700) <= 10


@pytest.mark.parametrize("reads", [True, False])
async def test_bot_hanging_up_ends_the_call(reads, tmp_path):
    from websockets.asyncio.server import serve

    async def hang_up(ws):
        async def drain():
            async for _ in ws:
                pass

        task = asyncio.ensure_future(drain()) if reads else None
        await asyncio.sleep(0.3)
        # A bot that stops reading never sees our close frame; the caller must not wait on it.
        await ws.close() if reads else ws.transport.close()
        if task is not None:
            await task

    port = free_port()
    async with serve(hang_up, "127.0.0.1", port, close_timeout=1):
        script = {"lines": [{"synth": {"ms": 400}, "wait": {"mode": "fixed", "ms": 5000}}]}
        began = time.monotonic()
        result = await run_call(Target(url=f"ws://127.0.0.1:{port}/", format="pcm16"),
                                parse_script(script), str(tmp_path))
        assert time.monotonic() - began < 3.0
    assert result.ended_by == "bot_closed"
    closed = by_type(result.events, "closed")
    assert len(closed) == 1 and closed[0]["by"] == "bot"
    assert 250 <= result.duration_ms <= 600
    caller, bot = read_recording(result.recording_path)
    assert len(caller) == result.duration_ms * 16


async def test_connect_failure_is_recorded(tmp_path):
    from evalkit.voicetest.caller import CallError

    port = free_port()  # nothing listens here
    with pytest.raises(CallError):
        await run_call(Target(url=f"ws://127.0.0.1:{port}/", format="pcm16", open_timeout_s=2),
                       parse_script({"lines": [{"synth": {"ms": 100}}]}), str(tmp_path))
    call_json = json.loads((tmp_path / "call.json").read_text())
    assert call_json["ended_by"] == "error" and call_json["errors"]


# The first line's wait under firstSpeaker auto (RUN_API section 3.3): a greeting that starts within
# 3 s is heard out, however long it is; with none, the caller speaks at 3 s, not at the 15 s maximum.
AUTO_FIRST_WAIT = {"mode": "bot_done", "start_within_ms": 3000, "silence_ms": 700, "max_ms": 15000}
GREETING_MS = 4500


@pytest.mark.parametrize("greets", [False, True])
async def test_the_first_line_waits_for_a_greeting_only_if_one_starts(greets, tmp_path):
    behaviour = {"turns": [{"delay_ms": 500, "reply_ms": 600}]}
    if greets:
        behaviour["greeting"] = {"after_ms": 300, "ms": GREETING_MS}
    script = {"lines": [{"text": "hello", "synth": {"ms": 700, "seed": 1}, "wait": AUTO_FIRST_WAIT}],
              "end": {"mode": "bot_done", "silence_ms": 500, "max_ms": 4000}}
    result, truth = await call("pcm16", 16000, tmp_path, behaviour, script, hook=None)
    assert result.ended_by == "script", result.errors
    events = result.events
    (wait_start,) = [e for e in by_type(events, "wait_start") if e["index"] == 0]
    assert wait_start["start_within_ms"] == 3000 and wait_start["max_ms"] == 15000
    (wait_end,) = [e for e in by_type(events, "wait_end") if e["index"] == 0]
    (line_start,) = by_type(events, "line_start")
    assert line_start["t"] == wait_end["t"]
    if greets:
        # Heard out to its end and the caller's silence after it, though it ran past the 3 s.
        (greeting_end,) = [r for r in truth if r["type"] == "speech_end" and r["kind"] == "greeting"]
        assert greeting_end["reason"] == "complete" and greeting_end["ms"] >= 300 + GREETING_MS - 10
        assert wait_end["reason"] == "bot_done"
        assert 700 <= line_start["t"] - greeting_end["ms"] <= 700 + 60
        assert not [r for r in truth if r["type"] == "interruption"]  # the caller never cut in
    else:
        assert wait_end["reason"] == "no_greeting"
        assert 3000 <= wait_end["t"] <= 3000 + 40
        assert not [r for r in truth if r["type"] == "speech_start" and r["kind"] == "greeting"]
    # Either way the call went on as scripted: the bot answered the line.
    (reply,) = [r for r in truth if r["type"] == "speech_start" and r["kind"] == "reply"]
    assert reply["ms"] > by_type(events, "line_end")[0]["t"]


async def test_a_bot_finishing_before_the_interruption_point_is_answered_at_once(tmp_path):
    from evalkit.voicetest.analysis.scorecard import analyze
    from evalkit.voicetest.summary import call_lines

    # The bot's answer to line 0 lasts 800 ms; line 1 plans to cut in 2,500 ms into it, with the
    # default 15 s maxMs. The caller should speak once the bot has been quiet for 700 ms.
    behaviour = {"default_turn": {"delay_ms": 600, "reply_ms": 800}}
    script = {
        "name": "interruption not reached",
        "lines": [
            {"text": "first", "synth": {"ms": 700, "seed": 1}, "wait": {"mode": "fixed", "ms": 0}},
            {"text": "cut in", "synth": {"ms": 600, "seed": 2}, "interrupt_after_bot_ms": 2500},
        ],
        "end": {"mode": "bot_done", "silence_ms": 600, "max_ms": 4000},
    }
    result, _ = await call("pcm16", 16000, tmp_path, behaviour=behaviour, script=script, hook=None)
    assert result.ended_by == "script", result.errors
    events = result.events

    (missed,) = by_type(events, "interrupt_not_reached")
    assert missed["index"] == 1 and 600 <= missed["bot_spoke_ms"] <= 900
    (ended,) = [e for e in by_type(events, "wait_end") if e["index"] == 1]
    assert ended["reason"] == "bot_done" and ended["t"] == missed["t"]
    (line_end,) = [e for e in by_type(events, "line_end") if e["index"] == 0]
    (line_start,) = [e for e in by_type(events, "line_start") if e["index"] == 1]
    # 600 ms to the answer, 800 ms of it, 700 ms of quiet: about 2.1 s, nowhere near the 15 s maxMs.
    assert 1900 <= line_start["t"] - line_end["t"] <= 2800

    card = analyze(result.out_dir)
    assert card["analysis"] == {"version": "timing-v2"}
    first, second = card["turns"]
    assert second["interruption_planned"] is True and second["interruption_status"] == "not_reached"
    assert second["interruption_stop_ms"] is None and second["interruption_bot_spoke_ms"] == missed["bot_spoke_ms"]
    assert second["response_delay_ms"] == pytest.approx(600, abs=60)
    assert first["silence_ms"] == 0 and card["call"]["max_silence_ms"] == 0
    assert card["call"]["interruption_not_reached_count"] == 1

    outcome = type("Outcome", (), {"scorecard": card, "error": None, "call_index": 0, "scenario_name": "x",
                                   "scenario_id": None, "repeat": 0, "duration_ms": None, "problems": [],
                                   "recording": None})()
    assert ("  interruptions    1 planned; 1 not reached: the bot finished before the planned interruption "
            "point") in call_lines(outcome)
