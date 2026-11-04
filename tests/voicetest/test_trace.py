"""The voice.call trace: built from a scorecard, sent through a client the test owns, captured before
export. Nothing reaches a backend: the export function is replaced."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

pytest.importorskip("evalkit")

from evalkit.core.client import EvalKitClient  # noqa: E402
from evalkit.core.types import EvalKitOptions  # noqa: E402

from evalkit.voicetest.trace import build_call_spans, call_attributes, send_call_trace, turn_attributes  # noqa: E402

T0 = 1_793_000_000_000.0
NS = "evalkit.voice.v1."

CALL_KEYS = {
    "transport", "sample_rate", "duration_ms", "first_word_ms", "response_delay_p50_ms",
    "response_delay_p95_ms", "talk_over_ms", "talk_over_count", "max_silence_ms", "vad_version",
    "recording_uri", "caller_health", "session",
}
TURN_KEYS = {
    "turn.index", "caller.speech_start_ms", "caller.speech_end_ms", "bot.speech_start_ms",
    "bot.speech_end_ms", "response_delay_ms", "talk_over_ms", "interruption.planned",
    "interruption.stop_ms", "silence_ms", "caller_unhealthy", "transcript.caller", "transcript.bot",
}


def _turn(index, lo, hi, line, bot, **extra):
    turn = {
        "index": index,
        "text": f"line {index}",
        "window": {"start_ms": lo, "end_ms": hi},
        "caller": {"speech_start_ms": line[0], "speech_end_ms": line[1]},
        "bot": {"speech_start_ms": bot[0] if bot else None, "speech_end_ms": bot[1] if bot else None},
        "response_delay_ms": bot[0] - line[1] if bot else None,
        "talk_over_ms": 0,
        "talk_over_count": 0,
        "interruption_planned": False,
        "interruption_stop_ms": None,
        "silence_ms": 0,
        "caller_unhealthy": False,
    }
    turn.update(extra)
    return turn


def _scorecard():
    return {
        "schema": "evalkit.voice.scorecard.v1",
        "vad": {"version": "webrtc-v1"},
        "call": {
            "session": "sim1:run2:scn3:0",
            "transport": "pcm16",
            "sample_rate": 16000,
            "t0_epoch_ms": T0,
            "duration_ms": 9000,
            "first_word_ms": 1800,
            "response_delay_p50_ms": 600,
            "response_delay_p95_ms": 800,
            "talk_over_ms": 0,
            "talk_over_count": 0,
            "max_silence_ms": 0,
        },
        "turns": [
            _turn(0, 500, 4000, (500, 1200), (1800, 3000)),
            _turn(1, 4000, 9000, (4000, 5000), (5800, 7000), interruption_planned=True, interruption_stop_ms=320),
            _turn(2, 9000, 9000, (9000, 9000), None, caller_unhealthy=True),
        ],
        "caller_health": {"healthy": False, "unhealthy_turns": [2], "vad_crosscheck": {"lines": [1, 2], "unmatched_segments": 0}},
    }


def _ms(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000.0


def test_spans_sit_at_the_recordings_times_and_carry_the_session():
    spans = build_call_spans(_scorecard(), recording_uri="file:///tmp/rec.wav")
    root, turns = spans[0], spans[1:]
    assert root.operation == "voice.call" and root.parent_span_id is None
    assert [s.operation for s in turns] == ["voice.turn"] * 3
    assert {s.trace_id for s in spans} == {root.trace_id}
    assert all(s.parent_span_id == root.span_id for s in turns)
    assert all(s.session_id == "sim1:run2:scn3:0" for s in spans)
    assert _ms(root.start_time) == pytest.approx(T0, abs=0.01)
    assert _ms(root.end_time) == pytest.approx(T0 + 9000, abs=0.01)
    assert _ms(turns[0].start_time) == pytest.approx(T0 + 500, abs=0.01)
    assert _ms(turns[0].end_time) == pytest.approx(T0 + 4000, abs=0.01)
    assert _ms(turns[1].start_time) == pytest.approx(T0 + 4000, abs=0.01)
    assert root.agent_name == ""


def test_attribute_names_match_the_contract():
    card = _scorecard()
    attrs = call_attributes(card, "file:///tmp/rec.wav")
    assert {k[len(NS):] for k in attrs} == CALL_KEYS
    assert attrs[NS + "session"] == "sim1:run2:scn3:0"
    assert attrs[NS + "vad_version"] == "webrtc-v1"
    health = json.loads(attrs[NS + "caller_health"])
    assert health["unhealthy_turns"] == [2] and "lines" not in health["vad_crosscheck"]

    first = turn_attributes(card["turns"][0])
    assert {k[len(NS):] for k in first} == TURN_KEYS - {"interruption.stop_ms"}
    assert first[NS + "response_delay_ms"] == 600 and first[NS + "transcript.bot"] == ""
    second = turn_attributes(card["turns"][1])
    assert second[NS + "interruption.planned"] is True and second[NS + "interruption.stop_ms"] == 320
    third = turn_attributes(card["turns"][2])
    # No bot speech: the bot figures are absent, never null.
    assert NS + "response_delay_ms" not in third and NS + "bot.speech_start_ms" not in third
    assert third[NS + "caller_unhealthy"] is True


def test_a_scorecard_without_t0_is_refused():
    card = _scorecard()
    card["call"]["t0_epoch_ms"] = None
    with pytest.raises(ValueError, match="t0"):
        build_call_spans(card)


@pytest.fixture
def capturing_client():
    def make(**opts):
        client = EvalKitClient(
            EvalKitOptions(subscription_key="tk_test", base_url="http://127.0.0.1:19599", service_name="voice-ci", **opts)
        )
        captured = []
        client._exporter.export = lambda batch: (captured.extend(batch), True)[1]
        made.append(client)
        return client, captured

    made = []
    yield make
    for client in made:
        client.destroy()


def test_sends_through_the_developers_client(capturing_client):
    client, captured = capturing_client(environment="ci")
    trace_id = send_call_trace(_scorecard(), client, recording_uri="file:///tmp/rec.wav")
    assert len(captured) == 4
    by_name = {}
    for envelope in captured:
        by_name.setdefault(envelope.operation, []).append(envelope)
    root = by_name["voice.call"][0]
    assert root.trace_id == trace_id
    assert len(by_name["voice.turn"]) == 3
    payload = root.to_dict()
    assert payload["sessionId"] == "sim1:run2:scn3:0"
    assert payload["serviceName"] == "voice-ci"
    attrs = payload["attributes"]
    assert attrs[NS + "transport"] == "pcm16" and attrs[NS + "duration_ms"] == 9000
    assert attrs["evalkit.environment"] == "ci" and "evalkit.sdk_version" in attrs
    assert sorted(e.attributes[NS + "turn.index"] for e in by_name["voice.turn"]) == [0, 1, 2]
    turn = next(e for e in by_name["voice.turn"] if e.attributes[NS + "turn.index"] == 0)
    assert turn.attributes[NS + "transcript.caller"] == "line 0"


def test_the_clients_content_setting_applies(capturing_client):
    client, captured = capturing_client(capture_content=False)
    send_call_trace(_scorecard(), client)
    turns = [e for e in captured if e.operation == "voice.turn"]
    assert turns and all(NS + "transcript.caller" not in e.attributes for e in turns)
    assert all(NS + "response_delay_ms" in e.attributes for e in turns[:2])


def test_without_a_client_it_needs_evalkit_init(monkeypatch):
    import evalkit

    monkeypatch.setattr(evalkit, "_instance", None)
    with pytest.raises(RuntimeError, match="evalkit.init"):
        send_call_trace(_scorecard())
