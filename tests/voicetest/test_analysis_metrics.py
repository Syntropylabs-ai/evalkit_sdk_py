"""CONTRACT section 5 definitions on hand-built timelines: no audio, no VAD."""

from __future__ import annotations

from evalkit.voicetest.analysis.intervals import gaps, normalize, percentile, subtract
from evalkit.voicetest.analysis.metrics import (
    CallerLine,
    CallTimeline,
    Wait,
    compute_call,
    compute_caller_health,
    compute_preamble,
    compute_turns,
    timeline_from_events,
)
from evalkit.voicetest.analysis.vad import get_spec, segments_from_frames


def _timeline(lines, duration=20000, waits=(), pacing=(), think=None, dropped=()):
    return CallTimeline(
        duration_ms=duration,
        lines=[CallerLine(index=i, start_ms=s, end_ms=e, **extra) for i, (s, e, extra) in enumerate(lines)],
        waits=[Wait(s, e, i) for s, e, i in waits],
        pacing=list(pacing),
        think_ms=dict(think or {}),
        dropped_frames=list(dropped),
    )


def _one(timeline, bot, tol=0):
    turns = compute_turns(timeline, bot, tol)
    return turns, compute_call(timeline, bot, turns)


# --- intervals --------------------------------------------------------------------------------------------

def test_interval_helpers():
    assert normalize([(5, 9), (0, 3), (3, 4), (8, 12), (20, 20)]) == [(0, 4), (5, 12)]
    assert subtract([(0, 100)], [(10, 20), (50, 200)]) == [(0, 10), (20, 50)]
    assert gaps([(10, 20), (30, 40)], 0, 50) == [(0, 10), (20, 30), (40, 50)]


def test_percentile_is_nearest_rank():
    assert percentile([], 95) is None
    assert percentile([7], 95) == 7
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile(list(range(1, 21)), 95) == 19
    assert percentile(list(range(1, 21)), 100) == 20


# --- segmentation rules -----------------------------------------------------------------------------------

def _frames(pattern):
    return [c == "#" for c in pattern]


def test_segment_needs_three_voiced_frames_to_start():
    spec = get_spec("webrtc-v1")
    assert segments_from_frames(_frames("##.##." + "." * 20), spec) == []
    assert segments_from_frames(_frames("..#####" + "." * 20), spec) == [(40, 140)]


def test_segment_ends_after_fifteen_unvoiced_frames_and_bridges_shorter_gaps():
    spec = get_spec("webrtc-v1")
    bridged = "#####" + "." * 14 + "#####" + "." * 15
    assert segments_from_frames(_frames(bridged), spec) == [(0, 480)]
    split = "#####" + "." * 15 + "#####" + "." * 15
    assert segments_from_frames(_frames(split), spec) == [(0, 100), (400, 500)]


def test_segments_under_100_ms_are_dropped_and_a_trailing_segment_is_kept():
    spec = get_spec("webrtc-v1")
    assert segments_from_frames(_frames("####" + "." * 20), spec) == []
    assert segments_from_frames(_frames("." * 3 + "#" * 6), spec) == [(60, 180)]


def test_vad_versions_are_frozen():
    v1 = get_spec("webrtc-v1")
    assert (v1.aggressiveness, v1.frame_ms, v1.start_frames, v1.end_frames, v1.min_segment_ms) == (2, 20, 3, 15, 100)
    assert v1.passes == "forward" and v1.onset_tolerance_ms == 0
    v2 = get_spec("webrtc-v2")
    assert v2.passes == "forward_and_reverse" and v2.onset_tolerance_ms == 20


# --- per-turn metrics -------------------------------------------------------------------------------------

def test_response_delay_is_first_bot_start_after_the_line_end():
    tl = _timeline([(1000, 2000, {}), (6000, 7000, {})])
    turns, call = _one(tl, [(2600, 3500), (4000, 4500), (7300, 8000)])
    assert [t["response_delay_ms"] for t in turns] == [600, 300]
    assert turns[0]["bot"] == {"speech_start_ms": 2600, "speech_end_ms": 4500, "segments": [[2600, 3500], [4000, 4500]]}
    assert turns[0]["bot_speech_ms"] == 1400
    assert call["response_delay_p50_ms"] == 300 and call["response_delay_p95_ms"] == 600
    assert call["first_word_ms"] == 2600 and call["bot_spoke_first"] is False


def test_no_bot_speech():
    tl = _timeline([(500, 1500, {})], duration=9000, waits=[(1500, 9000, 1)])
    turns, call = _one(tl, [])
    t = turns[0]
    assert t["response_delay_ms"] is None
    assert (t["talk_over_ms"], t["talk_over_count"], t["bot_speech_ms"]) == (0, 0, 0)
    assert t["bot"]["speech_start_ms"] is None
    # The caller's whole wait came before any bot word: that gap is not dead air.
    assert t["silence_ms"] == 0
    assert call["first_word_ms"] is None
    assert call["response_delay_p50_ms"] is None and call["response_delay_p95_ms"] is None
    assert call["max_silence_ms"] == 0


def test_bot_speaking_first():
    tl = _timeline([(3000, 4000, {})], duration=8000, waits=[(0, 3000, 0), (4000, 8000, 1)])
    bot = [(400, 2200), (4500, 6000)]
    turns, call = _one(tl, bot)
    assert call["first_word_ms"] == 400 and call["bot_spoke_first"] is True
    pre = compute_preamble(tl, bot)
    assert pre["bot_speech_ms"] == 1800 and pre["segments"] == [[400, 2200]]
    assert turns[0]["response_delay_ms"] == 500
    assert turns[0]["talk_over_count"] == 0 and turns[0]["caller_overlap_ms"] == 0


def test_bot_talking_over_the_caller():
    tl = _timeline([(1000, 3000, {})])
    turns, call = _one(tl, [(1500, 1900), (2700, 4000)])
    t = turns[0]
    assert t["talk_over_count"] == 2
    assert t["talk_over_ms"] == 400 + 300
    # Both segments started before the line ended, so none is a response.
    assert t["response_delay_ms"] is None
    assert call["talk_over_ms"] == 700 and call["talk_over_count"] == 2


def test_overlap_at_a_turn_boundary_is_the_callers():
    # The bot's reply to line 0 is still going when line 1 starts (unplanned).
    tl = _timeline([(1000, 2000, {}), (4000, 5000, {})], duration=9000)
    turns, call = _one(tl, [(2500, 4600), (5600, 6000)])
    first, second = turns
    assert first["bot_speech_ms"] == 1500  # clipped at line 1's start
    assert first["bot"]["speech_end_ms"] == 4600  # the segment's real end
    assert second["bot_speech_ms"] == 600 + 400
    assert second["caller_overlap_ms"] == 600
    assert second["talk_over_ms"] == 0 and second["talk_over_count"] == 0
    assert second["response_delay_ms"] == 600
    assert call["talk_over_ms"] == 0 and call["caller_overlap_ms"] == 600


def test_planned_interruption_stop_time_and_its_overlap_is_not_talk_over():
    planned = {"planned_interruption": True, "after_bot_ms": 800}
    tl = _timeline([(1000, 2000, {}), (3400, 5000, planned)], duration=9000)
    turns, call = _one(tl, [(2600, 3750), (5400, 6000)])
    t = turns[1]
    assert t["interruption_planned"] is True and t["interruption_after_bot_ms"] == 800
    assert t["interruption_stop_ms"] == 350
    assert t["interruption_overlap_ms"] == 350
    assert t["talk_over_ms"] == 0 and t["talk_over_count"] == 0 and t["caller_overlap_ms"] == 0
    assert call["interruption_count"] == 1 and call["interruption_stop_p50_ms"] == 350


def test_interruption_while_the_bot_is_silent():
    planned = {"planned_interruption": True, "after_bot_ms": 2000}
    tl = _timeline([(1000, 2000, {}), (4000, 5000, planned)], duration=9000)
    turns, _ = _one(tl, [(2500, 3200), (5500, 6000)])
    t = turns[1]
    assert t["interruption_planned"] is True
    assert t["interruption_stop_ms"] is None
    assert t["interruption_overlap_ms"] == 0


def test_an_interruption_the_bot_never_reached_is_scored_as_a_normal_turn():
    # The bot answered line 0 in 600 ms, short of the planned 2,000 ms, so line 1 was said 700 ms after.
    planned = {"planned_interruption": True, "after_bot_ms": 2000, "interruption_status": "not_reached",
               "bot_spoke_ms": 600}
    tl = _timeline([(1000, 2000, {}), (3300, 4300, planned)], duration=9000)
    turns, call = _one(tl, [(2000, 2600), (4800, 5400)])
    t = turns[1]
    assert t["interruption_planned"] is True and t["interruption_status"] == "not_reached"
    assert t["interruption_bot_spoke_ms"] == 600
    assert t["interruption_stop_ms"] is None and t["interruption_overlap_ms"] == 0
    assert call["interruption_count"] == 1 and call["interruption_not_reached_count"] == 1
    assert call["interruption_stop_p50_ms"] is None


def test_a_bot_restarting_during_an_interruption_is_talk_over():
    planned = {"planned_interruption": True, "after_bot_ms": 500}
    tl = _timeline([(1000, 2000, {}), (3000, 5000, planned)], duration=9000)
    turns, _ = _one(tl, [(2500, 3200), (3800, 4200)])
    t = turns[1]
    assert t["interruption_stop_ms"] == 200
    assert t["talk_over_count"] == 1 and t["talk_over_ms"] == 400


def test_unhealthy_caller_turn_is_flagged_and_left_out_of_timing():
    tl = _timeline(
        [(1000, 2000, {}), (4000, 5000, {})],
        duration=9000,
        pacing=[(1500, 12.0), (4200, 61.0)],
    )
    turns, call = _one(tl, [(2400, 3300), (6500, 7000)])
    assert [t["caller_error_ms"] for t in turns] == [12, 61]
    assert [t["caller_unhealthy"] for t in turns] == [False, True]
    assert turns[1]["response_delay_ms"] == 1500  # still measured, but not scored
    assert call["response_delay_p50_ms"] == 400 and call["response_delay_p95_ms"] == 400
    assert call["unscored_turns"] == [1] and call["scored_turn_count"] == 1
    health = compute_caller_health(tl, turns)
    assert health["unhealthy_turns"] == [1] and health["healthy"] is False
    assert health["late_frames"] == 2 and health["late_max_ms"] == 61


def test_dropped_frames_count_as_caller_error():
    tl = _timeline([(1000, 2000, {})], duration=5000, dropped=[(1200, 3)])
    turns, _ = _one(tl, [(2500, 3000)])
    assert turns[0]["caller_error_ms"] == 60 and turns[0]["caller_unhealthy"] is True


def test_silence_counts_dead_air_but_not_the_wait_before_the_first_word_or_think_time():
    # Line 0 ends at 2000; the bot answers 2,900 ms later (a response delay, not dead air), pauses
    # 3,500 ms mid-reply (dead air), and the caller thinks 3,200 ms before line 1.
    tl = _timeline(
        [(1000, 2000, {}), (14000, 15000, {})],
        duration=16000,
        waits=[(2000, 10800, 1)],
        think={1: 3200},
    )
    bot = [(4900, 5800), (9300, 9900)]
    turns, call = _one(tl, bot)
    t = turns[0]
    assert t["response_delay_ms"] == 2900
    assert t["silences"] == [[5800, 9300]]
    assert t["silence_ms"] == 3500
    assert call["max_silence_ms"] == 3500
    # The 4,100 ms between the bot's last word and line 1 is the caller's, think event or not.
    tl.think_ms = {}
    assert compute_turns(tl, bot)[0]["silence_ms"] == 3500


def test_the_callers_own_waits_are_not_silence_but_the_bot_going_quiet_mid_reply_is():
    # Line 0 waits fixed 6 s after the bot's short answer, line 1 waits with silenceMs 5000, and the call
    # ends 8 s after the bot's last answer: every long gap follows the bot's last word in its turn.
    tl = _timeline(
        [(0, 1000, {}), (8500, 9500, {}), (16000, 17000, {})],
        duration=27000,
        waits=[(1000, 8500, 1), (9500, 16000, 2), (17000, 27000, 3)],
    )
    bot = [(1500, 2500), (10000, 11000), (17500, 19000)]
    turns, call = _one(tl, bot)
    assert [t["silence_ms"] for t in turns] == [0, 0, 0]
    assert call["max_silence_ms"] == 0
    # The bot stops for 3,600 ms in the middle of its answer to line 2: that gap is the bot's.
    turns, call = _one(tl, [(1500, 2500), (10000, 11000), (17500, 18400), (22000, 23000)])
    assert [t["silence_ms"] for t in turns] == [0, 0, 3600]
    assert turns[2]["silences"] == [[18400, 22000]]
    assert call["max_silence_ms"] == 3600


def test_a_talk_over_does_not_pull_the_response_gap_into_the_reply():
    # The bot says something while the caller speaks, then answers 4 s after the line ends.
    tl = _timeline([(1000, 3000, {})], duration=12000)
    turns, _ = _one(tl, [(1500, 1800), (7000, 8000)])
    assert turns[0]["response_delay_ms"] == 4000 and turns[0]["talk_over_count"] == 1
    assert turns[0]["silence_ms"] == 0


def test_gaps_of_exactly_three_seconds_are_not_silence():
    tl = _timeline([(0, 1000, {}), (9000, 9500, {})], duration=10000, waits=[(1000, 1200, 1)])
    turns, _ = _one(tl, [(1200, 2000), (5000, 6000)])
    assert turns[0]["silence_ms"] == 0
    turns, _ = _one(tl, [(1200, 2000), (5001, 6000)])
    assert turns[0]["silence_ms"] == 3001


def test_onset_tolerance_turns_an_instant_answer_into_a_response():
    tl = _timeline([(1000, 2000, {})], duration=5000)
    bot = [(1990, 2600)]
    strict = compute_turns(tl, bot, 0)[0]
    assert strict["response_delay_ms"] is None and strict["talk_over_count"] == 1
    tolerant = compute_turns(tl, bot, 20)[0]
    assert tolerant["response_delay_ms"] == 0 and tolerant["talk_over_count"] == 0
    # A real talk-over is still one.
    assert compute_turns(tl, [(1900, 2600)], 20)[0]["talk_over_count"] == 1


# --- events -----------------------------------------------------------------------------------------------

def test_timeline_from_caller_events():
    events = [
        {"t": 0, "type": "connect"},
        {"t": 0, "type": "wait_start", "index": 0, "reason": "bot_done"},
        {"t": 900, "type": "wait_end", "index": 0, "reason": "bot_done"},
        {"t": 900, "type": "think", "index": 0, "ms": 0},
        {"t": 900, "type": "line_start", "index": 0, "text": "hello"},
        {"t": 1700, "type": "line_end", "index": 0, "text": "hello"},
        {"t": 1700, "type": "wait_start", "index": 1, "reason": "interrupt", "after_bot_ms": 500},
        {"t": 1700, "type": "interrupt_planned", "index": 1, "after_bot_ms": 500},
        {"t": 2400, "type": "pacing", "late_ms": 7.4, "frame": 120},
        {"t": 2500, "type": "bot_chunk", "bytes": 3200, "queued_ms": 900},
        {"t": 3000, "type": "wait_end", "index": 1, "reason": "interrupt"},
        {"t": 3000, "type": "line_start", "index": 1, "text": "wait"},
        {"t": 3100, "type": "clear", "dropped_ms": 640},
        {"t": 3500, "type": "error", "message": "boom"},
    ]
    tl = timeline_from_events(events, 4000)
    assert [(l.index, l.start_ms, l.end_ms, l.truncated, l.planned_interruption) for l in tl.lines] == [
        (0, 900, 1700, False, False),
        (1, 3000, 4000, True, True),
    ]
    assert tl.lines[1].after_bot_ms == 500
    assert [(w.start_ms, w.end_ms, w.index) for w in tl.waits] == [(0, 900, 0), (1700, 3000, 1)]
    assert tl.pacing == [(2400, 7)]
    assert tl.queue_depths_ms == [900] and tl.bot_chunk_count == 1
    assert tl.clears == [(3100, 640)] and tl.errors == ["boom"]


def test_a_planned_interruptions_status_comes_from_why_its_wait_ended():
    events = [
        {"t": 0, "type": "line_start", "index": 0, "text": "a"},
        {"t": 1000, "type": "line_end", "index": 0, "text": "a"},
        {"t": 1000, "type": "wait_start", "index": 1, "reason": "interrupt", "after_bot_ms": 500},
        {"t": 1000, "type": "interrupt_planned", "index": 1, "after_bot_ms": 500},
        {"t": 2200, "type": "wait_end", "index": 1, "reason": "interrupt"},
        {"t": 2200, "type": "line_start", "index": 1, "text": "b"},
        {"t": 3000, "type": "line_end", "index": 1, "text": "b"},
        {"t": 3000, "type": "wait_start", "index": 2, "reason": "interrupt", "after_bot_ms": 4000},
        {"t": 3000, "type": "interrupt_planned", "index": 2, "after_bot_ms": 4000},
        {"t": 5100, "type": "interrupt_not_reached", "index": 2, "bot_spoke_ms": 900},
        {"t": 5100, "type": "wait_end", "index": 2, "reason": "bot_done"},
        {"t": 5100, "type": "line_start", "index": 2, "text": "c"},
        {"t": 6000, "type": "line_end", "index": 2, "text": "c"},
        {"t": 6000, "type": "wait_start", "index": 3, "reason": "interrupt", "after_bot_ms": 500},
        {"t": 6000, "type": "interrupt_planned", "index": 3, "after_bot_ms": 500},
        {"t": 21000, "type": "wait_end", "index": 3, "reason": "max_wait"},
        {"t": 21000, "type": "line_start", "index": 3, "text": "d"},
        {"t": 22000, "type": "line_end", "index": 3, "text": "d"},
    ]
    tl = timeline_from_events(events, 23000)
    assert [(l.interruption_status, l.bot_spoke_ms) for l in tl.lines] == [
        (None, None), ("reached", None), ("not_reached", 900), ("max_wait", None),
    ]
    assert [(w.reason, w.end_reason) for w in tl.waits] == [
        ("interrupt", "interrupt"), ("interrupt", "bot_done"), ("interrupt", "max_wait"),
    ]
    turns = compute_turns(tl, [(1200, 2500), (3500, 4400)])
    assert [t["interruption_status"] for t in turns] == [None, "reached", "not_reached", "max_wait"]
    assert turns[1]["interruption_stop_ms"] == 300
    assert turns[2]["interruption_bot_spoke_ms"] == 900


# --- dynamic calls ------------------------------------------------------------------------------------------

def _dynamic_events():
    """A dynamic call's caller events: the opening, then a line decided in 3.6 s while the bot, which had
    answered, spoke again 3.2 s after its answer ended."""
    return [
        {"type": "line_start", "t": 0, "index": 0, "text": "Hi there."},
        {"type": "line_end", "t": 800, "index": 0},
        {"type": "turn_request", "t": 2600, "index": 1, "retake": 0, "attempt": 0},
        {"type": "turn_failed", "t": 2900, "index": 1, "status": 429},
        {"type": "turn_request", "t": 3900, "index": 1, "retake": 0, "attempt": 1},
        {"type": "turn_answer", "t": 5000, "index": 1, "retake": 0, "heard": "How can I help?"},
        {"type": "retake", "t": 5000, "index": 1, "bot_spoke_ms": 1100},
        {"type": "turn_request", "t": 5000, "index": 1, "retake": 1, "attempt": 0},
        {"type": "turn_answer", "t": 6200, "index": 1, "retake": 1, "heard": "How can I help? Hello?"},
        {"type": "think", "t": 6200, "index": 1, "ms": 3600},
        {"type": "line_start", "t": 6200, "index": 1, "text": "I was charged twice."},
        {"type": "line_end", "t": 7400, "index": 1},
        {"type": "turn_answer", "t": 10000, "index": 2, "retake": 0, "heard": "Refunded.", "final": True},
    ]


def test_a_dynamic_calls_think_time_is_never_silence_and_its_transcript_is_kept():
    events = _dynamic_events()
    # bot: answers 800-1900, then speaks again at 5100-5700 while the caller decides; answers line 1 at 8000
    bot = [(1300, 1900), (5100, 5700), (8000, 9000)]
    timeline = timeline_from_events(events, 10000, mode="dynamic")
    turns = compute_turns(timeline, bot)
    # the 3.2 s gap 1900-5100 lies inside the caller's think time (2600-6200) less 700 ms: not silence
    assert [t["silence_ms"] for t in turns] == [0, 0]
    undecided = compute_turns(timeline_from_events([e for e in events if e["type"] != "think"], 10000), bot)
    assert undecided[0]["silence_ms"] == 3200  # the same gap, were it not the caller deciding
    assert [t["heard"] for t in turns] == ["How can I help? Hello?", "Refunded."]
    assert [t["caller_text"] for t in turns] == ["Hi there.", "I was charged twice."]
    assert [t["caller"]["think_ms"] for t in turns] == [0, 3600]
    assert [t["response_delay_ms"] for t in turns] == [500, 600]
    call = compute_call(timeline, bot, turns)
    assert call["mode"] == "dynamic"
    assert call["transcript"] == [
        {"turn": 0, "role": "caller", "text": "Hi there."},
        {"turn": 0, "role": "bot", "text": "How can I help? Hello?"},
        {"turn": 1, "role": "caller", "text": "I was charged twice.", "think_ms": 3600},
        {"turn": 1, "role": "bot", "text": "Refunded."},
    ]
    health = compute_caller_health(timeline, turns)
    assert (health["think_ms_p50"], health["think_ms_max"]) == (3600, 3600)
    assert (health["turn_requests"], health["turn_failures"], health["retakes"]) == (2, 1, 1)
    assert health["healthy"] is True  # deciding is not lateness


def test_a_scripted_call_has_no_transcript_of_the_bot():
    timeline = timeline_from_events([{"type": "line_start", "t": 0, "index": 0, "text": "Hi"},
                                     {"type": "think", "t": 0, "index": 0, "ms": 0},
                                     {"type": "line_end", "t": 500, "index": 0}], 3000)
    turns = compute_turns(timeline, [(900, 1500)])
    assert turns[0]["heard"] is None and turns[0]["caller_text"] == "Hi"
    call = compute_call(timeline, [(900, 1500)], turns)
    assert call["mode"] == "scripted" and call["transcript"] is None
    assert compute_caller_health(timeline, turns)["think_ms_p50"] is None
