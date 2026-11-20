"""Turns, per-turn and call-level metrics, and caller health (CONTRACT section 5).

Everything here works on integer-ms segment lists, never on audio, so the definitions can be tested on
hand-built timelines and applied unchanged to a calibration call's ground truth.

Who caused an overlap is decided by who started second: a bot segment that starts while the caller is
speaking is the bot talking over the caller (``talk_over_*``); caller speech that starts while a bot
segment is already in progress is the caller's doing, either a planned interruption (scored by
``interruption_stop_ms``) or an unplanned one (``caller_overlap_ms``, never the bot's score).

Silence is the bot's only: a gap counts when it falls inside the bot's reply, from its first speech after
the caller's line to its last in the turn. Before and after the reply the caller is the one waiting, by
its script's rules, and the response delay already measures the wait for the bot's first word. A dynamic
call's think time (the caller deciding its next line) is the caller's too, so no gap inside it counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .intervals import Interval, clip, gaps, normalize, overlap, percentile, total_overlap

# The definitions below, as the scorecard's `analysis.version` names them. A scorecard with no `analysis`
# section was made by `timing-v1`, which counted any gap over 3 s in the turn as silence, the caller's
# own waits after the bot finished included, and gave a planned interruption no status.
ANALYSIS_VERSION = "timing-v2"
SILENCE_MIN_MS = 3000
CALLER_ERROR_BUDGET_MS = 50
CALLER_FRAME_MS = 20

# A planned interruption's `interruption_status`, from the caller's events (CONTRACT section 4).
INTERRUPTION_REACHED = "reached"  # the caller cut in at the planned point
INTERRUPTION_NOT_REACHED = "not_reached"  # the bot finished first, so the line was a normal turn
INTERRUPTION_MAX_WAIT = "max_wait"  # the wait ran out first: the bot never spoke, or never paused


@dataclass
class CallerLine:
    index: int
    start_ms: int
    end_ms: int
    text: str = ""
    planned_interruption: bool = False
    after_bot_ms: Optional[int] = None
    # The call ended before the line's line_end: its end is the call end.
    truncated: bool = False
    # A planned interruption's outcome (INTERRUPTION_*); None when unknown, which is scored as reached.
    interruption_status: Optional[str] = None
    # From `interrupt_not_reached`: how far into its speech the bot got before it finished.
    bot_spoke_ms: Optional[int] = None


@dataclass
class Wait:
    start_ms: int
    end_ms: int
    index: Optional[int] = None
    reason: Optional[str] = None
    end_reason: Optional[str] = None


@dataclass
class CallTimeline:
    """What ``events.jsonl`` says about the caller's side, on the sample clock."""

    duration_ms: int
    lines: List[CallerLine] = field(default_factory=list)
    waits: List[Wait] = field(default_factory=list)
    think_ms: Dict[int, int] = field(default_factory=dict)
    pacing: List[Tuple[int, int]] = field(default_factory=list)
    dropped_frames: List[Tuple[int, int]] = field(default_factory=list)
    queue_depths_ms: List[int] = field(default_factory=list)
    bot_chunk_count: int = 0
    bot_chunk_bytes: int = 0
    clears: List[Tuple[int, int]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    # "scripted" or "dynamic" (CONTRACT section 4.1), from call.json's script.
    mode: str = "scripted"
    # A dynamic call's turn answers: what the bot said before line `index`, as the turn endpoint heard
    # it (the last answer for the index, so a retake's replaces the first). Before a cut-in line, which
    # is never asked for, it is what the next answer heard before the cut-in.
    heard: Dict[int, Optional[str]] = field(default_factory=dict)
    # Where the caller was deciding a line: [line_start - think ms, line_start].
    think_spans: List[Tuple[int, int]] = field(default_factory=list)
    turn_requests: int = 0
    turn_failures: int = 0
    retakes: int = 0


def _int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(round(value))
    return default


_DROP_EVENT_TYPES = ("dropped", "frame_dropped", "drop", "dropped_frames")


def timeline_from_events(events: Iterable[Dict[str, Any]], duration_ms: int, mode: str = "scripted") -> CallTimeline:
    timeline = CallTimeline(duration_ms=int(duration_ms), mode="dynamic" if mode == "dynamic" else "scripted")
    starts: Dict[int, Dict[str, Any]] = {}
    ends: Dict[int, int] = {}
    planned: Dict[int, Optional[int]] = {}
    not_reached: Dict[int, Optional[int]] = {}
    wait_ends: Dict[int, str] = {}
    open_waits: List[Tuple[Optional[int], int, Optional[str]]] = []
    answered: Set[int] = set()
    for event in events:
        kind = event.get("type")
        t = _int(event.get("t"), 0) or 0
        index = _int(event.get("index"))
        if kind == "line_start" and index is not None and index not in starts:
            starts[index] = {"t": t, "text": event.get("text") or ""}
        elif kind == "line_end" and index is not None and index not in ends:
            ends[index] = t
        elif kind == "interrupt_planned" and index is not None:
            planned[index] = _int(event.get("after_bot_ms"))
        elif kind == "interrupt_not_reached" and index is not None:
            not_reached[index] = _int(event.get("bot_spoke_ms"))
        elif kind == "wait_start":
            reason = event.get("reason")
            open_waits.append((index, t, reason if isinstance(reason, str) else None))
        elif kind == "wait_end":
            end_reason = event.get("reason") if isinstance(event.get("reason"), str) else None
            if index is not None and end_reason is not None:
                wait_ends.setdefault(index, end_reason)
            # The wait with the same index, else the most recent one still open.
            match = next(
                (k for k in range(len(open_waits) - 1, -1, -1) if index is not None and open_waits[k][0] == index),
                len(open_waits) - 1 if open_waits else None,
            )
            if match is not None:
                w_index, w_start, w_reason = open_waits.pop(match)
                timeline.waits.append(Wait(w_start, max(w_start, t), w_index, w_reason, end_reason))
        elif kind == "think" and index is not None:
            think = max(0, _int(event.get("ms"), 0) or 0)
            timeline.think_ms[index] = timeline.think_ms.get(index, 0) + think
            if think:
                timeline.think_spans.append((max(0, t - think), t))
        elif kind == "turn_answer" and index is not None:
            heard = event.get("heard")
            timeline.heard[index] = heard.strip() if isinstance(heard, str) and heard.strip() else None
            answered.add(index)
            # The first answer after a cut-in heard the bot's interrupted answer too: it is the bot's
            # speech before the cut-in line, the line `index - 1`, for which no request was made.
            before = event.get("heard_before_cut_in")
            if isinstance(before, str) and before.strip() and index - 1 not in answered:
                timeline.heard[index - 1] = before.strip()
        elif kind == "turn_request":
            if not event.get("attempt"):
                timeline.turn_requests += 1
        elif kind == "turn_failed":
            timeline.turn_failures += 1
        elif kind == "retake":
            timeline.retakes += 1
        elif kind == "pacing":
            late = _int(event.get("late_ms"), 0) or 0
            timeline.pacing.append((t, late))
            dropped = _int(event.get("dropped_frames"), None) or _int(event.get("dropped"), None)
            if dropped:
                timeline.dropped_frames.append((t, dropped))
        elif kind in _DROP_EVENT_TYPES:
            timeline.dropped_frames.append((t, _int(event.get("frames"), 1) or 1))
        elif kind == "bot_chunk":
            timeline.bot_chunk_count += 1
            timeline.bot_chunk_bytes += _int(event.get("bytes"), 0) or 0
            depth = _int(event.get("queued_ms"))
            if depth is not None:
                timeline.queue_depths_ms.append(depth)
        elif kind == "clear":
            timeline.clears.append((t, _int(event.get("dropped_ms"), 0) or 0))
        elif kind == "error":
            message = event.get("message") or event.get("error") or event.get("detail") or ""
            timeline.errors.append(str(message)[:200])
    for w_index, w_start, w_reason in open_waits:
        timeline.waits.append(Wait(w_start, max(w_start, timeline.duration_ms), w_index, w_reason))
    timeline.waits.sort(key=lambda w: w.start_ms)
    for index, start in starts.items():
        end = ends.get(index)
        truncated = end is None
        end_ms = timeline.duration_ms if end is None else end
        timeline.lines.append(
            CallerLine(
                index=index,
                start_ms=start["t"],
                end_ms=max(start["t"], end_ms),
                text=str(start["text"]),
                planned_interruption=index in planned,
                after_bot_ms=planned.get(index),
                truncated=truncated,
                interruption_status=_interruption_status(index, planned, not_reached, wait_ends),
                bot_spoke_ms=not_reached.get(index),
            )
        )
    timeline.lines.sort(key=lambda line: (line.start_ms, line.index))
    return timeline


def _interruption_status(
    index: int, planned: Dict[int, Optional[int]], not_reached: Dict[int, Optional[int]], wait_ends: Dict[int, str]
) -> Optional[str]:
    """What became of a planned interruption, by the caller's own record of why its wait ended."""
    if index not in planned:
        return None
    if index in not_reached:
        return INTERRUPTION_NOT_REACHED
    reason = wait_ends.get(index)
    if reason == "interrupt":
        return INTERRUPTION_REACHED
    return INTERRUPTION_MAX_WAIT if reason is not None else None


def _reply_silences(caller: Interval, reply: Sequence[Interval], bot: Sequence[Interval], lo: int, hi: int,
                    thinking: Sequence[Interval] = ()) -> List[Interval]:
    """Gaps over SILENCE_MIN_MS inside the bot's reply, from its first speech after the caller's line to
    its last in the turn, where neither side speaks and the caller is not deciding its next line. A turn
    with no reply has none: its wait is the response delay's, and an unanswered turn is already over that
    limit."""
    if not reply:
        return []
    start, end = max(lo, reply[0][0]), min(hi, max(e for _, e in reply))
    speech = normalize(clip([caller], lo, hi) + clip(bot, lo, hi) + clip(list(thinking), lo, hi))
    return [g for g in gaps(speech, start, end) if g[1] - g[0] > SILENCE_MIN_MS]


def compute_turns(
    timeline: CallTimeline, bot_segments: Iterable[Sequence[int]], onset_tolerance_ms: int = 0
) -> List[Dict[str, Any]]:
    """Per-turn metrics. ``onset_tolerance_ms`` comes from the VAD version (0 in webrtc-v1): a bot
    segment starting less than that before the caller's line end counts as starting at the line end."""
    tol = max(0, int(onset_tolerance_ms))
    bot: List[Interval] = sorted((int(s), int(e)) for s, e in bot_segments)
    lines = timeline.lines
    turns: List[Dict[str, Any]] = []
    for n, line in enumerate(lines):
        lo = line.start_ms
        hi = lines[n + 1].start_ms if n + 1 < len(lines) else timeline.duration_ms
        hi = max(hi, lo)
        caller = (line.start_ms, line.end_ms)

        assigned = [s for s in bot if lo <= s[0] < hi]
        # The bot's reply: from the speech the response delay is measured to, on through the turn.
        reply = [s for s in assigned if s[0] >= line.end_ms - tol]
        response = max(0, reply[0][0] - line.end_ms) if reply else None

        bot_started = [s for s in bot if line.start_ms <= s[0] < line.end_ms - tol]
        talk_over_ms = sum(overlap(s, caller) for s in bot_started)
        in_progress = [s for s in bot if s[0] < line.start_ms < s[1]]
        caller_started_ms = sum(overlap(s, caller) for s in in_progress)
        status = line.interruption_status if line.planned_interruption else None
        # A line whose interruption point the bot never reached was said as a normal turn, so it is
        # scored as one: no stop time, and any overlap is the caller's own.
        if line.planned_interruption and status != INTERRUPTION_NOT_REACHED:
            stop = in_progress[-1][1] - line.start_ms if in_progress else None
            interruption_overlap, caller_overlap = caller_started_ms, 0
        else:
            stop, interruption_overlap, caller_overlap = None, 0, caller_started_ms

        silences = _reply_silences(caller, reply, bot, lo, hi, timeline.think_spans)

        late = max((ms for t, ms in timeline.pacing if lo <= t < hi), default=0)
        late_count = sum(1 for t, ms in timeline.pacing if lo <= t < hi)
        dropped = sum(frames for t, frames in timeline.dropped_frames if lo <= t < hi)
        caller_error = max(late, dropped * CALLER_FRAME_MS)

        turns.append(
            {
                "index": line.index,
                "text": line.text,
                "caller_text": line.text,
                # What the bot said in the turn, as the turn endpoint heard it: the answer sent with the
                # request for the next line (CONTRACT section 5). None for a scripted call.
                "heard": timeline.heard.get(line.index + 1) if timeline.mode == "dynamic" else None,
                "window": {"start_ms": lo, "end_ms": hi},
                "caller": {
                    "speech_start_ms": line.start_ms,
                    "speech_end_ms": line.end_ms,
                    "truncated": line.truncated,
                    "think_ms": timeline.think_ms.get(line.index, 0),
                },
                "bot": {
                    "speech_start_ms": assigned[0][0] if assigned else None,
                    "speech_end_ms": assigned[-1][1] if assigned else None,
                    "segments": [list(s) for s in assigned],
                },
                "response_delay_ms": response,
                "talk_over_ms": talk_over_ms,
                "talk_over_count": len(bot_started),
                "caller_overlap_ms": caller_overlap,
                "interruption_planned": line.planned_interruption,
                "interruption_after_bot_ms": line.after_bot_ms,
                "interruption_status": status,
                "interruption_bot_spoke_ms": line.bot_spoke_ms if status == INTERRUPTION_NOT_REACHED else None,
                "interruption_stop_ms": stop,
                "interruption_overlap_ms": interruption_overlap,
                "silence_ms": sum(g[1] - g[0] for g in silences),
                "silences": [list(g) for g in silences],
                "bot_speech_ms": total_overlap(bot, (lo, hi)),
                "caller_error_ms": caller_error,
                "caller_late_frames": late_count,
                "caller_dropped_frames": dropped,
                "caller_unhealthy": caller_error > CALLER_ERROR_BUDGET_MS,
            }
        )
    return turns


def compute_preamble(timeline: CallTimeline, bot_segments: Iterable[Sequence[int]]) -> Dict[str, Any]:
    """Bot speech before the caller's first line: a greeting, for bots that speak first."""
    bot = sorted((int(s), int(e)) for s, e in bot_segments)
    end = timeline.lines[0].start_ms if timeline.lines else timeline.duration_ms
    before = [s for s in bot if s[0] < end]
    return {
        "window": {"start_ms": 0, "end_ms": end},
        "bot_speech_ms": total_overlap(bot, (0, end)),
        "segments": [list(s) for s in before],
    }


def _ints(values: Iterable[Optional[int]]) -> List[int]:
    return [v for v in values if v is not None]


def _pct(values: Sequence[int], p: float) -> Optional[int]:
    value = percentile(values, p)
    return None if value is None else int(value)


def compute_call(
    timeline: CallTimeline, bot_segments: Iterable[Sequence[int]], turns: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Call-level figures. Turns flagged ``caller_unhealthy`` are left out of every timing aggregate."""
    bot = sorted((int(s), int(e)) for s, e in bot_segments)
    scored = [t for t in turns if not t["caller_unhealthy"]]
    delays = _ints(t["response_delay_ms"] for t in scored)
    stops = _ints(t["interruption_stop_ms"] for t in scored if t["interruption_planned"])
    silences = [g[1] - g[0] for t in scored for g in t["silences"]]
    first_line = timeline.lines[0].start_ms if timeline.lines else None
    return {
        "duration_ms": timeline.duration_ms,
        "first_word_ms": bot[0][0] if bot else None,
        "bot_spoke_first": bool(bot) and (first_line is None or bot[0][0] < first_line),
        "response_delay_p50_ms": _pct(delays, 50),
        "response_delay_p95_ms": _pct(delays, 95),
        "talk_over_ms": sum(t["talk_over_ms"] for t in scored),
        "talk_over_count": sum(t["talk_over_count"] for t in scored),
        "max_silence_ms": max(silences, default=0),
        "silence_ms": sum(t["silence_ms"] for t in scored),
        "caller_overlap_ms": sum(t["caller_overlap_ms"] for t in scored),
        "interruption_count": sum(1 for t in turns if t["interruption_planned"]),
        "interruption_not_reached_count": sum(
            1 for t in turns if t.get("interruption_status") == INTERRUPTION_NOT_REACHED
        ),
        "interruption_stop_p50_ms": _pct(stops, 50),
        "interruption_stop_p95_ms": _pct(stops, 95),
        "bot_speech_ms": sum(e - s for s, e in bot),
        "turn_count": len(turns),
        "scored_turn_count": len(scored),
        "unscored_turns": [t["index"] for t in turns if t["caller_unhealthy"]],
        "mode": timeline.mode,
        "transcript": compute_transcript(timeline, turns),
    }


def compute_transcript(timeline: CallTimeline, turns: Sequence[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """A dynamic call's conversation in order: each caller line, then what the bot said after it as the
    turn endpoint heard it. A caller entry carries `think_ms` when its line was decided during the call.
    None for a scripted call, which has no transcript of the bot."""
    if timeline.mode != "dynamic":
        return None
    out: List[Dict[str, Any]] = []
    for turn in turns:
        entry: Dict[str, Any] = {"turn": turn["index"], "role": "caller", "text": turn.get("caller_text") or ""}
        think = (turn.get("caller") or {}).get("think_ms") or 0
        if think:
            entry["think_ms"] = think
        out.append(entry)
        if turn.get("heard"):
            out.append({"turn": turn["index"], "role": "bot", "text": turn["heard"]})
    return out


def compute_caller_health(timeline: CallTimeline, turns: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The caller's own figures, reported apart from the bot's scores."""
    think = list(timeline.think_ms.values())
    decided = sorted(ms for ms in think if ms > 0)
    late = [ms for _, ms in timeline.pacing]
    depths = timeline.queue_depths_ms
    unhealthy = [t["index"] for t in turns if t["caller_unhealthy"]]
    return {
        "healthy": not unhealthy and not timeline.errors,
        "think_ms_total": sum(think),
        "think_ms_max": max(think, default=0),
        # Over the lines decided during a dynamic call; None when every line was planned before it.
        "think_ms_p50": _pct(decided, 50),
        "turn_requests": timeline.turn_requests,
        "turn_failures": timeline.turn_failures,
        "retakes": timeline.retakes,
        "late_frames": len(late),
        "late_max_ms": max(late, default=0),
        "dropped_frames": sum(frames for _, frames in timeline.dropped_frames),
        "queue_depth_max_ms": max(depths, default=0),
        "queue_depth_p95_ms": _pct(depths, 95),
        "bot_chunks": timeline.bot_chunk_count,
        "bot_chunk_bytes": timeline.bot_chunk_bytes,
        "clears": len(timeline.clears),
        "clear_dropped_ms": sum(ms for _, ms in timeline.clears),
        "errors": len(timeline.errors),
        "error_messages": timeline.errors[:5],
        "turn_error_budget_ms": CALLER_ERROR_BUDGET_MS,
        "unhealthy_turns": unhealthy,
    }
