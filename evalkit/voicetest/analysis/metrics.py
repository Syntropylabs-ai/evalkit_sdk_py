"""Turns, per-turn and call-level metrics, and caller health (CONTRACT section 5).

Everything here works on integer-ms segment lists, never on audio, so the definitions can be tested on
hand-built timelines and applied unchanged to a calibration call's ground truth.

Who caused an overlap is decided by who started second: a bot segment that starts while the caller is
speaking is the bot talking over the caller (``talk_over_*``); caller speech that starts while a bot
segment is already in progress is the caller's doing, either a planned interruption (scored by
``interruption_stop_ms``) or an unplanned one (``caller_overlap_ms``, never the bot's score).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .intervals import Interval, clip, gaps, normalize, overlap, percentile, subtract, total_overlap

SILENCE_MIN_MS = 3000
CALLER_ERROR_BUDGET_MS = 50
CALLER_FRAME_MS = 20


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


@dataclass
class Wait:
    start_ms: int
    end_ms: int
    index: Optional[int] = None
    reason: Optional[str] = None


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


def _int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(round(value))
    return default


_DROP_EVENT_TYPES = ("dropped", "frame_dropped", "drop", "dropped_frames")


def timeline_from_events(events: Iterable[Dict[str, Any]], duration_ms: int) -> CallTimeline:
    timeline = CallTimeline(duration_ms=int(duration_ms))
    starts: Dict[int, Dict[str, Any]] = {}
    ends: Dict[int, int] = {}
    planned: Dict[int, Optional[int]] = {}
    open_waits: List[Tuple[Optional[int], int, Optional[str]]] = []
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
        elif kind == "wait_start":
            reason = event.get("reason")
            open_waits.append((index, t, reason if isinstance(reason, str) else None))
        elif kind == "wait_end":
            # The wait with the same index, else the most recent one still open.
            match = next(
                (k for k in range(len(open_waits) - 1, -1, -1) if index is not None and open_waits[k][0] == index),
                len(open_waits) - 1 if open_waits else None,
            )
            if match is not None:
                w_index, w_start, w_reason = open_waits.pop(match)
                timeline.waits.append(Wait(w_start, max(w_start, t), w_index, w_reason))
        elif kind == "think" and index is not None:
            timeline.think_ms[index] = timeline.think_ms.get(index, 0) + max(0, _int(event.get("ms"), 0) or 0)
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
            )
        )
    timeline.lines.sort(key=lambda line: (line.start_ms, line.index))
    return timeline


def _first_word_after(bot: Sequence[Interval], t: int) -> Optional[int]:
    """The earliest instant at or after ``t`` at which the bot is speaking."""
    for start, end in bot:
        if end > t:
            return max(start, t)
    return None


def _excluded_from_silence(
    timeline: CallTimeline, bot: Sequence[Interval], lines_by_index: Dict[int, CallerLine]
) -> List[Interval]:
    """Caller think time, and the caller's waiting before the bot's first word (that gap is the response
    delay, which is scored on its own)."""
    excluded: List[Interval] = []
    for index, ms in timeline.think_ms.items():
        line = lines_by_index.get(index)
        if line is not None and ms > 0:
            excluded.append((line.start_ms - ms, line.start_ms))
    for wait in timeline.waits:
        first = _first_word_after(bot, wait.start_ms)
        end = wait.end_ms if first is None else min(wait.end_ms, first)
        if end > wait.start_ms:
            excluded.append((wait.start_ms, end))
    return normalize(excluded)


def compute_turns(
    timeline: CallTimeline, bot_segments: Iterable[Sequence[int]], onset_tolerance_ms: int = 0
) -> List[Dict[str, Any]]:
    """Per-turn metrics. ``onset_tolerance_ms`` comes from the VAD version (0 in webrtc-v1): a bot
    segment starting less than that before the caller's line end counts as starting at the line end."""
    tol = max(0, int(onset_tolerance_ms))
    bot: List[Interval] = sorted((int(s), int(e)) for s, e in bot_segments)
    lines = timeline.lines
    lines_by_index = {line.index: line for line in lines}
    excluded = _excluded_from_silence(timeline, bot, lines_by_index)
    turns: List[Dict[str, Any]] = []
    for n, line in enumerate(lines):
        lo = line.start_ms
        hi = lines[n + 1].start_ms if n + 1 < len(lines) else timeline.duration_ms
        hi = max(hi, lo)
        caller = (line.start_ms, line.end_ms)

        assigned = [s for s in bot if lo <= s[0] < hi]
        response = next((max(0, s[0] - line.end_ms) for s in assigned if s[0] >= line.end_ms - tol), None)

        bot_started = [s for s in bot if line.start_ms <= s[0] < line.end_ms - tol]
        talk_over_ms = sum(overlap(s, caller) for s in bot_started)
        in_progress = [s for s in bot if s[0] < line.start_ms < s[1]]
        caller_started_ms = sum(overlap(s, caller) for s in in_progress)
        if line.planned_interruption:
            stop = in_progress[-1][1] - line.start_ms if in_progress else None
            interruption_overlap, caller_overlap = caller_started_ms, 0
        else:
            stop, interruption_overlap, caller_overlap = None, 0, caller_started_ms

        speech = normalize(clip([caller], lo, hi) + clip(bot, lo, hi))
        silent = subtract(gaps(speech, lo, hi), excluded)
        silences = [g for g in silent if g[1] - g[0] > SILENCE_MIN_MS]

        late = max((ms for t, ms in timeline.pacing if lo <= t < hi), default=0)
        late_count = sum(1 for t, ms in timeline.pacing if lo <= t < hi)
        dropped = sum(frames for t, frames in timeline.dropped_frames if lo <= t < hi)
        caller_error = max(late, dropped * CALLER_FRAME_MS)

        turns.append(
            {
                "index": line.index,
                "text": line.text,
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
        "interruption_stop_p50_ms": _pct(stops, 50),
        "interruption_stop_p95_ms": _pct(stops, 95),
        "bot_speech_ms": sum(e - s for s, e in bot),
        "turn_count": len(turns),
        "scored_turn_count": len(scored),
        "unscored_turns": [t["index"] for t in turns if t["caller_unhealthy"]],
    }


def compute_caller_health(timeline: CallTimeline, turns: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The caller's own figures, reported apart from the bot's scores."""
    think = list(timeline.think_ms.values())
    late = [ms for _, ms in timeline.pacing]
    depths = timeline.queue_depths_ms
    unhealthy = [t["index"] for t in turns if t["caller_unhealthy"]]
    return {
        "healthy": not unhealthy and not timeline.errors,
        "think_ms_total": sum(think),
        "think_ms_max": max(think, default=0),
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
