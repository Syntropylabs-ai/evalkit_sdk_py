"""The readable summary ``evalkit voice-test run`` prints: per call as it finishes, then the run.

The figures are the scorecard's (CONTRACT section 5), in ms as the evaluators display them. The run's
evaluator results are printed from what ``GET /v1/voice-tests/:runId`` returns for each (RUN_API section
4): its name, score and verdict, and its `figures` written out as the evaluators display them (section 6),
for example "p95 1,840 ms, limit 1,200 ms, 3 of 9 turns over".
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional

TASK_SUCCESS_KEY = "voice_task_success"
_EXCERPT_CHARS = 90
_KEY_SOURCES = {
    "request": "your key, sent with the run",
    "organization": "your organization's key",
    "platform": "the SyntropyLabs model",
}


def ms(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    return f"{int(round(value)):,} ms"


def _seconds(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    return f"{value / 1000.0:.1f} s"


def _bytes(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.1f} MB"
    return f"{value / 1024:.0f} kB"


def caller_text(caller: Any, minutes_left: Any) -> Optional[str]:
    """The model that decides the dynamic calls' lines and whose key it runs on, and the SyntropyLabs
    model minutes left this month; None when the plan names neither."""
    parts: List[str] = []
    if isinstance(caller, Mapping):
        name = "/".join(str(caller[k]) for k in ("provider", "model") if caller.get(k)) or "named by the API"
        source = _KEY_SOURCES.get(str(caller.get("keySource") or ""))
        parts.append(f"caller model: {name}" + (f" ({source})" if source else ""))
    if isinstance(minutes_left, (int, float)) and not isinstance(minutes_left, bool):
        parts.append("SyntropyLabs model minutes: unlimited" if minutes_left < 0
                     else f"{minutes_left:,.0f} SyntropyLabs model minutes left this month")
    return "; ".join(parts) or None


def _clip(text: Any, limit: int = _EXCERPT_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def conversation_lines(transcript: Any, turns_taken: Any) -> List[str]:
    """A dynamic call's conversation, short: how many turns, its first and last exchanges."""
    if not isinstance(transcript, list):
        return []
    entries = [e for e in transcript if isinstance(e, Mapping) and e.get("text")]
    taken = turns_taken if isinstance(turns_taken, int) else None
    head = "  conversation     " + (f"{_plural(taken, 'turn')} after the opening" if taken is not None
                                    else f"{len(entries)} entries")
    lines = [head]
    shown = entries if len(entries) <= 4 else entries[:2] + [None] + entries[-2:]
    for entry in shown:
        if entry is None:
            lines.append("    ...")
            continue
        role = "caller" if entry.get("role") == "caller" else "bot"
        lines.append(f"    {role + ':':<8}{_clip(entry.get('text'))}")
    return lines


def call_title(outcome: Any) -> str:
    name = outcome.scenario_name or outcome.scenario_id or "call"
    return f"call {outcome.call_index}: {name} (repeat {outcome.repeat})"


def call_lines(outcome: Any, timing_only: bool = False) -> List[str]:
    """The block printed when a call finishes."""
    card = outcome.scorecard or {}
    call = card.get("call") or {}
    turns = card.get("turns") or []
    state = "ok" if outcome.error is None else f"FAILED, {outcome.error['kind']}: {outcome.error['message']}"
    head = call_title(outcome)
    if outcome.duration_ms is not None:
        head += f", {_seconds(outcome.duration_ms)}"
    lines = [f"{head}: {state}"]
    if not card:
        lines.extend(f"  ! {p}" for p in outcome.problems)
        return lines
    delays = [t.get("response_delay_ms") for t in turns]
    answered = sum(1 for d in delays if d is not None)
    lines.append(
        f"  response delay   p50 {ms(call.get('response_delay_p50_ms'))}, p95 "
        f"{ms(call.get('response_delay_p95_ms'))}; {answered} of {len(turns)} turns answered"
        + ("" if not turns else "; per turn " + " / ".join(
            ms(d) if d is not None else "none" for d in delays))
    )
    if call.get("first_word_ms") is not None and call.get("bot_spoke_first"):
        lines.append(f"  bot spoke first  at {ms(call.get('first_word_ms'))}")
    count = call.get("talk_over_count") or 0
    lines.append("  talk-over        none" if not count else
                 f"  talk-over        {count} time{'s' if count != 1 else ''}, {ms(call.get('talk_over_ms'))}")
    planned = call.get("interruption_count") or 0
    not_reached = call.get("interruption_not_reached_count") or 0
    if planned:
        text = f"  interruptions    {planned} planned"
        if not_reached < planned:
            text += (f"; the bot stopped in p50 {ms(call.get('interruption_stop_p50_ms'))}, "
                     f"p95 {ms(call.get('interruption_stop_p95_ms'))}")
        if not_reached:
            text += f"; {not_reached} not reached: the bot finished before the planned interruption point"
        lines.append(text)
    else:
        lines.append("  interruptions    none planned")
    lines.append(f"  longest silence  {ms(call.get('max_silence_ms'))}")
    lines.extend(conversation_lines(call.get("transcript"), getattr(outcome, "turns_taken", None)))
    lines.append("  caller           " + _health(card.get("caller_health") or {}, call))
    if outcome.recording:
        lines.append(f"  recording        uploaded ({_bytes(outcome.recording.get('bytes'))})")
    elif timing_only:
        lines.append("  recording        not uploaded (timing only)")
    lines.extend(f"  ! {p}" for p in outcome.problems)
    return lines


def _health(health: Mapping[str, Any], call: Mapping[str, Any]) -> str:
    late = health.get("late_frames") or 0
    dropped = health.get("dropped_frames") or 0
    unscored = list(call.get("unscored_turns") or health.get("unhealthy_turns") or [])
    think = health.get("think_ms_p50")
    thought = (f"; deciding p50 {ms(think)}, max {ms(health.get('think_ms_max'))}"
               if isinstance(think, (int, float)) and not isinstance(think, bool) else "")
    if health.get("healthy", True) and not unscored:
        detail = f"{late} late frame{'s' if late != 1 else ''}" if late else "no late frames"
        return f"healthy ({detail}{thought})"
    parts = [f"{late} late frames (max {ms(health.get('late_max_ms'))})"]
    if dropped:
        parts.append(f"{dropped} dropped")
    if unscored:
        parts.append("turns " + ", ".join(str(t) for t in unscored) + " left out of scoring")
    return "UNHEALTHY: " + "; ".join(parts)


def _count(value: Any) -> Any:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def figures_text(figures: Any) -> str:
    """An evaluator's `figures` as the evaluators display them: the measured figure, the limit, and how
    many turns went over it. Empty when there is nothing to show (an evaluator that was not scored)."""
    if not isinstance(figures, Mapping):
        return ""
    parts: List[str] = []
    if "slowestMs" in figures:  # interruption stops: the slowest is what the evaluator states
        parts.append(f"slowest stop {ms(figures['slowestMs'])}")
    elif "p95Ms" in figures:
        parts.append(f"p95 {ms(figures['p95Ms'])}")
    elif "talkOverMs" in figures:
        count = _count(figures.get("talkOverCount")) or 0
        parts.append("no talk-over" if not count
                     else f"{_plural(count, 'talk-over')}, {ms(figures['talkOverMs'])} in all")
    elif "longestSilenceMs" in figures:
        longest = figures.get("longestSilenceMs")
        parts.append(f"longest silence {ms(longest)}" if longest else "no silence over 3,000 ms")
    if "limitMs" in figures:
        parts.append(f"limit {ms(figures['limitMs'])}")
    for over, total, word in (("turnsOver", "turns", "turn"), ("interruptionsOver", "interruptions", "interruption")):
        if _count(figures.get(over)) is not None and _count(figures.get(total)) is not None:
            parts.append(f"{figures[over]} of {_plural(figures[total], word)} over")
            break
    text = ", ".join(parts)
    unanswered = _count(figures.get("unanswered"))
    if unanswered:
        text += f" ({unanswered} unanswered)"
    not_reached = _count(figures.get("notReached"))
    if text and not_reached:
        text += f" ({not_reached} not reached)"
    unscored = figures.get("unscoredTurns")
    if text and isinstance(unscored, list) and unscored:
        text += "; turns " + ", ".join(str(t) for t in unscored) + " left out (caller unhealthy)"
    return text


def _evaluator_rows(items: Iterable[Any]) -> List[str]:
    rows = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        name = (item.get("name") or item.get("key") or item.get("evaluator") or item.get("metricName")
                or item.get("ruleId") or "?")
        score = item.get("score")
        passed = item.get("passed")
        shown = (item.get("display") or item.get("summary") or item.get("message") or item.get("reason")
                 or figures_text(item.get("figures")))
        cells = [str(name)]
        has_score = isinstance(score, (int, float)) and not isinstance(score, bool)
        if has_score:
            cells.append(f"{score:.2f}")
        if isinstance(passed, bool):
            cells.append("passed" if passed else "failed")
        elif not has_score:
            cells.append("not scored")
        if shown:
            cells.append(str(shown))
        rows.append("  " + "  ".join(cells))
    return rows


def _evaluators(summary: Mapping[str, Any]) -> List[str]:
    for key in ("evaluators", "evaluatorResults", "scores"):
        if isinstance(summary.get(key), list):
            return _evaluator_rows(summary[key])
    rows: List[str] = []
    for call in summary.get("calls") or []:
        if not isinstance(call, Mapping):
            continue
        for key in ("evaluators", "evaluatorResults", "scores"):
            if isinstance(call.get(key), list) and call[key]:
                index = call.get("callIndex")
                rows.append(f"  call {index}:" if index is not None else "  call:")
                rows.extend("  " + row for row in _evaluator_rows(call[key]))
                break
    return rows


def _task_success(summary: Mapping[str, Any]) -> List[str]:
    """Voice Task Success per call: did the bot get each caller what they came for (RUN_API section 6)."""
    rows: List[str] = []
    for call in summary.get("calls") or []:
        if not isinstance(call, Mapping):
            continue
        for item in call.get("evaluators") or []:
            if not isinstance(item, Mapping) or item.get("key") != TASK_SUCCESS_KEY:
                continue
            cells = [f"  call {call.get('callIndex')}:"]
            score = item.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                cells.append(f"{score:.2f}")
            passed = item.get("passed")
            cells.append(("passed" if passed else "failed") if isinstance(passed, bool) else
                         str(item.get("status") or "not scored"))
            if item.get("reason"):
                cells.append(_clip(item["reason"], 160))
            rows.append(" ".join(cells))
    return rows


def run_lines(report: Any) -> List[str]:
    """The block printed once the run is over."""
    calls = report.calls
    failed = [c for c in calls if c.error is not None]
    lines = [f"run {report.run_id}: {len(calls)} call{'s' if len(calls) != 1 else ''}, "
             f"{len(calls) - len(failed)} ok, {len(failed)} failed"]
    described = caller_text(getattr(report, "caller", None), getattr(report, "platform_minutes_left", None))
    if described:
        lines.append(described)
    charged = (report.summary or {}).get("platformMinutes")
    if isinstance(charged, (int, float)) and not isinstance(charged, bool) and charged > 0:
        lines.append(f"this run used {charged:,.1f} SyntropyLabs model minutes")
    if report.summary:
        rows = _evaluators(report.summary)
        if rows:
            lines.append("evaluators:")
            lines.extend(rows)
        success = _task_success(report.summary)
        if success:
            lines.append("task success:")
            lines.extend(success)
    lines.extend(f"! {p}" for p in report.problems)
    if report.out_dir:
        lines.append(f"recordings: {report.out_dir}")
    if report.console_url:
        lines.append(f"console: {report.console_url}")
    lines.append(f"verdict: {report.verdict} (exit {report.exit_code})")
    return lines


def check_lines(result: Mapping[str, Any]) -> List[str]:
    lines = [f"connected:   {'yes' if result.get('connected') else 'no'}"
             + (f" ({result['connectError']})" if result.get("connectError") else "")]
    if result.get("connected"):
        lines.append(f"bot audio:   {result.get('botChunks', 0)} chunks, {_bytes(result.get('botBytes'))}"
                     + (", speech heard" if result.get("botSpeech") else ", no speech heard"))
        if result.get("botSpokeFirst"):
            lines.append(f"greeting:    the bot spoke first, at {ms(result.get('firstWordMs'))}")
        answered = result.get("answered")
        lines.append("answered:    " + (f"yes, after {ms(result.get('responseDelayMs'))}" if answered else "no"))
        errors = result.get("formatErrors") or []
        lines.append("format:      " + ("no errors" if not errors else f"{len(errors)} kind(s) of error"))
        for item in errors:
            lines.append(f"  {item['count']} x {item['detail']}")
        for message in result.get("errors") or []:
            lines.append(f"error:       {message}")
        if result.get("closedByBot"):
            lines.append(f"closed:      by the bot (code {result.get('closeCode')})")
    for hint in result.get("hints") or []:
        lines.append(f"hint:        {hint}")
    lines.append(f"result:      {'ok' if result.get('ok') else 'problems found'}")
    return lines
