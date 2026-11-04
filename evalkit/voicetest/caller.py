"""Runs one scripted call against a voice bot over WebSocket (CONTRACT sections 2 to 4).

The caller sends a 20 ms frame every 20 ms from a monotonic clock, silence included, for the whole
call. Received bot audio goes through the playback buffer and is rendered one frame behind real
time onto the heard timeline, which is what the recording's bot channel holds. Every event is
stamped with its position on the sample clock.

Live decisions (when to say the next line) use a live speech tracker on the heard audio. They only
decide when the caller speaks; the measurements are made after the call from the recording.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional

from .audio import FRAME_MS, SpeechTracker, frame_samples, samples_to_ms
from .recording import (
    CALL_NAME,
    CHANNELS,
    EVENTS_NAME,
    RECORDING_NAME,
    CallRecorder,
    EventLog,
    write_call_json,
)
from .script import Script
from .sources import LineSource, render_lines
from .transport import (
    BotAudio,
    BotClear,
    BotMark,
    BotOther,
    PlaybackBuffer,
    Target,
    TransportClosed,
    make_transport,
)

log = logging.getLogger("evalkit.voicetest.caller")

CALLER_VERSION = "0.1.0"
# The detector the analysis applies to this call; recorded in call.json so a later re-analysis matches.
VAD_VERSION = "webrtc-v2"
CONTRACT_VERSION = "v1"
PACING_LATE_MS = 5.0
# Sleep to within this much of each deadline, then yield to the event loop until it passes. asyncio
# rounds sleeps up to whole ms on Linux; the short spin keeps frames on time while still serving
# the socket, so arrival times stay sharp.
SPIN_MS = 1.0
# Bot speech has to last this long inside a wait to count as the bot taking its turn.
MIN_BOT_TURN_MS = 100
# The bot counts as still speaking if it was voiced this recently (two frames).
SPEAKING_NOW_MS = 40


class CallError(RuntimeError):
    pass


def _is_connection_closed(exc: BaseException) -> bool:
    from websockets.exceptions import ConnectionClosed

    return isinstance(exc, ConnectionClosed)


@dataclass
class PacingStats:
    late_ms: List[float] = field(default_factory=list)

    def add(self, late_ms: float) -> None:
        self.late_ms.append(late_ms)

    def summary(self) -> Dict[str, Any]:
        values = sorted(self.late_ms)
        n = len(values)

        def pct(p: float) -> float:
            if not n:
                return 0.0
            return round(values[min(n - 1, int(p * (n - 1) + 0.5))], 3)

        return {
            "frames": n,
            "late_over_5ms": sum(1 for v in values if v > PACING_LATE_MS),
            "max_late_ms": round(values[-1], 3) if n else 0.0,
            "mean_late_ms": round(sum(values) / n, 3) if n else 0.0,
            "p50_late_ms": pct(0.50),
            "p95_late_ms": pct(0.95),
            "p99_late_ms": pct(0.99),
        }


@dataclass
class CallResult:
    out_dir: str
    session: str
    duration_ms: int
    frames_sent: int
    ended_by: str  # "script" | "max_call" | "bot_closed" | "aborted" | "error"
    pacing: Dict[str, Any]
    errors: List[str]
    events: List[Dict[str, Any]]

    @property
    def recording_path(self) -> str:
        return os.path.join(self.out_dir, RECORDING_NAME)

    @property
    def events_path(self) -> str:
        return os.path.join(self.out_dir, EVENTS_NAME)

    @property
    def call_path(self) -> str:
        return os.path.join(self.out_dir, CALL_NAME)


def new_session_token() -> str:
    return "local:" + uuid.uuid4().hex


class _ScriptRunner:
    """Decides, frame by frame, what the caller sends next.

    `next_frame(now)` is called after the heard audio up to `now` has been observed and returns the
    frame that starts at `now + frame`; None means the call ends before it.
    """

    def __init__(self, script: Script, audio: List, rate: int, events: EventLog, tracker) -> None:
        import numpy as np

        self._np = np
        self.script = script
        self.audio = audio
        self.rate = rate
        self.frame = frame_samples(rate)
        self.events = events
        self.tracker = tracker
        self.index = 0
        self.speaking = False
        self.line_pos = 0
        self.line_start = 0
        self.wait_start = 0
        # Whether the bot has started speaking in the current wait; kept once true, so a later short
        # sound cannot make a wait with start_within_ms give up on a bot already talking.
        self.bot_started = False
        self.ended_by: Optional[str] = None
        self._silence = np.zeros(self.frame, dtype=np.int16)
        self._begin_wait(0, 0)

    def _rule(self, index: int):
        if index >= len(self.script.lines):
            return self.script.end, None
        line = self.script.lines[index]
        return line.wait, line.interrupt_after_bot_ms

    def _begin_wait(self, index: int, at: int) -> None:
        rule, interrupt = self._rule(index)
        self.wait_start = at
        self.bot_started = False
        fields: Dict[str, Any] = {"index": index}
        if interrupt is not None:
            fields.update(reason="interrupt", after_bot_ms=interrupt, max_ms=rule.max_ms)
        elif rule.mode == "fixed":
            fields.update(reason="fixed", ms=rule.ms)
        else:
            fields.update(reason="bot_done", silence_ms=rule.silence_ms, max_ms=rule.max_ms)
            if rule.start_within_ms is not None:
                fields["start_within_ms"] = rule.start_within_ms
        if index >= len(self.script.lines):
            fields["end"] = True
        self.events.log("wait_start", sample=at, **fields)
        if interrupt is not None:
            self.events.log("interrupt_planned", sample=at, index=index, after_bot_ms=interrupt)

    def _ms(self, samples: int) -> float:
        return samples * 1000.0 / self.rate

    def _wait_done(self, now: int, start: int) -> Optional[str]:
        rule, interrupt = self._rule(self.index)
        tr = self.tracker
        elapsed = self._ms(start - self.wait_start)
        if interrupt is not None:
            if (
                tr.onset_sample is not None
                and tr.last_voiced_end is not None
                and self._ms(now - tr.last_voiced_end) <= SPEAKING_NOW_MS
                and self._ms(tr.last_voiced_end - tr.onset_sample) >= interrupt
            ):
                return "interrupt"
        elif rule.mode == "fixed":
            return "fixed" if elapsed >= rule.ms else None
        else:
            spoke = (
                tr.last_voiced_end is not None
                and tr.onset_sample is not None
                and self._ms(tr.last_voiced_end - max(self.wait_start, tr.onset_sample))
                >= MIN_BOT_TURN_MS
            )
            self.bot_started = self.bot_started or spoke
            if (
                rule.start_within_ms is not None
                and not self.bot_started
                and elapsed >= rule.start_within_ms
                and not self._bot_voicing(now)
            ):
                return "no_greeting"
            if spoke and self._ms(now - tr.last_voiced_end) >= rule.silence_ms:
                return "bot_done"
        if elapsed >= rule.max_ms:
            return "max_wait"
        return None

    def _bot_voicing(self, now: int) -> bool:
        """Bot audio that may yet count as speech: voiced frames short of an onset, or speech still
        sounding. Speech that began before `start_within_ms` is told apart from none only once the
        tracker has seen enough of it, so the wait holds on until it has."""
        tr = self.tracker
        if tr.onset_pending:
            return True
        return (
            tr.speaking
            and tr.last_voiced_end is not None
            and tr.last_voiced_end > self.wait_start
            and self._ms(now - tr.last_voiced_end) <= SPEAKING_NOW_MS
        )

    def next_frame(self, now: int):
        start = now + self.frame
        if not self.speaking:
            if samples_to_ms(start, self.rate) >= self.script.max_call_ms:
                self.events.log("wait_end", sample=start, index=self.index, reason="max_call")
                self.ended_by = "max_call"
                return None
            reason = self._wait_done(now, start)
            if reason is None:
                return self._silence
            self.events.log("wait_end", sample=start, index=self.index, reason=reason)
            if self.index >= len(self.script.lines):
                self.ended_by = "script"
                return None
            line = self.script.lines[self.index]
            self.events.log("think", sample=start, index=self.index, ms=0)
            self.events.log("line_start", sample=start, index=self.index, text=line.text)
            self.speaking = True
            self.line_pos = 0
            self.line_start = start
        elif samples_to_ms(start, self.rate) >= self.script.max_call_ms:
            self.ended_by = "max_call"
            return None
        pcm = self.audio[self.index]
        chunk = pcm[self.line_pos : self.line_pos + self.frame]
        self.line_pos += self.frame
        if self.line_pos >= len(pcm):
            end = self.line_start + len(pcm)
            line = self.script.lines[self.index]
            self.events.log("line_end", sample=end, index=self.index, text=line.text)
            self.speaking = False
            self.index += 1
            self._begin_wait(self.index, end)
        if len(chunk) < self.frame:
            chunk = self._np.concatenate([chunk, self._silence[: self.frame - len(chunk)]])
        return chunk


async def run_call(
    target: Target,
    script: Script,
    out_dir: str,
    session: Optional[str] = None,
    sources: Optional[Mapping[str, LineSource]] = None,
    tick_hook: Optional[Callable[[int], Optional[Awaitable[None]]]] = None,
    spin_ms: float = SPIN_MS,
    stop: Optional[asyncio.Event] = None,
) -> CallResult:
    """Run one call and write recording.wav, events.jsonl and call.json into `out_dir`.

    `tick_hook(frame_index)` runs before each frame is paced; tests use it to stall the loop.
    Setting `stop` ends the call at the next frame boundary the way the script's end would (stop
    message, close, every file written), with `ended_by: "aborted"`, so an interrupted run still has
    a recording it can analyse.
    Raises CallError if the WebSocket cannot be opened (call.json then records the error).
    """
    session = session or new_session_token()
    rate = target.sample_rate
    frame = frame_samples(rate)
    audio = render_lines(script, rate, sources)  # before connecting: no synthesis mid-call
    os.makedirs(out_dir, exist_ok=True)

    call: Dict[str, Any] = {
        "contract": CONTRACT_VERSION,
        "caller_version": CALLER_VERSION,
        "vad": {"version": VAD_VERSION},
        "target": target.describe(),
        "format": target.format,
        "sample_rate": rate,
        "session": session,
        "script": script.raw,
    }
    transport = make_transport(target, session)
    try:
        await transport.connect()
    except Exception as exc:
        message = f"connect failed: {type(exc).__name__}: {exc}"
        events = EventLog(os.path.join(out_dir, EVENTS_NAME), rate)
        events.log("error", t=0, message=message, stage="connect")
        events.close()
        call.update(duration_ms=0, ended_by="error", errors=[message])
        write_call_json(out_dir, call)
        raise CallError(message) from exc

    t0 = time.monotonic()
    t0_epoch = time.time()
    events = EventLog(os.path.join(out_dir, EVENTS_NAME), rate)
    recorder = CallRecorder(os.path.join(out_dir, RECORDING_NAME), rate)
    playback = PlaybackBuffer(rate)
    tracker = SpeechTracker(rate)
    pacing = PacingStats()
    errors: List[str] = []
    state = {"bot_closed": False, "stopped": False, "closed_logged": False}

    def position(mono: float) -> int:
        return max(0, int((mono - t0) * rate))

    def error(message: str, sample: int, **fields) -> None:
        errors.append(message)
        events.log("error", sample=sample, message=message, **fields)

    async def echo(reached) -> None:
        for mark in reached:
            if state["stopped"] or state["bot_closed"]:
                return
            try:
                await transport.send_mark(mark.name)
            except Exception as exc:
                error(f"mark echo failed: {exc}", mark.position)
                return
            events.log("mark_echoed", sample=mark.position, name=mark.name, cleared=mark.cleared)

    async def receive() -> None:
        while True:
            ev = await transport.recv()
            mono = time.monotonic()
            at = position(mono)
            if isinstance(ev, BotAudio):
                s = playback.push(ev.pcm, at)
                events.log(
                    "bot_chunk",
                    sample=at,
                    bytes=ev.nbytes,
                    arrival_wall=round(mono * 1000.0, 3),
                    queued_ms=samples_to_ms(s.queued, rate),
                    heard_ms=samples_to_ms(s.start, rate),
                    samples=s.end - s.start,
                )
            elif isinstance(ev, BotMark):
                reach = playback.push_mark(ev.name, at)
                events.log("mark_sent", sample=at, name=ev.name, heard_ms=samples_to_ms(reach, rate))
            elif isinstance(ev, BotClear):
                dropped, released = playback.clear(at)
                events.log("clear", sample=at, dropped_ms=samples_to_ms(dropped, rate))
                await echo(released)
            elif isinstance(ev, BotOther):
                events.log("error", sample=at, message="unknown frame ignored", detail=ev.detail,
                           kind="unknown_frame")
            elif isinstance(ev, TransportClosed):
                if not state["stopped"]:
                    state["bot_closed"] = True
                    state["closed_logged"] = True
                    events.log("closed", sample=at, by="bot", code=ev.code, reason=ev.reason)
                return

    events.log("connect", t=0, format=target.format, sample_rate=rate)
    ended_by = "script"
    aborted = False
    k = 0
    rx: Optional[asyncio.Task] = None
    try:
        start_msg = await transport.send_start()
        if start_msg is not None:
            events.log("start_sent", t=0)
        rx = asyncio.ensure_future(receive())
        runner = _ScriptRunner(script, audio, rate, events, tracker)
        next_frame = runner.next_frame(-frame)
        spin = spin_ms / 1000.0
        while next_frame is not None:
            if stop is not None and stop.is_set():
                aborted = True
                break
            if tick_hook is not None:
                maybe = tick_hook(k)
                if maybe is not None:
                    await maybe
            deadline = t0 + k * FRAME_MS / 1000.0
            remaining = deadline - time.monotonic()
            if remaining > spin:
                await asyncio.sleep(remaining - spin)
            while time.monotonic() < deadline:
                await asyncio.sleep(0)
            if rx.done():
                break
            late_ms = (time.monotonic() - deadline) * 1000.0
            try:
                await transport.send_frame(next_frame, k * FRAME_MS)
            except Exception as exc:
                if not _is_connection_closed(exc):
                    raise
                # The bot hung up between frames; the receive task records the close.
                await asyncio.wait([rx], timeout=1.0)
                if not state["closed_logged"]:
                    state["bot_closed"] = state["closed_logged"] = True
                    events.log("closed", sample=k * frame, by="bot")
                break
            pacing.add(late_ms)
            if late_ms > PACING_LATE_MS:
                events.log("pacing", sample=k * frame, late_ms=round(late_ms, 1), frame=k)
            recorder.add_caller(next_frame)
            heard, reached = playback.render(k * frame)
            recorder.add_bot(heard)
            tracker.feed(heard)
            await echo(reached)
            k += 1
            next_frame = runner.next_frame((k - 1) * frame)
        if state["bot_closed"]:
            ended_by = "bot_closed"
        elif aborted:
            ended_by = "aborted"
        else:
            ended_by = runner.ended_by or "script"
    except Exception as exc:  # the call is over; keep whatever was recorded
        ended_by = "error"
        error(f"{type(exc).__name__}: {exc}", k * frame)
        log.warning("call failed: %s", exc)
    finally:
        total = k * frame
        end_deadline = t0 + k * FRAME_MS / 1000.0
        delay = end_deadline - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        heard, _ = playback.render(total)
        recorder.add_bot(heard)
        if rx is not None and rx.done() and not rx.cancelled() and rx.exception() is not None:
            exc = rx.exception()
            ended_by = "error"
            error(f"receive failed: {type(exc).__name__}: {exc}", total)
        if not state["bot_closed"]:
            try:
                if await transport.send_stop() is not None:
                    events.log("stop_sent", sample=total)
            except Exception as exc:
                error(f"stop failed: {exc}", total)
        state["stopped"] = True
        await transport.close()
        if rx is not None and not rx.done():
            rx.cancel()
            try:
                await rx
            except (asyncio.CancelledError, Exception):
                pass
        if not state["closed_logged"]:
            events.log("closed", sample=total, by="caller")
        recorder.finish(total)
        events.close()

    duration_ms = samples_to_ms(total, rate)
    call.update(
        t0_epoch_ms=round(t0_epoch * 1000.0, 3),
        t0_iso=datetime.datetime.fromtimestamp(t0_epoch, datetime.timezone.utc).isoformat(),
        t0_monotonic_ms=round(t0 * 1000.0, 3),
        duration_ms=duration_ms,
        frames_sent=k,
        ended_by=ended_by,
        pacing=pacing.summary(),
        recording={
            "file": RECORDING_NAME,
            "sample_rate": 16000,
            "channels": list(CHANNELS),
        },
        errors=errors,
    )
    write_call_json(out_dir, call)
    log.info("call %s ended (%s) after %d ms", session, ended_by, duration_ms)
    return CallResult(
        out_dir=str(out_dir),
        session=session,
        duration_ms=duration_ms,
        frames_sent=k,
        ended_by=ended_by,
        pacing=pacing.summary(),
        errors=errors,
        events=events.events,
    )

