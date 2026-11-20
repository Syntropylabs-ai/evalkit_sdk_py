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
import collections
import datetime
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

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
from .script import DynamicTurns, Script, WaitRule
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
# What `_wait_done` returns when an interrupting line's bot finished short of the interruption point;
# logged as `interrupt_not_reached`, then the wait ends as `bot_done`.
_NOT_REACHED = "interrupt_not_reached"


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
    # "script" | "max_call" | "bot_closed" | "aborted" | "error", and "turn_failed" for a dynamic call
    # whose next line could not be had (`turn_failure` says why)
    ended_by: str
    pacing: Dict[str, Any]
    errors: List[str]
    events: List[Dict[str, Any]]
    mode: str = "scripted"
    turn_failure: Optional[str] = None
    # What went wrong without failing the call: the bot's last answer that could not be transcribed.
    warnings: List[str] = field(default_factory=list)

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
            # A bot whose answer ended short of the interruption point will not reach it. Waiting on to
            # max_ms would leave dead air and still interrupt nothing, so the line is said as a normal
            # turn, by the wait's own end-of-speech rule. The bot must be quiet beyond the speaking
            # window too, so a silence_ms of 0 cannot cut in on a bot that is still talking.
            quiet = None if tr.last_voiced_end is None else self._ms(now - tr.last_voiced_end)
            if self._bot_spoke() and quiet >= rule.silence_ms and quiet > SPEAKING_NOW_MS:
                return _NOT_REACHED
        elif rule.mode == "fixed":
            return "fixed" if elapsed >= rule.ms else None
        else:
            spoke = self._bot_spoke()
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

    def _bot_spoke(self) -> bool:
        """The bot has taken its turn in the current wait: at least MIN_BOT_TURN_MS of its speech fell
        after the wait began."""
        tr = self.tracker
        return (
            tr.last_voiced_end is not None
            and tr.onset_sample is not None
            and self._ms(tr.last_voiced_end - max(self.wait_start, tr.onset_sample)) >= MIN_BOT_TURN_MS
        )

    def _bot_spoke_ms(self) -> int:
        """How far into its speech the bot got: the stretch the interruption point is measured on."""
        tr = self.tracker
        if tr.onset_sample is None or tr.last_voiced_end is None:
            return 0
        return int(round(self._ms(tr.last_voiced_end - tr.onset_sample)))

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
            if reason == _NOT_REACHED:
                self.events.log(_NOT_REACHED, sample=start, index=self.index, bot_spoke_ms=self._bot_spoke_ms())
                reason = "bot_done"
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


class _BotTail:
    """The bot channel as heard, at the call's rate, kept for its last minute or so: what a dynamic call
    sends as the bot's answer. Holds the frames the playback buffer renders, which nothing else changes."""

    def __init__(self, rate: int, keep_ms: int) -> None:
        self.rate = rate
        self.keep = rate * (keep_ms + 2000) // 1000
        self.cap = rate * keep_ms // 1000
        self.position = 0
        self._held = 0
        self._chunks: collections.deque = collections.deque()

    def add(self, pcm) -> None:
        n = len(pcm)
        if not n:
            return
        self._chunks.append((self.position, pcm))
        self.position += n
        self._held += n
        while self._chunks and self._held - len(self._chunks[0][1]) >= self.keep:
            self._held -= len(self._chunks.popleft()[1])

    def first_sample(self, start: int) -> int:
        """The sample `since(start)` begins at: `start`, or later when that is more than `keep_ms` ago."""
        return max(int(start), self.position - self.cap)

    def since(self, start: int) -> List[Any]:
        """The heard audio from sample `start` on, at most its last `keep_ms`, as views of the frames
        (joined off the loop, by the request)."""
        start = self.first_sample(start)
        out = []
        for at, pcm in self._chunks:
            if at + len(pcm) > start:
                out.append(pcm[max(0, start - at):])
        return out


@dataclass
class _Line:
    text: str
    pcm: Any
    wait: WaitRule
    interrupt_after_bot_ms: Optional[int] = None


class _DynamicRunner(_ScriptRunner):
    """Plays a dynamic script (CONTRACT section 4.1): the lines planned before the call, the opening
    first, then after each bot answer the line the turn endpoint returns for it.

    It is driven frame by frame like `_ScriptRunner` and shares its wait rules and interruption logic.
    A turn request runs on a worker thread; until its answer is in, the caller sends silence, so the
    frame loop's pacing is never held up. The caller decides nothing itself: every line comes from the
    endpoint, and the only choices made here are when to say it.
    """

    def __init__(self, script: Script, audio: List, rate: int, events: EventLog, tracker, client, bot: _BotTail,
                 loop: asyncio.AbstractEventLoop) -> None:
        from .turns import MAX_RETAKES, RETAKE_BOT_SPOKE_MS

        self.client = client
        self.bot = bot
        self.loop = loop
        self.retake_ms = RETAKE_BOT_SPOKE_MS
        self.max_retakes = MAX_RETAKES
        self.turns: DynamicTurns = script.turns or DynamicTurns()
        self.lines: List[_Line] = [_Line(line.text, pcm, line.wait, line.interrupt_after_bot_ms)
                                   for line, pcm in zip(script.lines, audio)]
        # The caller's last line, once it is known: the turn endpoint said done, or turns.max is reached.
        self.last_index: Optional[int] = len(self.lines) - 1 if len(self.lines) - 1 >= self.turns.max else None
        self.phase = "wait"  # wait | ask | rewait (speaking is `self.speaking`)
        # Where the caller's previous line ended: the bot's answer is sent from there.
        self.prev_end = 0
        # Where each line said so far started and ended, by index: the first request after a cut-in
        # sends the bot's audio from the end of the line before the cut-in.
        self._line_starts: Dict[int, int] = {}
        self._line_ends: Dict[int, int] = {}
        self.turn_failure: Optional[str] = None
        self._pending: Optional[asyncio.Future] = None
        self._retry_at: Optional[int] = None
        self._attempt = 0
        self._asked: Optional[Dict[str, Any]] = None
        self._answer = None
        self._retakes: Dict[int, int] = {}
        self._rewait_start = 0
        self._segments: List[List[int]] = []  # bot speech seen by the tracker: [onset, last voiced end]
        super().__init__(script, audio, rate, events, tracker)

    # -- rules ---------------------------------------------------------------------------------

    def _is_end(self, index: int) -> bool:
        return self.last_index is not None and index > self.last_index

    def _rule(self, index: int):
        if self._is_end(index):
            return self.script.end, None
        if index < len(self.lines):
            line = self.lines[index]
            return line.wait, line.interrupt_after_bot_ms
        return self.turns.wait, None

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
        if self._is_end(index):
            fields["end"] = True
        self.events.log("wait_start", sample=at, **fields)
        if interrupt is not None:
            self.events.log("interrupt_planned", sample=at, index=index, after_bot_ms=interrupt)

    def owed_final(self) -> Optional[int]:
        """The turn of the heard-only request still owed once the caller's last line is said: the bot's
        answer to it is sent so the transcript ends with what the bot last said."""
        return self.index if self._is_end(self.index) and not self.speaking else None

    # -- the bot's speech ----------------------------------------------------------------------

    def _track(self) -> None:
        tr = self.tracker
        if tr.onset_sample is None or tr.last_voiced_end is None:
            return
        if self._segments and self._segments[-1][0] == tr.onset_sample:
            self._segments[-1][1] = tr.last_voiced_end
        else:
            self._segments.append([tr.onset_sample, tr.last_voiced_end])
            if len(self._segments) > 64:
                del self._segments[:-64]

    def _spoke_since(self, sample: int) -> int:
        """ms of bot speech after `sample`."""
        total = sum(max(0, end - max(start, sample)) for start, end in self._segments)
        return int(round(self._ms(total)))

    def _bot_quiet(self, now: int) -> bool:
        """The bot is not speaking: quiet for the turn wait's silence_ms, with no speech starting."""
        tr = self.tracker
        if tr.onset_pending:
            return False
        if tr.last_voiced_end is None:
            return True
        quiet = self._ms(now - tr.last_voiced_end)
        return quiet >= self.turns.wait.silence_ms and quiet > SPEAKING_NOW_MS

    # -- the frame -----------------------------------------------------------------------------

    def next_frame(self, now: int):
        start = now + self.frame
        self._track()
        over = samples_to_ms(start, self.rate) >= self.script.max_call_ms
        if self.speaking:
            if over:
                self.ended_by = "max_call"
                return None
            return self._speech()
        if over:
            return self._out_of_time(start)
        if self.phase == "wait":
            reason = self._wait_done(now, start)
            if reason is None:
                return self._silence
            if reason == _NOT_REACHED:
                self.events.log(_NOT_REACHED, sample=start, index=self.index, bot_spoke_ms=self._bot_spoke_ms())
                reason = "bot_done"
            self.events.log("wait_end", sample=start, index=self.index, reason=reason)
            return self._after_wait(start)
        if self.phase == "rewait":
            reason = None
            if self._bot_quiet(now):
                reason = "bot_done"
            elif self._ms(start - self._rewait_start) >= self.turns.wait.max_ms:
                reason = "max_wait"
            if reason is None:
                return self._silence
            self.events.log("wait_end", sample=start, index=self.index, reason=reason)
            return self._decide(now, start, waited=True)
        return self._poll(now, start)

    def _speech(self):
        line = self.lines[self.index]
        pcm = line.pcm
        chunk = pcm[self.line_pos : self.line_pos + self.frame]
        self.line_pos += self.frame
        if self.line_pos >= len(pcm):
            end = self.line_start + len(pcm)
            self.events.log("line_end", sample=end, index=self.index, text=line.text)
            self.speaking = False
            self.prev_end = end
            self._line_ends[self.index] = end
            self.index += 1
            self.phase = "wait"
            self._begin_wait(self.index, end)
        if len(chunk) < self.frame:
            chunk = self._np.concatenate([chunk, self._silence[: self.frame - len(chunk)]])
        return chunk

    def _start_line(self, start: int, think_ms: int):
        line = self.lines[self.index]
        self.events.log("think", sample=start, index=self.index, ms=think_ms)
        self.events.log("line_start", sample=start, index=self.index, text=line.text)
        self.speaking = True
        self.line_pos = 0
        self.line_start = start
        self._line_starts[self.index] = start
        return self._speech()

    def _bot_audio(self, turn: int) -> Tuple[List[Any], Optional[int]]:
        """The bot audio a request for `turn` sends, and where in it the caller began a cut-in, in ms
        (RUN_API section 4.1, `botAudio.cutInAtMs`). When the line before `turn` cut into the bot's
        answer, the audio starts at the end of the line before that one, so the answer the cut-in
        interrupted is heard as well as the reply to the cut-in; else it starts where the caller's
        previous line ended and there is no cut-in point. A cut-in the bot's answer ended before is
        sent the same way: that answer was not heard by any request either."""
        cut = turn - 1
        line = self.lines[cut] if 1 <= cut < len(self.lines) else None
        if (line is not None and line.interrupt_after_bot_ms is not None
                and cut in self._line_starts and cut - 1 in self._line_ends):
            begin, cut_at = self._line_ends[cut - 1], self._line_starts[cut]
            first = self.bot.first_sample(begin)
            if cut_at > first:  # else the interrupted answer is older than the most a request carries
                cut_ms = int(round(self._ms(cut_at - first)))
                if cut_ms >= 1:
                    return self.bot.since(begin), cut_ms
        return self.bot.since(self.prev_end), None

    def _after_wait(self, start: int):
        if self._is_end(self.index):
            self.ended_by = "script"
            return None
        if self.index < len(self.lines):  # planned before the call, or a cut-in written with the line before
            return self._start_line(start, 0)
        self._ask(start, retake=0)
        return self._silence

    def _out_of_time(self, start: int):
        if self.phase == "ask":
            message = (f"turn {self.index}: no answer from the turn endpoint within the call's budget "
                       f"({self.script.max_call_ms} ms)")
            self.events.log("turn_failed", sample=start, index=self.index, status=None, message=message)
            self.turn_failure = message
            self.ended_by = "turn_failed"
            return None
        self.events.log("wait_end", sample=start, index=self.index, reason="max_call")
        self.ended_by = "max_call"
        return None

    # -- the turn request ----------------------------------------------------------------------

    def _ask(self, start: int, retake: int) -> None:
        chunks, cut_in_at_ms = self._bot_audio(self.index)
        self._asked = {"index": self.index, "retake": retake, "at": start, "chunks": chunks,
                       "cut_in_at_ms": cut_in_at_ms}
        self._attempt = 0
        self._submit(start)

    def _submit(self, start: int) -> None:
        asked = self._asked
        audio_ms = int(round(self._ms(sum(len(c) for c in asked["chunks"]))))
        fields: Dict[str, Any] = {}
        if asked["cut_in_at_ms"] is not None:
            fields["cut_in_at_ms"] = asked["cut_in_at_ms"]
        self.events.log("turn_request", sample=start, index=asked["index"], retake=asked["retake"],
                        bot_audio_ms=audio_ms, attempt=self._attempt, **fields)
        left_s = (self.script.max_call_ms - samples_to_ms(start, self.rate)) / 1000.0
        timeout = max(1.0, min(float(self.client.api.timeout_s), left_s))
        self._retry_at = None
        self._pending = self.loop.run_in_executor(self.client.executor, self._request, asked["index"],
                                                  asked["retake"], asked["chunks"], timeout,
                                                  asked["cut_in_at_ms"])
        self.phase = "ask"

    def _request(self, turn: int, retake: int, chunks: List[Any], timeout: float,
                 cut_in_at_ms: Optional[int] = None):
        """On the worker thread: the bot audio joined, sent, and the answer decoded."""
        np = self._np
        pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        return self.client.ask(turn, retake, pcm, self.rate, timeout, cut_in_at_ms)

    def _poll(self, now: int, start: int):
        if self._pending is None:
            if self._retry_at is not None and start >= self._retry_at:
                self._attempt += 1
                self._submit(start)
            return self._silence
        if not self._pending.done():
            return self._silence
        from .turns import TurnError

        future, self._pending = self._pending, None
        try:
            answer, round_trip, _ = future.result()
        except TurnError as exc:
            return self._failed(start, exc)
        except Exception as exc:  # the caller's own failure to ask: not worth asking again
            return self._failed(start, TurnError(self.client.scrub(
                f"turn {self.index}: {type(exc).__name__}: {exc}")))
        self.events.log("turn_answer", sample=start, index=self.index, retake=self._asked["retake"],
                        ms=round_trip, done=answer.done, heard=answer.heard,
                        **({"heard_before_cut_in": answer.heard_before_cut_in} if answer.heard_before_cut_in else {}))
        self._answer = answer
        if answer.done and answer.line is None:
            self.ended_by = "script"  # the bot's last answer was in this request
            return None
        return self._decide(now, start, waited=False)

    def _failed(self, start: int, exc):
        message = str(exc)
        self.events.log("turn_failed", sample=start, index=self.index, status=exc.status, message=message)
        if exc.retryable:
            delay_s = exc.retry_after_s if exc.retry_after_s is not None else min(8.0, 0.5 * (2 ** self._attempt))
            retry_at = start + int(round(max(0.0, delay_s) * self.rate))
            if samples_to_ms(retry_at, self.rate) < self.script.max_call_ms:
                self._retry_at = retry_at
                return self._silence
            message += " (no time left in the call's budget to ask again)"
        self.turn_failure = message
        self.ended_by = "turn_failed"
        return None

    def _decide(self, now: int, start: int, waited: bool):
        """The answer is in: say its line, unless the bot is still speaking (wait for it first) or spoke
        long enough since the request that the line no longer answers it (ask once more)."""
        if not waited and not self._bot_quiet(now):
            rule = self.turns.wait
            self.phase = "rewait"
            self._rewait_start = start
            self.events.log("wait_start", sample=start, index=self.index, reason="bot_done",
                            silence_ms=rule.silence_ms, max_ms=rule.max_ms, again=True)
            return self._silence
        spoke = self._spoke_since(self._asked["at"])
        taken = self._retakes.get(self.index, 0)
        if spoke >= self.retake_ms and taken < self.max_retakes:
            self._retakes[self.index] = taken + 1
            self.events.log("retake", sample=start, index=self.index, bot_spoke_ms=spoke)
            self._answer = None
            self._ask(start, retake=taken + 1)
            return self._silence
        think = int(round(self._ms(start - self._asked["at"])))
        self._commit(self._answer)
        self._answer = None
        return self._start_line(start, think)

    def _commit(self, answer) -> None:
        index = self.index
        del self.lines[index:]
        self.lines.append(_Line(answer.line.text, answer.line.pcm, self.turns.wait))
        if answer.done or index >= self.turns.max:
            self.last_index = index
            return
        cut = answer.cut_in
        if cut is not None:
            wait = WaitRule(mode="bot_done", silence_ms=self.turns.wait.silence_ms, max_ms=self.turns.wait.max_ms)
            self.lines.append(_Line(cut.text, cut.pcm, wait, cut.after_bot_ms))
            if index + 1 >= self.turns.max:
                self.last_index = index + 1

    def cancel(self) -> None:
        if self._pending is not None and not self._pending.done():
            self._pending.cancel()
        self._pending = None

    async def final_request(self, turn: int, sample: int, deadline: float,
                            stop: Optional[asyncio.Event] = None) -> Optional[str]:
        """After the call: the bot's answer to the caller's last line, sent so it is heard and kept
        (RUN_API section 4.1). Returns a warning when it could not be had; the call itself is over and
        is not failed for it. Setting `stop` abandons it, so a drain or a cancel never waits on it."""
        from .turns import TurnError

        chunks, cut_in_at_ms = self._bot_audio(turn)
        cut_field = {} if cut_in_at_ms is None else {"cut_in_at_ms": cut_in_at_ms}
        attempt = 0
        while True:
            audio_ms = int(round(self._ms(sum(len(c) for c in chunks))))
            self.events.log("turn_request", sample=sample, index=turn, retake=0, bot_audio_ms=audio_ms,
                            attempt=attempt, final=True, **cut_field)
            left = deadline - time.monotonic()
            try:
                if left <= 1.0:
                    raise TurnError(f"turn {turn}: no time left in the call's budget", retryable=False)
                if stop is not None and stop.is_set():
                    raise TurnError(f"turn {turn}: the run was stopped", retryable=False)
                request = self.loop.run_in_executor(self.client.executor, self._request, turn, 0, chunks,
                                                    min(float(self.client.api.timeout_s), left), cut_in_at_ms)
                if stop is not None:
                    stopped = asyncio.ensure_future(stop.wait())
                    try:
                        await asyncio.wait({request, stopped}, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        stopped.cancel()
                    if not request.done():
                        request.cancel()
                        raise TurnError(f"turn {turn}: the run was stopped", retryable=False)
                answer, round_trip, _ = await request
            except TurnError as exc:
                self.events.log("turn_failed", sample=sample, index=turn, status=exc.status, message=str(exc))
                delay = exc.retry_after_s if exc.retry_after_s is not None else min(8.0, 0.5 * (2 ** attempt))
                if not exc.retryable or time.monotonic() + delay >= deadline - 1.0:
                    return f"the bot's last answer was not transcribed: {exc}"
                attempt += 1
                if stop is None:
                    await asyncio.sleep(delay)
                else:
                    try:
                        await asyncio.wait_for(stop.wait(), delay)
                    except asyncio.TimeoutError:
                        pass
                continue
            except Exception as exc:
                message = self.client.scrub(f"turn {turn}: {type(exc).__name__}: {exc}")
                self.events.log("turn_failed", sample=sample, index=turn, status=None, message=message)
                return f"the bot's last answer was not transcribed: {message}"
            self.events.log("turn_answer", sample=sample, index=turn, retake=0, ms=round_trip,
                            done=answer.done, heard=answer.heard, final=True,
                            **({"heard_before_cut_in": answer.heard_before_cut_in}
                               if answer.heard_before_cut_in else {}))
            return None


async def run_call(
    target: Target,
    script: Script,
    out_dir: str,
    session: Optional[str] = None,
    sources: Optional[Mapping[str, LineSource]] = None,
    tick_hook: Optional[Callable[[int], Optional[Awaitable[None]]]] = None,
    spin_ms: float = SPIN_MS,
    stop: Optional[asyncio.Event] = None,
    turn_client: Any = None,
) -> CallResult:
    """Run one call and write recording.wav, events.jsonl and call.json into `out_dir`.

    `tick_hook(frame_index)` runs before each frame is paced; tests use it to stall the loop.
    Setting `stop` ends the call at the next frame boundary the way the script's end would (stop
    message, close, every file written), with `ended_by: "aborted"`, so an interrupted run still has
    a recording it can analyse.
    A dynamic script (``mode: dynamic``) needs `turn_client` (a `turns.TurnClient`), which it asks for
    each line after the planned ones.
    Raises CallError if the WebSocket cannot be opened (call.json then records the error).
    """
    session = session or new_session_token()
    rate = target.sample_rate
    frame = frame_samples(rate)
    if script.dynamic and turn_client is None:
        raise ValueError("a dynamic script needs the turn endpoint: pass turn_client")
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
    bot_tail: Optional[_BotTail] = None
    runner: Optional[_ScriptRunner] = None
    warnings: List[str] = []
    try:
        start_msg = await transport.send_start()
        if start_msg is not None:
            events.log("start_sent", t=0)
        rx = asyncio.ensure_future(receive())
        if script.dynamic:
            from .turns import MAX_BOT_AUDIO_MS

            bot_tail = _BotTail(rate, MAX_BOT_AUDIO_MS)
            runner = _DynamicRunner(script, audio, rate, events, tracker, turn_client, bot_tail,
                                    asyncio.get_running_loop())
        else:
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
            if bot_tail is not None:
                bot_tail.add(heard)
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
        if bot_tail is not None:
            bot_tail.add(heard)
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
        if isinstance(runner, _DynamicRunner):
            runner.cancel()
            owed = runner.owed_final() if ended_by in ("script", "bot_closed") else None
            if owed is not None:
                # After the hang-up, so the request adds nothing to the call's length.
                warning = await runner.final_request(owed, total, t0 + script.max_call_ms / 1000.0, stop)
                if warning:
                    warnings.append(warning)
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
    turn_failure = runner.turn_failure if isinstance(runner, _DynamicRunner) else None
    if turn_failure:
        call["turn_failure"] = turn_failure
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
        mode=script.mode,
        turn_failure=turn_failure,
        warnings=warnings,
    )

