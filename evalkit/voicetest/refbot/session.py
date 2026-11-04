"""One reference-bot call, as a state machine driven only by the caller audio it receives.

The bot keeps no clock of its own: its timeline is the count of caller samples received, and it
acts only while processing a caller frame. Frame k covers samples [kF, (k+1)F) and is processed at
"now" = kF, the moment the caller sent it. Everything the bot sends then reaches the caller one
network round trip after the caller's position kF, so its plan for when the caller hears each
sample is exact up to that round trip (README.md has the mapping).

A sub-frame start is placed exactly by prefixing the first chunk with silence. Realtime speech is
sent one frame per caller frame, `lead_ms` ahead of playback so jitter never starves the caller's
buffer; burst speech is sent whole at once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..audio import FrameAssembler, SpeechTracker, frame_samples, ms_to_samples
from .behaviour import Behaviour, Speech
from .truth import CallTruth

log = logging.getLogger("evalkit.voicetest.refbot")

# What the IO layer sends, in order: ("audio", pcm) | ("mark", name) | ("clear", None) | ("text", obj)
Out = Tuple[str, Any]


@dataclass
class _Utterance:
    kind: str  # "greeting" | "reply" | "talk_over"
    turn: int
    speech: Speech
    pcm: Any
    planned: int  # where its first sample should be heard
    heard_start: Optional[int] = None
    sent: int = 0
    limit: Optional[int] = None  # samples that will be heard, once stopped early
    cut: Optional[int] = None  # heard position a clear ended it at
    stop_at: Optional[int] = None  # pending stop or clear from an interruption
    stop_mode: Optional[str] = None
    mark_sent: bool = False

    @property
    def total(self) -> int:
        return len(self.pcm) if self.limit is None else self.limit

    @property
    def heard_end(self) -> Optional[int]:
        if self.heard_start is None:
            return None
        end = self.heard_start + self.total
        return end if self.cut is None else min(end, self.cut)


class RefBotSession:
    def __init__(
        self,
        behaviour: Behaviour,
        fmt: str,
        rate: int,
        truth: CallTruth,
        render: Callable[[Speech, int], Any],
    ) -> None:
        self.b = behaviour
        self.fmt = fmt
        self.rate = rate
        self.truth = truth
        self.render = render
        self.frame = frame_samples(rate)
        self.lead = ms_to_samples(behaviour.lead_ms, rate)
        self.burst_chunk = ms_to_samples(behaviour.burst_chunk_ms, rate)
        self.tracker = SpeechTracker(
            rate,
            aggressiveness=behaviour.vad_aggressiveness,
            onset_frames=behaviour.onset_ms // 20,
            offset_ms=behaviour.endpoint_ms,
        )
        self.assembler = FrameAssembler(rate)
        self.received = 0
        self.caller_turn = -1
        self.sent_until = 0  # heard position just past the last sample sent, if played contiguously
        self.current: Optional[_Utterance] = None
        self.queue: List[_Utterance] = []
        self.started = False

    # -- entry points ------------------------------------------------------------------------

    def start(self) -> List[Out]:
        """Called once the call's format and rate are known, before any caller audio."""
        self.started = True
        outs: List[Out] = []
        if self.b.send_unknown_frame:
            key = "event" if self.fmt == "twilio" else "type"
            outs.append(("text", {key: "evalkit_refbot_probe"}))
            self.truth.write("unknown_frame_sent")
        if self.b.greeting is not None:
            g = self.b.greeting
            self._schedule("greeting", -1, g.speech, ms_to_samples(g.after_ms, self.rate))
        return outs

    def on_audio(self, pcm) -> List[Out]:
        outs: List[Out] = []
        for frame in self.assembler.push(pcm):
            now = self.received
            self.received += len(frame)
            self.truth.now = now
            outs.extend(self._step(now, self.tracker.feed(frame)))
        return outs

    def on_mark(self, name: str) -> None:
        self.truth.now = self.received
        self.truth.write("mark_received", self.received, name=name)

    # -- the step ----------------------------------------------------------------------------

    def _schedule(self, kind: str, turn: int, speech: Speech, planned: int) -> None:
        u = _Utterance(kind=kind, turn=turn, speech=speech, pcm=self.render(speech, self.rate),
                       planned=planned)
        self.queue.append(u)
        self.queue.sort(key=lambda q: q.planned)
        self.truth.write("speech_planned", planned, kind=kind, turn=turn, mode=speech.mode,
                         speech_ms=speech.ms)

    def _audible(self, now: int) -> Optional[_Utterance]:
        u = self.current
        if u is None or u.heard_start is None:
            return None
        end = u.heard_end
        return u if u.heard_start <= now < end else None

    def _step(self, now: int, speech_events) -> List[Out]:
        outs: List[Out] = []
        for ev in speech_events:
            if ev.kind == "onset":
                self._on_caller_onset(now, ev.sample)
            else:
                self._on_caller_offset(now, ev.sample)
        outs.extend(self._apply_stop(now))
        outs.extend(self._send(now))
        self._retire(now + self.frame)
        return outs

    def _on_caller_onset(self, now: int, onset: int) -> None:
        self.caller_turn += 1
        turn = self.caller_turn
        self.truth.write("caller_speech_start", onset, turn=turn, detected_ms=self.truth.ms(now))
        u = self._audible(now)
        if u is not None:
            policy = u.speech.on_interrupt
            self.truth.write(
                "interruption",
                onset,
                turn=u.turn,
                kind=u.kind,
                caller_turn=turn,
                action=policy,
                stop_delay_ms=u.speech.stop_delay_ms,
            )
            if policy != "ignore" and u.stop_at is None:
                u.stop_at = onset + ms_to_samples(u.speech.stop_delay_ms, self.rate)
                u.stop_mode = policy
        for q in [q for q in self.queue if q.kind == "reply"]:
            self.queue.remove(q)
            self.truth.write("reply_cancelled", q.planned, turn=q.turn)
        talk_over = self.b.turn(turn).talk_over
        if talk_over is not None:
            planned = onset + ms_to_samples(talk_over.at_ms, self.rate)
            self._schedule("talk_over", turn, talk_over.speech, planned)

    def _on_caller_offset(self, now: int, end: int) -> None:
        turn = self.caller_turn
        self.truth.write("caller_speech_end", end, turn=turn, detected_ms=self.truth.ms(now))
        behaviour = self.b.turn(turn)
        if behaviour.reply is None:
            self.truth.write("silent_turn", end, turn=turn)
            return
        self._schedule("reply", turn, behaviour.reply, end + ms_to_samples(behaviour.delay_ms, self.rate))

    def _apply_stop(self, now: int) -> List[Out]:
        u = self.current
        if u is None or u.stop_at is None or u.stop_mode is None:
            return []
        if u.stop_mode == "clear":
            if now < u.stop_at:
                return []
            u.stop_mode = None
            end_before = u.heard_end
            u.cut = now
            u.limit = u.sent
            self.sent_until = min(self.sent_until, now)
            self.truth.write("clear_sent", now, turn=u.turn, kind=u.kind)
            if end_before is not None and end_before > now:
                self.truth.write("speech_end", now, turn=u.turn, kind=u.kind, reason="cleared")
                self.current = None
            return [("clear", None)]
        # "stop": send nothing past stop_at; what was already sent still plays
        u.stop_mode = None
        played_to = u.heard_start + u.sent if u.heard_start is not None else None
        if played_to is None:
            return []
        keep = max(u.stop_at, played_to) - u.heard_start
        if keep < len(u.pcm):
            u.limit = keep
            self.truth.write("stop_sending", u.heard_start + keep, turn=u.turn, kind=u.kind,
                             requested_ms=self.truth.ms(u.stop_at))
        return []

    def _send(self, now: int) -> List[Out]:
        import numpy as np

        outs: List[Out] = []
        if self.current is None and self.queue and self.queue[0].planned < now + self.frame:
            self.current = self.queue.pop(0)
        u = self.current
        if u is None or (u.heard_start is None and u.planned >= now + self.frame):
            return outs
        prefix = 0
        if u.heard_start is None:
            base = max(now, self.sent_until)
            u.heard_start = max(u.planned, base)
            prefix = u.heard_start - base
            self.sent_until = base
            self.truth.write(
                "speech_start",
                u.heard_start,
                turn=u.turn,
                kind=u.kind,
                mode=u.speech.mode,
                planned_ms=self.truth.ms(u.planned),
                speech_ms=u.speech.ms,
            )
        if u.speech.mode == "burst":
            n = u.total - u.sent
        else:
            n = min(u.total - u.sent, now + self.frame + self.lead - self.sent_until - prefix)
        if n > 0 or prefix:
            n = max(n, 0)
            chunk = u.pcm[u.sent : u.sent + n]
            if prefix:
                chunk = np.concatenate([np.zeros(prefix, dtype=np.int16), chunk])
            step = self.burst_chunk if u.speech.mode == "burst" else len(chunk)
            for i in range(0, len(chunk), max(1, step)):
                outs.append(("audio", chunk[i : i + step]))
            u.sent += n
            self.sent_until += len(chunk)
        if u.sent >= u.total and not u.mark_sent:
            u.mark_sent = True
            if self.fmt == "twilio" and self.b.marks:
                name = u.kind if u.turn < 0 else f"{u.kind}-{u.turn}"
                outs.append(("mark", name))
                self.truth.write("mark_sent", u.heard_start + u.total, name=name, turn=u.turn)
        return outs

    def _retire(self, horizon: int) -> None:
        """Close out the current speech once it has fully played by the end of this frame."""
        u = self.current
        if u is None or u.heard_start is None or u.sent < u.total:
            return
        end = u.heard_end
        if end is not None and end <= horizon and u.stop_mode is None:
            reason = "complete" if u.limit is None or u.limit >= len(u.pcm) else "stopped"
            self.truth.write("speech_end", end, turn=u.turn, kind=u.kind, reason=reason)
            self.current = None

    def finish(self, reason: str) -> None:
        self.truth.now = self.received
        u = self.current
        if u is not None and u.heard_start is not None:
            end = min(u.heard_end, self.received) if u.heard_end is not None else self.received
            self.truth.write("speech_end", end, turn=u.turn, kind=u.kind, reason="call_ended")
        self.truth.write("closed", self.received, reason=reason)

    def summary(self) -> Dict[str, Any]:
        return {"received_ms": self.truth.ms(self.received), "caller_turns": self.caller_turn + 1}
