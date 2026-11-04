"""One voice call as the bot saw it: a ``voice.session`` root and one ``voice.turn`` span per turn.

The framework integrations translate their own events into the calls below; this module owns what a
turn is and what its span carries (voice test contract v1, section 8), so both frameworks report the
same shape and the test caller's turns line up with them by session and time.

A turn starts when the bot hears the user start an utterance and ends when the bot stops speaking its
reply. Turns are counted from the first user utterance, from 0, as the caller counts its lines; bot
speech before that (a greeting) belongs to no turn. Every time is wall-clock nanoseconds at which the
framework says the moment happened, not when the event reached us, because frameworks deliver events
late (Pipecat queues them to observers) and some report a whole turn at once (LiveKit Agents).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional

from ..._version import SDK_VERSION
from ...core.client import _new_span_id, _new_trace_id
from ...core.context import _current_span, current_agent, current_identity
from ...core.types import TraceEnvelope
from .session_token import clean_token, current_voice_session

if TYPE_CHECKING:
    from ...core.client import EvalKitClient

_log = logging.getLogger("evalkit")

NS = "evalkit.voice.v1."
TURN_INDEX = NS + "turn.index"
USER_SPEECH_END_MS = NS + "user.speech_end_ms"
EOU_DELAY_MS = NS + "eou.delay_ms"
STT_DURATION_MS = NS + "stt.duration_ms"
LLM_TTFT_MS = NS + "llm.ttft_ms"
TTS_TTFB_MS = NS + "tts.ttfb_ms"
INTERRUPTED = NS + "interrupted"
TRANSCRIPT_USER = NS + "transcript.user"
TRANSCRIPT_BOT = NS + "transcript.bot"
SESSION = NS + "session"

TIMING_ATTRIBUTES = (EOU_DELAY_MS, STT_DURATION_MS, LLM_TTFT_MS, TTS_TTFB_MS)
# Measured from the end of the user's speech, so they only belong to a turn whose user has spoken.
_RESPONSE_TIMINGS = (LLM_TTFT_MS, TTS_TTFB_MS)

_NS_PER_MS = 1_000_000


def _iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def seconds_to_ms(value: object) -> Optional[int]:
    """A duration the framework reports in seconds, as whole ms; None for a missing or unset value
    (frameworks report "not measured" as None, 0, or a negative number)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value < 0:
        return None
    return int(round(value * 1000))


def seconds_to_ns(value: object) -> Optional[int]:
    """A wall-clock time the framework reports as Unix seconds, in ns."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return int(value * 1e9)


@dataclass
class _Turn:
    index: int
    start_ns: int
    user_stop_ns: Optional[int] = None
    bot_start_ns: Optional[int] = None
    bot_end_ns: Optional[int] = None
    interrupted: bool = False
    timings: Dict[str, int] = field(default_factory=dict)
    user_text: List[str] = field(default_factory=list)
    bot_text: List[str] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        """The user's part is over: the bot decided they stopped, or it has started to reply."""
        return self.user_stop_ns is not None or self.bot_start_ns is not None


def _join(parts: List[str]) -> str:
    return "".join(parts).strip()


def _append(parts: List[str], text: str, spaced: bool) -> None:
    if not text:
        return
    if spaced and parts and not parts[-1].endswith((" ", "\n")) and not text.startswith((" ", "\n")):
        parts.append(" ")
    parts.append(text)


class VoiceTurnRecorder:
    """Builds one call's spans. Not thread-safe: a framework delivers one call's events on one loop."""

    def __init__(
        self,
        client: "EvalKitClient",
        *,
        provider: str,
        session: Optional[str] = None,
        started_ns: Optional[int] = None,
    ) -> None:
        self._ek = client
        self._provider = provider
        self._session = clean_token(session) or current_voice_session()
        identity = current_identity()
        self._user_id = identity.user_id
        self._ambient_session = identity.session_id
        self._agent = current_agent()
        # Only an EvalKit span the developer opened is a parent. A framework's own OpenTelemetry span
        # may go to an exporter other than ours, and the call's root would then point at nothing.
        parent = _current_span.get()
        self._trace_id = parent.trace_id if parent else _new_trace_id()
        self._parent_span_id = parent.span_id if parent else None
        self._root_span_id = _new_span_id()
        self._started_ns = started_ns or time.time_ns()
        self._last_ns = self._started_ns
        self._opened = 0
        self._current: Optional[_Turn] = None
        # The interrupted turn whose audio is still playing out; it ends when the bot falls silent.
        self._closing: Optional[_Turn] = None
        self._speaking: Optional[_Turn] = None
        self._bot_speaking = False
        self._closed = False
        self._error: Optional[str] = None

    @property
    def session(self) -> Optional[str]:
        # A token set for the call after the integration was made (the start message is often read
        # once the pipeline exists) still applies, as long as it is in this context.
        if self._session is None:
            self._session = current_voice_session()
        return self._session

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def turns_opened(self) -> int:
        return self._opened

    @property
    def last_seen_ns(self) -> int:
        return self._last_ns

    def set_session(self, token: object) -> None:
        clean = clean_token(token)
        if clean is not None:
            self._session = clean

    def set_started(self, at_ns: Optional[int]) -> None:
        """When the bot's session started, if it is known better than when the recorder was made."""
        if at_ns and self._opened == 0:
            self._started_ns = at_ns

    def _seen(self, at_ns: Optional[int]) -> int:
        at = at_ns or time.time_ns()
        if at > self._last_ns:
            self._last_ns = at
        return at

    # User side.

    def user_started(self, at_ns: Optional[int] = None) -> None:
        at = self._seen(at_ns)
        current = self._current
        if current is not None and not current.answered:
            return  # the user went on speaking within the same utterance
        if current is not None:
            if self._speaking is current and self._bot_speaking:
                # The bot was talking over the user: its own view of an interruption.
                current.interrupted = True
                self._finish_closing(at)
                self._closing = current
            else:
                self._finish(current, at)
        self._current = _Turn(index=self._opened, start_ns=at)
        self._opened += 1

    def user_stopped(self, at_ns: Optional[int] = None, *, speech_end_ns: Optional[int] = None) -> None:
        """The bot decided the user's utterance is over. ``speech_end_ns`` is when the speech itself
        ended, as the bot's voice activity detection saw it, which gives the end-of-utterance delay."""
        at = self._seen(at_ns)
        turn = self._current
        if turn is None:
            return
        turn.user_stop_ns = at
        if speech_end_ns is not None and turn.start_ns <= speech_end_ns <= at:
            turn.timings[EOU_DELAY_MS] = int(round((at - speech_end_ns) / _NS_PER_MS))

    def user_text(self, text: object) -> None:
        if isinstance(text, str) and text.strip() and self._current is not None:
            _append(self._current.user_text, text.strip(), spaced=True)

    # Bot side.

    def bot_started(self, at_ns: Optional[int] = None) -> None:
        at = self._seen(at_ns)
        if self._bot_speaking and self._speaking is self._current:
            return
        if self._closing is not None and self._speaking is self._closing:
            self._finish_closing(at)
        self._bot_speaking = True
        self._speaking = self._current
        turn = self._current
        if turn is not None and turn.bot_start_ns is None:
            turn.bot_start_ns = at

    def bot_stopped(self, at_ns: Optional[int] = None) -> None:
        at = self._seen(at_ns)
        if not self._bot_speaking:
            return
        self._bot_speaking = False
        turn, self._speaking = self._speaking, None
        if turn is None:
            return
        turn.bot_end_ns = at
        if turn is self._closing:
            self._closing = None
            self._emit(turn, at)

    def bot_text(self, text: object, *, spaced: bool = True) -> None:
        if not isinstance(text, str) or not text:
            return
        turn = self._speaking if self._bot_speaking else None
        if turn is None and self._current is not None and self._current.answered:
            turn = self._current
        if turn is not None:
            _append(turn.bot_text, text, spaced)

    def interrupted(self, at_ns: Optional[int] = None) -> None:
        """The bot stopped its own reply for the user (an interruption it acted on)."""
        self._seen(at_ns)
        if self._bot_speaking and self._speaking is not None:
            self._speaking.interrupted = True

    def timing(self, attribute: str, ms: Optional[int], *, replace: bool = False) -> None:
        """A per-turn figure; the first one in a turn wins, since later ones measure follow-up steps
        (the second sentence's audio, the inference after a tool call)."""
        if ms is None or attribute not in TIMING_ATTRIBUTES:
            return
        turn = self._current
        if turn is None:
            return
        if attribute in _RESPONSE_TIMINGS and not turn.answered:
            return  # left over from the reply the user just cut off
        if replace or attribute not in turn.timings:
            turn.timings[attribute] = int(ms)

    def failed(self, message: str) -> None:
        self._error = message[:500] if message else "error"

    # Ending.

    def close(self, at_ns: Optional[int] = None) -> None:
        """The call is over: the open turns end, then the root."""
        if self._closed:
            return
        at = self._seen(at_ns)
        self._closed = True
        try:
            self._finish_closing(at)
            if self._current is not None:
                self._finish(self._current, at)
                self._current = None
        finally:
            self._emit_root(at)

    def _finish_closing(self, at: int) -> None:
        turn, self._closing = self._closing, None
        if turn is not None:
            if self._speaking is turn:
                self._speaking = None
                self._bot_speaking = False
            self._emit(turn, turn.bot_end_ns or at)

    def _finish(self, turn: _Turn, at: int) -> None:
        if self._speaking is turn:
            # Still talking when the turn was cut off by the end of the call.
            self._speaking = None
            self._bot_speaking = False
            self._emit(turn, at)
            return
        self._emit(turn, turn.bot_end_ns or at)

    def _common(self) -> Dict[str, object]:
        return dict(
            service_name=self._ek._opts.service_name,
            environment=self._ek._opts.environment,
            app_version=self._ek._opts.app_version,
            user_id=self._user_id,
            session_id=self.session or self._ambient_session,
            agent_name=self._agent,
            sdk_version=SDK_VERSION,
            provider=self._provider,
        )

    def _session_attrs(self, attrs: Dict[str, object]) -> None:
        token = self.session
        if token:
            attrs[SESSION] = token
            attrs["session.id"] = token

    def _emit(self, turn: _Turn, end_ns: int) -> None:
        try:
            start_ns = turn.start_ns
            # A turn always has width: its span is placed against the caller's by time.
            end_ns = max(end_ns, start_ns + _NS_PER_MS)
            attrs: Dict[str, object] = {TURN_INDEX: turn.index, INTERRUPTED: bool(turn.interrupted)}
            if turn.user_stop_ns is not None:
                attrs[USER_SPEECH_END_MS] = max(0, int(round((turn.user_stop_ns - self._started_ns) / _NS_PER_MS)))
            attrs.update(turn.timings)
            user_text = _join(turn.user_text)
            bot_text = _join(turn.bot_text)
            attrs[TRANSCRIPT_USER] = user_text
            attrs[TRANSCRIPT_BOT] = bot_text
            self._session_attrs(attrs)
            self._ek.enqueue(TraceEnvelope(
                trace_id=self._trace_id,
                span_id=_new_span_id(),
                parent_span_id=self._root_span_id,
                created_at=_iso(start_ns),
                span_type="agent",
                status="OK",
                operation="voice.turn",
                prompt=user_text or None,
                completion=bot_text or None,
                attributes=attrs,
                latency_ms=int((end_ns - start_ns) / _NS_PER_MS),
                start_time=_iso(start_ns),
                end_time=_iso(end_ns),
                **self._common(),  # type: ignore[arg-type]
            ))
        except Exception as exc:
            _log.debug("evalkit: recording a voice turn failed: %s", exc)

    def _emit_root(self, end_ns: int) -> None:
        try:
            start_ns = self._started_ns
            end_ns = max(end_ns, start_ns + _NS_PER_MS)
            attrs: Dict[str, object] = {}
            self._session_attrs(attrs)
            self._ek.enqueue(TraceEnvelope(
                trace_id=self._trace_id,
                span_id=self._root_span_id,
                parent_span_id=self._parent_span_id,
                created_at=_iso(start_ns),
                span_type="log",
                status="ERROR" if self._error else "OK",
                status_message=self._error,
                operation="voice.session",
                attributes=attrs or None,
                latency_ms=int((end_ns - start_ns) / _NS_PER_MS),
                start_time=_iso(start_ns),
                end_time=_iso(end_ns),
                **self._common(),  # type: ignore[arg-type]
            ))
        except Exception as exc:
            _log.debug("evalkit: recording a voice session failed: %s", exc)
