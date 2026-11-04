"""LiveKit Agents: ``voice.turn`` spans from an ``AgentSession``'s own events.

Why the session's events and not LiveKit's OpenTelemetry spans: those spans reach EvalKit only when
LiveKit's tracer was bound to EvalKit's provider (``livekit.agents.telemetry`` takes the global
provider when it is imported, and makes its own for LiveKit Cloud otherwise), the user's and the
agent's halves of a turn are separate spans to be paired after the fact, and their names and
attributes have changed between minor versions. The events below have kept their names and payloads
since 1.0: each committed user utterance and each spoken reply arrives as a ``ChatMessage`` through
``conversation_item_added``, with the reply's ``interrupted`` flag, and since 1.4 with the turn's
timings on ``ChatMessage.metrics``. Before 1.4, the timings come from ``metrics_collected`` and the
user and agent state changes instead.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .session_token import clean_token, voice_session_from_livekit_participant
from .turns import (
    EOU_DELAY_MS,
    LLM_TTFT_MS,
    STT_DURATION_MS,
    TTS_TTFB_MS,
    VoiceTurnRecorder,
    seconds_to_ms,
    seconds_to_ns,
)

if TYPE_CHECKING:
    from ...core.client import EvalKitClient

_log = logging.getLogger("evalkit")


def _messages_carry_metrics() -> bool:
    """Whether this LiveKit version puts a turn's timings on its chat messages (1.4 and later)."""
    try:
        from livekit.agents.llm import ChatMessage

        return "metrics" in (getattr(ChatMessage, "model_fields", None) or {})
    except Exception:
        return False


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _text(item: Any) -> str:
    try:
        text = getattr(item, "text_content", None)
        if isinstance(text, str):
            return text
        parts = [c for c in (getattr(item, "content", None) or []) if isinstance(c, str)]
        return "\n".join(parts)
    except Exception:
        return ""


class LiveKitAgentsHooks:
    """Listens to one ``AgentSession`` and records its call."""

    def __init__(
        self,
        client: "EvalKitClient",
        session: Any,
        *,
        session_id: Optional[str] = None,
        participant: Any = None,
        legacy_metrics: Optional[bool] = None,
    ) -> None:
        self._session = session
        token = clean_token(session_id) or voice_session_from_livekit_participant(participant)
        self.recorder = VoiceTurnRecorder(client, provider="livekit-agents", session=token)
        self._legacy = (not _messages_carry_metrics()) if legacy_metrics is None else legacy_metrics
        # Fallback timeline, for messages that carry no timings (older versions, unreliable VAD).
        self._user_speaking_ns: Optional[int] = None
        self._agent_speaking_ns: Optional[int] = None
        # Older versions report a turn's figures through metrics_collected: the end of utterance
        # before the user's message is added, the reply's figures with the reply's speech id, which
        # keeps a cut-off reply's late figures out of the next turn.
        self._pending_eou: Optional[Dict[str, Any]] = None
        self._pending_reply: Dict[str, Tuple[int, Optional[str]]] = {}
        self._turn_speech_id: Optional[str] = None
        self._listeners: List[Tuple[str, Any]] = []
        self._listen("conversation_item_added", self._on_item)
        self._listen("user_state_changed", self._on_user_state)
        self._listen("agent_state_changed", self._on_agent_state)
        self._listen("close", self._on_close)
        if self._legacy:
            # Deprecated from 1.4, where it logs a warning for every listener; the messages carry
            # the same figures there.
            self._listen("metrics_collected", self._on_metrics)

    @property
    def session(self) -> Optional[str]:
        return self.recorder.session

    def set_session(self, token: Any) -> None:
        self.recorder.set_session(token)

    def _listen(self, event: str, handler: Any) -> None:
        def callback(ev: Any) -> None:
            try:
                handler(ev)
            except Exception as exc:
                _log.debug("evalkit: the LiveKit Agents hook skipped a %s event: %s", event, exc)

        try:
            self._session.on(event, callback)
            self._listeners.append((event, callback))
        except Exception as exc:
            _log.debug("evalkit: could not listen to LiveKit Agents %s: %s", event, exc)

    def _unlisten(self) -> None:
        off = getattr(self._session, "off", None)
        for event, callback in self._listeners:
            try:
                if off is not None:
                    off(event, callback)
            except Exception:
                pass
        self._listeners = []

    def _token_from_room(self) -> None:
        if self.recorder.session is not None:
            return
        try:
            room_io = getattr(self._session, "_room_io", None)
            participant = getattr(room_io, "linked_participant", None) if room_io is not None else None
            token = voice_session_from_livekit_participant(participant)
            if token is not None:
                self.recorder.set_session(token)
        except Exception:
            pass

    def _on_user_state(self, ev: Any) -> None:
        at = seconds_to_ns(_get(ev, "created_at")) or time.time_ns()
        if _get(ev, "new_state") == "speaking" and self._user_speaking_ns is None:
            self._user_speaking_ns = at

    def _on_agent_state(self, ev: Any) -> None:
        at = seconds_to_ns(_get(ev, "created_at")) or time.time_ns()
        if _get(ev, "old_state") == "initializing":
            self.recorder.set_started(at)
        if _get(ev, "new_state") == "speaking":
            self._agent_speaking_ns = at

    def _on_item(self, ev: Any) -> None:
        item = _get(ev, "item")
        if item is None or (_get(item, "type") or "message") != "message":
            return
        at = seconds_to_ns(_get(ev, "created_at")) or seconds_to_ns(_get(item, "created_at")) or time.time_ns()
        role = _get(item, "role")
        if role == "user":
            self._user_message(item, at)
        elif role == "assistant":
            self._assistant_message(item, at)

    def _user_message(self, item: Any, at: int) -> None:
        self._token_from_room()
        rec = self.recorder
        metrics = _get(item, "metrics")
        eou, self._pending_eou = self._pending_eou, None
        started = seconds_to_ns(_get(metrics, "started_speaking_at")) or self._user_speaking_ns or at
        speech_end = seconds_to_ns(_get(metrics, "stopped_speaking_at"))
        eou_s = _get(metrics, "end_of_turn_delay")
        if speech_end is not None and isinstance(eou_s, (int, float)) and eou_s >= 0:
            decided = speech_end + int(eou_s * 1e9)
        else:
            decided = (eou or {}).get("decided_ns") or at
        self._user_speaking_ns = None
        rec.user_started(min(started, decided))
        rec.user_text(_text(item))
        rec.user_stopped(decided)
        legacy = eou or {}
        eou_ms = seconds_to_ms(eou_s)
        stt_ms = seconds_to_ms(_get(metrics, "transcription_delay"))
        rec.timing(EOU_DELAY_MS, eou_ms if eou_ms is not None else legacy.get("eou_ms"))
        rec.timing(STT_DURATION_MS, stt_ms if stt_ms is not None else legacy.get("stt_ms"))
        self._turn_speech_id = legacy.get("speech_id")
        early, self._pending_reply = self._pending_reply, {}
        for attribute, (ms, speech_id) in early.items():
            if self._same_reply(speech_id):
                rec.timing(attribute, ms)

    def _assistant_message(self, item: Any, at: int) -> None:
        rec = self.recorder
        metrics = _get(item, "metrics")
        started = seconds_to_ns(_get(metrics, "started_speaking_at")) or self._agent_speaking_ns or at
        stopped = seconds_to_ns(_get(metrics, "stopped_speaking_at")) or at
        self._agent_speaking_ns = None
        rec.bot_started(min(started, stopped))
        rec.bot_text(_text(item))
        if _get(item, "interrupted"):
            rec.interrupted(stopped)
        rec.bot_stopped(stopped)
        rec.timing(LLM_TTFT_MS, seconds_to_ms(_get(metrics, "llm_node_ttft")))
        rec.timing(TTS_TTFB_MS, seconds_to_ms(_get(metrics, "tts_node_ttfb")))

    def _same_reply(self, speech_id: Optional[str]) -> bool:
        return speech_id is None or self._turn_speech_id is None or speech_id == self._turn_speech_id

    def _on_metrics(self, ev: Any) -> None:
        metrics = _get(ev, "metrics")
        kind = _get(metrics, "type")
        speech_id = _get(metrics, "speech_id")
        if kind == "eou_metrics":
            # Reported as the user's turn ends, before its message is added. Its zeros mean the end
            # of speech was not detected, not an instant answer.
            eou_ms = seconds_to_ms(_get(metrics, "end_of_utterance_delay")) or None
            speech_end = seconds_to_ns(_get(metrics, "last_speaking_time"))
            if speech_end is not None and eou_ms is not None:
                decided: Optional[int] = speech_end + eou_ms * 1_000_000
            else:
                stamped = seconds_to_ns(_get(metrics, "timestamp"))
                hook = seconds_to_ns(_get(metrics, "on_user_turn_completed_delay")) or 0
                decided = stamped - hook if stamped else None
            self._pending_eou = {
                "eou_ms": eou_ms,
                "stt_ms": seconds_to_ms(_get(metrics, "transcription_delay")) or None,
                "speech_id": speech_id,
                "decided_ns": decided,
            }
            return
        if kind == "llm_metrics":
            attribute, ms = LLM_TTFT_MS, seconds_to_ms(_get(metrics, "ttft"))
        elif kind == "tts_metrics":
            attribute, ms = TTS_TTFB_MS, seconds_to_ms(_get(metrics, "ttfb"))
        else:
            return
        if ms is None:
            return
        if self._pending_eou is not None:
            self._pending_reply.setdefault(attribute, (ms, speech_id))
        elif self.recorder.turns_opened and self._same_reply(speech_id):
            self.recorder.timing(attribute, ms)

    def _on_close(self, ev: Any) -> None:
        error = _get(ev, "error")
        if error is not None:
            self.recorder.failed(type(error).__name__)
        self.recorder.close(seconds_to_ns(_get(ev, "created_at")))
        self._unlisten()


def instrument_livekit_agents(client: "EvalKitClient", session: Any, **kwargs: Any) -> LiveKitAgentsHooks:
    return LiveKitAgentsHooks(client, session, **kwargs)
