"""Pipecat: a pipeline observer that turns the frames it sees into ``voice.turn`` spans.

Why an observer and not Pipecat's own OpenTelemetry spans: those are off unless the pipeline is built
with ``enable_tracing=True``, and their ``turn`` span starts when the pipeline starts (turn 1) and
ends a fixed timeout after the bot stops talking, so it is not the turn the test caller measures. The
frames an observer sees (user and bot started/stopped speaking, transcriptions, the text sent to
speech, metrics, interruptions) have been the same since well before 1.0, and they carry the times
the turn needs.

Frame classes are looked up by name, so a version that renames or drops one (``StartInterruptionFrame``
became ``InterruptionFrame``) loses only that signal.
"""
from __future__ import annotations

import inspect
import logging
import sys
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

from .session_token import (
    QUERY_PARAM,
    TWILIO_PARAM,
    clean_token,
    voice_session_from_start_message,
    voice_session_from_url,
)
from .turns import (
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

_ROLE_FRAMES = {
    "start": ("StartFrame",),
    "end": ("EndFrame",),
    "cancel": ("CancelFrame",),
    "error": ("ErrorFrame",),
    "vad_started": ("VADUserStartedSpeakingFrame",),
    "vad_stopped": ("VADUserStoppedSpeakingFrame",),
    "user_started": ("UserStartedSpeakingFrame",),
    "user_stopped": ("UserStoppedSpeakingFrame",),
    "bot_started": ("BotStartedSpeakingFrame",),
    "bot_stopped": ("BotStoppedSpeakingFrame",),
    "interruption": ("InterruptionFrame", "StartInterruptionFrame"),
    "transcription": ("TranscriptionFrame",),
    "tts_text": ("TTSTextFrame",),
    "metrics": ("MetricsFrame",),
}

# Pipecat names a processor after its class, and its services' classes end in these.
_SERVICE_SUFFIXES = (("STTService", "stt"), ("TTSService", "tts"), ("LLMService", "llm"))

_TIMING_BY_ROLE = {"stt": STT_DURATION_MS, "llm": LLM_TTFT_MS, "tts": TTS_TTFB_MS}

# A frame is reported once per processor it passes on older Pipecat versions, and the speaking frames
# are broadcast both ways as two frames; remembering recent ids is enough to see each moment once.
_SEEN_FRAMES = 4096
# How far before the user's turn began its speech may have started, as the bot's VAD timed it.
_MAX_VAD_LEAD_NS = 10 * 1_000_000_000
# The start message is among a connection's first frames; after these, the serializer is left alone.
_MAX_SNIFFED_MESSAGES = 50
# An EndFrame is seen as it enters the pipeline, ahead of the speech still queued before it (a
# goodbye), so the call ends when the bot falls silent, the pipeline is cleaned up, or after this.
_END_GRACE_S = 10.0

_roles: Optional[Tuple[Tuple[str, Tuple[type, ...]], ...]] = None
_observer_class: Optional[type] = None
_metrics_hint_given = False


def _frame_roles() -> Tuple[Tuple[str, Tuple[type, ...]], ...]:
    global _roles
    if _roles is None:
        import pipecat.frames.frames as frames

        found = []
        for role, names in _ROLE_FRAMES.items():
            classes = tuple(c for c in (getattr(frames, n, None) for n in names) if isinstance(c, type))
            if classes:
                found.append((role, classes))
        _roles = tuple(found)
    return _roles


def _service_classes() -> Tuple[Tuple[type, str], ...]:
    """Pipecat's service base classes, from the modules the app has loaded. A service object can only
    exist once its module is loaded, and importing one here would run its import-time warnings."""
    found = []
    for module, name, role in (
        ("pipecat.services.stt_service", "STTService", "stt"),
        ("pipecat.services.tts_service", "TTSService", "tts"),
        ("pipecat.services.llm_service", "LLMService", "llm"),
    ):
        cls = getattr(sys.modules.get(module), name, None)
        if isinstance(cls, type):
            found.append((cls, role))
    return tuple(found)


def _service_role(source: Any, processor_name: Any) -> Optional[str]:
    for cls, role in _service_classes():
        try:
            if isinstance(source, cls):
                return role
        except Exception:
            pass
    name = processor_name if isinstance(processor_name, str) else str(getattr(source, "name", "") or "")
    name = name.split("#", 1)[0]
    for suffix, role in _SERVICE_SUFFIXES:
        if suffix in name:
            return role
    return None


def _wall_ns(data: Any) -> int:
    """When the frame was pushed, on the wall clock. Observers are fed from a queue, so the time an
    observer runs is later than the push; the pipeline clock tells by how much."""
    now = time.time_ns()
    try:
        pushed = getattr(data, "timestamp", None)
        clock = data.source.get_clock()
        lag = clock.get_time() - pushed
        if isinstance(lag, int) and 0 <= lag < 60 * 1_000_000_000:
            return now - lag
    except Exception:
        pass
    return now


def _vad_moment(frame: Any, at: int, seconds_attr: str) -> int:
    """When the speech the VAD frame reports really started or ended: the VAD only decides after
    ``start_secs`` of speech or ``stop_secs`` of silence, and newer frames say when it decided."""
    decided = seconds_to_ns(getattr(frame, "timestamp", None)) or at
    lead = getattr(frame, seconds_attr, None)
    if isinstance(lead, (int, float)) and not isinstance(lead, bool) and lead > 0:
        return decided - int(lead * 1e9)
    return decided


def token_from_websocket(websocket: Any, query_param: str = QUERY_PARAM) -> Optional[str]:
    """The token a test call gave a Pipecat bot's FastAPI WebSocket: in the URL's query, or in the
    Twilio ``start`` message that ``pipecat.runner.utils.parse_telephony_websocket`` already read."""
    if websocket is None:
        return None
    try:
        token = voice_session_from_url(getattr(websocket, "query_params", None), query_param)
        if token is None:
            url = getattr(websocket, "url", None)
            token = voice_session_from_url(str(url), query_param) if url is not None else None
        if token is None:
            parsed = getattr(websocket, "_pipecat_parsed_telephony", None)
            if isinstance(parsed, tuple) and len(parsed) == 2:
                token = token_from_call_data(parsed[1])
        return token
    except Exception:
        return None


def token_from_call_data(call_data: Any) -> Optional[str]:
    """The token in what ``parse_telephony_websocket`` returns for Twilio: its custom parameters."""
    try:
        body = call_data.get("body") if hasattr(call_data, "get") else getattr(call_data, "body", None)
        if body is None:
            body = getattr(call_data, "body", None)
        if hasattr(body, "get"):
            return clean_token(body.get(TWILIO_PARAM))
    except Exception:
        pass
    return None


def watch_serializer(serializer: Any, on_token: Callable[[str], None]) -> bool:
    """Read the token from the ``start`` message, in either format, as the transport hands it to
    its serializer. For bots that let the transport read the whole stream instead of parsing the
    first messages themselves."""
    original = getattr(serializer, "deserialize", None)
    if original is None or getattr(original, "_evalkit", False):
        return False
    state = {"left": _MAX_SNIFFED_MESSAGES}

    async def deserialize(data: Any, *args: Any, **kwargs: Any) -> Any:
        if state["left"] > 0:
            state["left"] -= 1
            try:
                if isinstance(data, str) or (isinstance(data, (bytes, bytearray)) and data[:1] == b"{"):
                    token = voice_session_from_start_message(data)
                    if token is not None:
                        state["left"] = 0
                        on_token(token)
            except Exception:
                pass
        result = original(data, *args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    deserialize._evalkit = True  # type: ignore[attr-defined]
    try:
        serializer.deserialize = deserialize
    except Exception:
        return False
    return True


class _FrameLog:
    def __init__(self) -> None:
        self._ids: "OrderedDict[int, None]" = OrderedDict()

    def repeat(self, frame: Any) -> bool:
        frame_id = getattr(frame, "id", None)
        if frame_id is None:
            return False
        sibling = getattr(frame, "broadcast_sibling_id", None)
        if frame_id in self._ids or (sibling is not None and sibling in self._ids):
            return True
        self._ids[frame_id] = None
        if len(self._ids) > _SEEN_FRAMES:
            self._ids.popitem(last=False)
        return False


class _PipecatTurns:
    """What the observer does with each frame, apart from the Pipecat base class."""

    def __init__(self, recorder: VoiceTurnRecorder) -> None:
        self.recorder = recorder
        self._frames = _FrameLog()
        self._speech_started_ns: Optional[int] = None
        self._speech_ended_ns: Optional[int] = None
        # The turn whose speech-to-text figure came from a processing time, which a TTFB replaces.
        self._stt_processing_turn: Optional[int] = None
        self._saw_timings = False
        self._ending_ns: Optional[int] = None

    def push(self, data: Any) -> None:
        frame = data.frame
        if self.recorder.closed or self._frames.repeat(frame):
            return
        role = next((r for r, classes in _frame_roles() if isinstance(frame, classes)), None)
        if role is None:
            return
        at = _wall_ns(data)
        rec = self.recorder
        if role == "user_started":
            start = self._speech_started_ns
            if start is None or start > at or at - start > _MAX_VAD_LEAD_NS:
                start = at
            self._speech_started_ns = None
            rec.user_started(start)
        elif role == "user_stopped":
            ended = self._speech_ended_ns
            self._speech_ended_ns = None
            rec.user_stopped(at, speech_end_ns=ended)
        elif role == "vad_started":
            started = self._speech_started_ns
            if started is None or at - started > _MAX_VAD_LEAD_NS:
                self._speech_started_ns = _vad_moment(frame, at, "start_secs")
        elif role == "vad_stopped":
            # Without stop_secs (older versions) the VAD's decision time is not the end of speech,
            # and an end-of-utterance delay measured from it would be short by that silence.
            stop_secs = getattr(frame, "stop_secs", None)
            self._speech_ended_ns = _vad_moment(frame, at, "stop_secs") if stop_secs else None
        elif role == "bot_started":
            rec.bot_started(at)
        elif role == "bot_stopped":
            rec.bot_stopped(at)
            if self._ending_ns is not None:
                self.close()
        elif role == "interruption":
            rec.interrupted(at)
        elif role == "transcription":
            rec.user_text(getattr(frame, "text", None))
        elif role == "tts_text":
            rec.bot_text(getattr(frame, "text", None), spaced=not getattr(frame, "includes_inter_frame_spaces", False))
        elif role == "metrics":
            self._metrics(frame, data)
        elif role == "start":
            rec.set_started(at)
        elif role == "error":
            if getattr(frame, "fatal", False):
                rec.failed(str(getattr(frame, "error", "") or "fatal pipeline error"))
        elif role == "end":
            if self._ending_ns is None:
                self._ending_ns = at
                self._close_later()
        elif role == "cancel":
            rec.close(at)

    def _metrics(self, frame: Any, data: Any) -> None:
        rec = self.recorder
        for item in getattr(frame, "data", None) or ():
            kind = type(item).__name__
            if kind not in ("TTFBMetricsData", "ProcessingMetricsData"):
                continue
            role = _service_role(getattr(data, "source", None), getattr(item, "processor", None))
            attribute = _TIMING_BY_ROLE.get(role or "")
            ms = seconds_to_ms(getattr(item, "value", None))
            if attribute is None or ms is None:
                continue
            self._saw_timings = True
            turn = rec.turns_opened - 1
            if kind == "ProcessingMetricsData":
                # A segmented speech-to-text service reports only its processing time; streaming
                # ones report TTFB (end of speech to final transcript), which is the figure we want.
                if role == "stt":
                    rec.timing(attribute, ms)
                    self._stt_processing_turn = turn
                continue
            if role == "stt" and self._stt_processing_turn == turn:
                self._stt_processing_turn = None
                rec.timing(attribute, ms, replace=True)
            else:
                rec.timing(attribute, ms)

    def _close_later(self) -> None:
        try:
            import asyncio

            asyncio.get_running_loop().call_later(_END_GRACE_S, self.close)
        except Exception:
            pass

    def close(self) -> None:
        global _metrics_hint_given
        rec = self.recorder
        if rec.closed:
            return
        rec.close(max(rec.last_seen_ns, self._ending_ns or 0) if self._ending_ns else None)
        if rec.turns_opened and not self._saw_timings and not _metrics_hint_given:
            _metrics_hint_given = True
            _log.info("evalkit: this Pipecat call reported no STT, LLM or TTS metrics, so its voice turns "
                      "carry no such timings; build the pipeline worker with PipelineParams(enable_metrics=True).")


def _make_observer_class() -> type:
    global _observer_class
    if _observer_class is not None:
        return _observer_class
    from pipecat.observers.base_observer import BaseObserver

    class EvalKitPipecatObserver(BaseObserver):  # type: ignore[misc, valid-type]
        """Records the pipeline's call as a ``voice.session`` trace with one ``voice.turn`` per turn."""

        def __init__(self, turns: _PipecatTurns, **kwargs: Any) -> None:
            try:
                super().__init__(observe_every_push=False, **kwargs)
            except TypeError:
                super().__init__(**kwargs)
            self._evalkit = turns

        @property
        def session(self) -> Optional[str]:
            return self._evalkit.recorder.session

        def set_session(self, token: Any) -> None:
            """Give the call its test session token once the bot has read it."""
            self._evalkit.recorder.set_session(token)

        async def on_push_frame(self, data: Any) -> None:
            try:
                self._evalkit.push(data)
            except Exception as exc:
                _log.debug("evalkit: the Pipecat observer skipped a frame: %s", exc)

        async def cleanup(self) -> None:
            # A pipeline torn down before its EndFrame reached the observer still ends its spans.
            try:
                self._evalkit.close()
            except Exception:
                pass
            parent = getattr(super(), "cleanup", None)
            if parent is not None:
                await parent()

    _observer_class = EvalKitPipecatObserver
    return _observer_class


def pipecat_observer(
    client: "EvalKitClient",
    *,
    session_id: Optional[str] = None,
    websocket: Any = None,
    call_data: Any = None,
    serializer: Any = None,
    query_param: str = QUERY_PARAM,
) -> Any:
    token = (
        clean_token(session_id)
        or token_from_call_data(call_data)
        or token_from_websocket(websocket, query_param)
    )
    recorder = VoiceTurnRecorder(client, provider="pipecat", session=token)
    observer = _make_observer_class()(_PipecatTurns(recorder))
    if serializer is not None:
        watch_serializer(serializer, recorder.set_session)
    return observer


def instrument_pipecat(client: "EvalKitClient", task: Any, **kwargs: Any) -> Any:
    observer = pipecat_observer(client, **kwargs)
    task.add_observer(observer)
    return observer
