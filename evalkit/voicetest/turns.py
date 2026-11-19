"""A dynamic call's turns (RUN_API section 4.1): the bot's answer sent as heard, the next caller line
back, spoken.

The caller's frame loop never waits on this module: each request runs on a worker thread, and the loop
sends silence until its answer is in. Everything that costs time is done on that thread too: the bot's
audio is resampled to 16 kHz and written as a WAV, and the line that comes back is decoded to the
call's rate, so the loop only reads memory.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import time
import wave
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from .api import VoiceTestApi, VoiceTestApiError
from .audio_cache import MAX_AUDIO_BYTES, AudioError

# RUN_API section 4.1: the bot's answer is sent as heard, at most its last 60 s.
BOT_AUDIO_RATE = 16000
MAX_BOT_AUDIO_MS = 60_000
# The bot speaking this long in all while a line is being decided makes the caller drop the line and
# ask once more with the longer audio.
RETAKE_BOT_SPOKE_MS = 1000
MAX_RETAKES = 1
MAX_TEXT_CHARS = 2000


class TurnError(RuntimeError):
    """A turn the caller cannot go on from. `status` is the HTTP status (None when nothing answered or
    the answer could not be used); `retryable` says whether asking again may help, and `retry_after_s`
    is the server's own wait."""

    def __init__(self, message: str, status: Optional[int] = None, retryable: bool = False,
                 retry_after_s: Optional[float] = None) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after_s = retry_after_s


@dataclass
class SpokenLine:
    """A caller line the turn endpoint wrote, as PCM at the call's rate with its silent edges trimmed.
    `after_bot_ms` is set on a cut-in: how far into the bot's next answer to say it."""

    text: str
    pcm: Any
    after_bot_ms: Optional[int] = None


@dataclass
class TurnAnswer:
    turn: int
    heard: Optional[str]
    done: bool
    line: Optional[SpokenLine]
    cut_in: Optional[SpokenLine] = None
    think_ms: Optional[int] = None


def retryable_status(status: Optional[int]) -> bool:
    """A rate limit, an overloaded server or no answer at all: worth asking again. Any other 4xx (a key
    the provider refused, a turn out of order) ends the call at once."""
    return status is None or status in (408, 425, 429) or status >= 500


def bot_audio_wav(pcm: Any, rate: int) -> Tuple[str, int]:
    """The bot's answer as heard, at the call's `rate`, as a base64 16 kHz mono PCM16 WAV and its length
    in ms. A longer answer keeps its last MAX_BOT_AUDIO_MS."""
    import numpy as np

    from .audio import resample

    x = np.asarray(pcm, dtype=np.int16)
    keep = rate * MAX_BOT_AUDIO_MS // 1000
    if len(x) > keep:
        x = x[-keep:]
    y = resample(x, rate, BOT_AUDIO_RATE) if rate != BOT_AUDIO_RATE else x
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(BOT_AUDIO_RATE)
        w.writeframes(np.asarray(y, dtype="<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii"), int(round(len(y) * 1000 / BOT_AUDIO_RATE))


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TurnError(f"{where} has no text")
    return value.strip()[:MAX_TEXT_CHARS]


def _audio_bytes(spec: Mapping[str, Any], where: str, timeout_s: float) -> bytes:
    data = spec.get("data")
    if isinstance(data, str) and data:
        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            raise TurnError(f"{where}: the audio is not base64") from None
    elif isinstance(spec.get("url"), str) and spec["url"].startswith(("http://", "https://")):
        # A signed URL instead of the bytes: fetched like a plan's line, with no credential of ours.
        from .net import HttpError, request

        try:
            _, raw = request("GET", spec["url"], timeout=timeout_s, max_bytes=MAX_AUDIO_BYTES)
        except HttpError as exc:
            raise TurnError(f"{where}: the audio could not be downloaded: {exc}", exc.status,
                            retryable=retryable_status(exc.status), retry_after_s=exc.retry_after_s) from None
    else:
        raise TurnError(f"{where} has no audio")
    sha = spec.get("sha256")
    if isinstance(sha, str) and sha.strip() and hashlib.sha256(raw).hexdigest() != sha.strip().lower():
        raise TurnError(f"{where}: the audio does not match its sha256")
    return raw


def _spoken(raw: Any, where: str, rate: int, timeout_s: float, cut_in: bool = False) -> Optional[SpokenLine]:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TurnError(f"{where} is not an object")
    from .audio_cache import decode_audio

    text = _text(raw.get("text"), where)
    spec = raw.get("audio")
    if not isinstance(spec, Mapping):
        raise TurnError(f"{where} has no audio")
    data = _audio_bytes(spec, where, timeout_s)
    try:
        pcm = decode_audio(data, {"encoding": spec.get("encoding") or "wav", **{
            k: v for k, v in spec.items() if k in ("sampleRate", "sample_rate")}}, rate, trim=True)
    except AudioError as exc:
        raise TurnError(f"{where}: {exc}") from None
    except Exception as exc:  # wave.Error, a truncated file: bytes that are not the audio claimed
        raise TurnError(f"{where}: the audio cannot be decoded: {exc}") from None
    if not len(pcm):
        raise TurnError(f"{where}: the audio has no sound above the trim level")
    after = None
    if cut_in:
        after = raw.get("afterBotMs", raw.get("after_bot_ms"))
        if isinstance(after, bool) or not isinstance(after, (int, float)) or after < 0:
            raise TurnError(f"{where} has no afterBotMs")
        after = int(round(after))
    return SpokenLine(text=text, pcm=pcm, after_bot_ms=after)


def parse_answer(body: Mapping[str, Any], turn: int, rate: int, timeout_s: float = 30.0) -> TurnAnswer:
    """The turn endpoint's answer, its lines decoded to the call's `rate`. Raises TurnError (not
    retryable) when it cannot be used."""
    if not isinstance(body, Mapping):
        raise TurnError(f"turn {turn}: the answer is not an object")
    answered = body.get("turn")
    if answered is not None and answered != turn:
        raise TurnError(f"turn {turn}: the answer is for turn {answered}")
    done = body.get("done") is True
    heard = body.get("heard")
    heard = heard.strip()[:20_000] if isinstance(heard, str) else None
    line = _spoken(body.get("line"), f"turn {turn}: the line", rate, timeout_s)
    if line is None and not done:
        raise TurnError(f"turn {turn}: the answer has no line and is not done")
    cut_in = None if done else _spoken(body.get("cutIn"), f"turn {turn}: the cut-in line", rate, timeout_s,
                                       cut_in=True)
    think = body.get("thinkMs")
    think = int(think) if isinstance(think, (int, float)) and not isinstance(think, bool) else None
    return TurnAnswer(turn=turn, heard=heard or None, done=done, line=line, cut_in=cut_in, think_ms=think)


class TurnClient:
    """The turn endpoint of one dynamic call: `api` with the run's credential (the subscription key, or a
    hosted runner's run token), and the request `keys` the run was started with, if any (a hosted run
    has none). `executor` is where its blocking requests run."""

    def __init__(self, api: VoiceTestApi, run_id: str, call_index: int,
                 keys: Optional[Mapping[str, Optional[str]]] = None, executor: Any = None) -> None:
        self.api = api
        self.run_id = run_id
        self.call_index = int(call_index)
        self.keys = {k: v for k, v in (keys or {}).items() if isinstance(v, str) and v.strip()}
        self.executor = executor
        for value in self.keys.values():
            api.redact(value)

    def scrub(self, text: str) -> str:
        return self.api._scrub(text)

    def ask(self, turn: int, retake: int, bot_pcm: Any, rate: int, timeout_s: float) -> Tuple[TurnAnswer, int, int]:
        """Blocking: one request for `turn`. Returns the answer, the round trip in ms and the bot audio
        sent in ms. Raises TurnError."""
        data, audio_ms = bot_audio_wav(bot_pcm, rate)
        started = time.monotonic()
        try:
            body = self.api.next_turn(self.run_id, self.call_index, turn, retake,
                                      {"data": data, "durationMs": audio_ms}, self.keys, timeout_s=timeout_s)
        except VoiceTestApiError as exc:
            message = exc.server_message or str(exc)
            raise TurnError(self.scrub(f"turn {turn}: {message}"), exc.status,
                            retryable=retryable_status(exc.status), retry_after_s=exc.retry_after_s) from None
        round_trip = int(round((time.monotonic() - started) * 1000))
        try:
            answer = parse_answer(body, turn, rate, timeout_s)
        except TurnError as exc:
            raise TurnError(self.scrub(str(exc)), exc.status, exc.retryable, exc.retry_after_s) from None
        if answer.heard:
            answer.heard = self.scrub(answer.heard)
        return answer, round_trip, audio_ms
