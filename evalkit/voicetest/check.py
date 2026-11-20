"""``evalkit voice-test check``: one short call to see whether a bot is reachable and speaks the format.

No backend is involved. The caller connects, waits for a greeting as the bot's first speaker says
(the same waits a run's plan gives line 0), says one recorded line, and waits for an answer; the call
is then analysed like any other, so "answered" means speech the analysis found on the bot's channel
after the line, not merely bytes on the socket. Frames the format does not define are listed, with a
hint when they look like the other format. When nothing usable comes back (no audio, or audio with no
speech in it), the first hint is the other format, the other pcm16 rates and the endpoint path.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlsplit

log = logging.getLogger("evalkit.voicetest.check")

# The line the check says: a person's voice, since a bot that detects speech with a neural voice
# activity detector (Silero, as Pipecat and LiveKit Agents use) does not hear the synthetic signal
# the runs' tests use as speech, and never answers it. "Hello, can you hear me?" spoken by the
# Kokoro-82M text-to-speech model (Apache-2.0), voice af_heart, as a 16 kHz mono PCM16 WAV.
LINE_TEXT = "Hello, can you hear me?"
LINE_WAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "check-line.wav")


def line_ms() -> int:
    """The check line's length in ms, from its WAV header."""
    import wave

    with wave.open(LINE_WAV, "rb") as w:
        return int(-(-w.getnframes() * 1000 // w.getframerate()))


def check_script(answer_timeout_ms: int, first_speaker: str = "auto") -> Dict[str, Any]:
    from .script import first_line_wait

    # Lets a bot that greets first finish before the line, and speaks to one that does not as the
    # first speaker says: at once, or when no greeting has started.
    wait = first_line_wait(first_speaker)
    longest_wait = wait.get("max_ms", wait.get("ms", 0))
    return {
        "name": "check",
        "max_call_ms": longest_wait + line_ms() + answer_timeout_ms + 2000,
        "lines": [{"text": LINE_TEXT, "wav": {"path": LINE_WAV, "trim": True}, "wait": wait}],
        "end": {"mode": "bot_done", "silence_ms": 700, "max_ms": answer_timeout_ms},
    }


def _rates(rates: List[int]) -> str:
    names = [str(r) for r in rates]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]


def _format_hint(fmt: str, sample_rate: int, path: str) -> str:
    """What to try when nothing usable came back. A bot spoken to in the wrong format usually sends
    nothing at all, since it waits for a start message of its own format, so the other format comes
    first; a pcm16 bot at another rate sends audio the analysis hears no speech in."""
    from .audio import SUPPORTED_RATES

    if fmt == "pcm16":
        others = [r for r in SUPPORTED_RATES if r != sample_rate]
        return (f"nothing usable came back in pcm16 at {sample_rate} Hz: try --format twilio, or "
                f"another --sample-rate ({_rates(others)}); then check the endpoint path ({path})")
    return (f"nothing usable came back in the twilio format: try --format pcm16 with --sample-rate "
            f"{_rates(list(SUPPORTED_RATES))}; then check the endpoint path ({path})")


def _hints(fmt: str, details: List[str], chunks: int, speech: bool, answered: bool,
           sample_rate: int, path: str = "/") -> List[str]:
    hints: List[str] = []
    joined = " ".join(details)
    if fmt == "pcm16" and ("event " in joined or "not a JSON object" in joined):
        hints.append("the bot sends JSON events, as Twilio Media Streams does: try --format twilio")
    if fmt == "twilio" and "binary frame" in joined:
        hints.append("the bot sends binary audio frames: try --format pcm16")
    if not (chunks and speech) and not hints:
        hints.append(_format_hint(fmt, sample_rate, path))
    if not chunks:
        hints.append("the bot sent no audio: if the format and path are right, check that it answers "
                     "on this URL and any --header it needs")
    elif speech and not answered:
        hints.append("the bot spoke, but not after the caller's line: it may not hear the caller "
                     "(format or sample rate), or need longer (--timeout)")
    return hints


async def check_async(target: Any, *, answer_timeout_ms: int = 8000, out_dir: Optional[str] = None,
                      session: Optional[str] = None, first_speaker: str = "auto") -> Dict[str, Any]:
    """Runs the check call against `target` and returns the findings (see `check`)."""
    from .analysis.scorecard import write_scorecard
    from .caller import CallError, run_call
    from .script import parse_script

    script = parse_script(check_script(answer_timeout_ms, first_speaker))  # a bad first speaker fails first
    keep = out_dir is not None
    directory = out_dir if keep else tempfile.mkdtemp(prefix="evalkit-voicetest-check-")
    result: Dict[str, Any] = {"url": target.describe()["url"], "format": target.format,
                              "sampleRate": target.sample_rate, "connected": False, "ok": False}
    try:
        try:
            call = await run_call(target, script, directory, session=session)
        except CallError as exc:
            result["connectError"] = str(exc)
            result["hints"] = ["nothing answered at this URL: is the bot running, and is the URL's "
                               "scheme, host, port and path right?"]
            return result
        result["connected"] = True
        events = call.events
        chunks = [e for e in events if e["type"] == "bot_chunk"]
        unknown = collections.Counter(e.get("detail") or e.get("message") for e in events
                                      if e["type"] == "error" and e.get("kind") == "unknown_frame")
        result["botChunks"] = len(chunks)
        result["botBytes"] = sum(int(e.get("bytes") or 0) for e in chunks)
        result["formatErrors"] = [{"detail": d, "count": n} for d, n in unknown.most_common()]
        result["errors"] = list(call.errors)
        closed = [e for e in events if e["type"] == "closed" and e.get("by") == "bot"]
        result["closedByBot"] = bool(closed)
        if closed:
            result["closeCode"] = closed[-1].get("code")
        result["endedBy"] = call.ended_by
        card: Dict[str, Any] = {}
        try:
            with open(write_scorecard(directory), "r", encoding="utf-8") as fh:
                card = json.load(fh)
        except Exception as exc:  # report what the socket showed even when the analysis fails
            result["errors"].append(f"analysis failed: {type(exc).__name__}: {exc}")
        figures = card.get("call") or {}
        turns = card.get("turns") or []
        delay = turns[0].get("response_delay_ms") if turns else None
        result["botSpeech"] = bool((card.get("segments") or {}).get("bot"))
        result["botSpokeFirst"] = bool(figures.get("bot_spoke_first"))
        result["firstWordMs"] = figures.get("first_word_ms")
        result["answered"] = delay is not None
        result["responseDelayMs"] = delay
        result["hints"] = _hints(target.format, [str(d) for d in unknown], len(chunks),
                                 result["botSpeech"], result["answered"], target.sample_rate,
                                 urlsplit(target.url).path or "/")
        result["ok"] = bool(result["answered"] and not unknown and not call.errors
                            and call.ended_by in ("script", "bot_closed"))
        if keep:
            result["outDir"] = os.path.abspath(directory)
        return result
    finally:
        if not keep:
            shutil.rmtree(directory, ignore_errors=True)


def check(url: str, *, format: str, sample_rate: int = 16000, headers: Optional[Mapping[str, str]] = None,
          session_query_param: str = "evalkit_session", send_start: bool = True,
          answer_timeout_ms: int = 8000, out_dir: Optional[str] = None,
          session: Optional[str] = None, first_speaker: str = "auto") -> Dict[str, Any]:
    """One short call to `url`: whether it connected, whether the bot answered the line (and how
    fast), the frames it sent that the format does not define, and hints. ``ok`` is True when the bot
    answered with no format errors. `first_speaker` (``auto``, ``bot`` or ``caller``) sets the wait
    before the line, as for a run. Needs no subscription key and sends nothing to SyntropyLabs."""
    from .transport import Target

    target = Target(url=url, format=format, sample_rate=sample_rate, headers=dict(headers or {}),
                    session_query_param=session_query_param, send_start=send_start)
    return asyncio.run(check_async(target, answer_timeout_ms=answer_timeout_ms, out_dir=out_dir,
                                   session=session, first_speaker=first_speaker))
