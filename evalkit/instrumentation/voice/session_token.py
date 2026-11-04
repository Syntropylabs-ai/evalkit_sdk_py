"""The voice test session token (voice test contract v1, section 7), read where a call carries it.

A test call carries one token: in the WebSocket URL's query, in the Twilio Media Streams `start`
message and in the raw-PCM `start` message. The bot's spans that carry it as `session.id` join the
test call's session, so its turns line up with the caller's by session and time.
"""
from __future__ import annotations

import json
import logging
from contextvars import ContextVar, Token
from dataclasses import replace
from typing import Any, Mapping, NamedTuple, Optional
from urllib.parse import parse_qs, urlsplit

from ...core.context import _current_identity, set_identity

_log = logging.getLogger("evalkit")

QUERY_PARAM = "evalkit_session"
TWILIO_PARAM = "evalkit_session"
PCM16_FIELD = "session"
LIVEKIT_ATTRIBUTE = "evalkit.session"

# A token is a simulation session id plus a call index, well under this. Anything longer is not one,
# and cutting it would join the call to the wrong session, so it is refused rather than shortened.
MAX_TOKEN_CHARS = 512

_voice_session: ContextVar[Optional[str]] = ContextVar("evalkit_voice_session", default=None)


def clean_token(value: Any) -> Optional[str]:
    """The token as sent, or None for anything that cannot be one."""
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token or len(token) > MAX_TOKEN_CHARS:
        return None
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in token):
        return None
    return token


def _as_mapping(message: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(message, Mapping):
        return message
    if isinstance(message, (bytes, bytearray)):
        try:
            message = bytes(message).decode("utf-8")
        except Exception:
            return None
    if isinstance(message, str):
        text = message.strip()
        if not text.startswith("{"):
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


def voice_session_from_url(url: Any, param: str = QUERY_PARAM) -> Optional[str]:
    """The token in a WebSocket URL's query: a full URL, a bare query string, or a mapping of query
    parameters (Starlette's ``websocket.query_params``). ``param`` is the target's
    ``session_query_param`` when it set one."""
    try:
        if url is None:
            return None
        if isinstance(url, Mapping) or (hasattr(url, "get") and hasattr(url, "keys")):
            return clean_token(url.get(param))
        text = str(url)
        query = urlsplit(text).query if ("?" in text or "://" in text) else text
        values = parse_qs(query, keep_blank_values=False).get(param)
        return clean_token(values[0]) if values else None
    except Exception:
        return None


def voice_session_from_twilio_start(message: Any) -> Optional[str]:
    """The token in a Twilio Media Streams ``start`` message, ``start.customParameters.evalkit_session``.

    Takes the message as received (JSON text or bytes) or parsed, or its inner ``start`` object.
    """
    try:
        data = _as_mapping(message)
        if data is None:
            return None
        start = data.get("start") if isinstance(data.get("start"), Mapping) else data
        params = start.get("customParameters")
        if not isinstance(params, Mapping):
            return None
        return clean_token(params.get(TWILIO_PARAM))
    except Exception:
        return None


def voice_session_from_pcm16_start(message: Any) -> Optional[str]:
    """The token in a raw-PCM ``start`` text frame, ``{"type": "start", "session": "<token>", ...}``."""
    try:
        data = _as_mapping(message)
        if data is None or data.get("type") != "start":
            return None
        return clean_token(data.get(PCM16_FIELD))
    except Exception:
        return None


def voice_session_from_start_message(message: Any) -> Optional[str]:
    """The token in either format's ``start`` message, for a bot that accepts both."""
    return voice_session_from_twilio_start(message) or voice_session_from_pcm16_start(message)


def voice_session_from_livekit_participant(participant: Any) -> Optional[str]:
    """The token in a LiveKit participant's ``evalkit.session`` attribute.

    The test caller does not join LiveKit rooms yet; this reads the attribute the contract reserves
    for when it does. Takes the participant or its attributes mapping.
    """
    try:
        attrs = participant if isinstance(participant, Mapping) else getattr(participant, "attributes", None)
        if not isinstance(attrs, Mapping):
            return None
        return clean_token(attrs.get(LIVEKIT_ATTRIBUTE))
    except Exception:
        return None


class VoiceSessionToken(NamedTuple):
    identity: Token
    voice: Token


def set_voice_session(token: Any) -> Optional[VoiceSessionToken]:
    """Make ``token`` the call's session: ``session.id`` on every span recorded in this context from
    here on, and ``evalkit.voice.v1.session`` on the voice spans. Scoped like ``set_user``: per thread
    and per asyncio task, and inherited by the tasks started after it, so call it before the
    framework starts its pipeline or session. Returns a handle for ``clear_voice_session``, or None
    when ``token`` is not a usable token."""
    clean = clean_token(token)
    if clean is None:
        if token is not None:
            _log.debug("evalkit: ignoring a voice session token that is empty, too long or not text")
        return None
    return VoiceSessionToken(set_identity(session_id=clean), _voice_session.set(clean))


def clear_voice_session(handle: Optional[VoiceSessionToken] = None) -> None:
    """Undo ``set_voice_session``: with its handle, the session in force before comes back."""
    try:
        if handle is not None:
            _voice_session.reset(handle.voice)
            _current_identity.reset(handle.identity)
            return
        _voice_session.set(None)
        _current_identity.set(replace(_current_identity.get(), session_id=None))
    except Exception:
        pass


def current_voice_session() -> Optional[str]:
    return _voice_session.get()
