"""What every wire format shares: the target settings, the events a bot can send, and the client
interface the caller drives (CONTRACT section 3)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Union
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..audio import SUPPORTED_RATES, frame_samples

log = logging.getLogger("evalkit.voicetest.transport")

FORMATS = ("twilio", "pcm16")
TWILIO_RATE = 8000
MAX_HEADERS = 10
DEFAULT_SESSION_PARAM = "evalkit_session"
# Bursts of bot audio arrive as single messages; a few minutes of 24 kHz PCM is well under this.
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


@dataclass(repr=False)
class Target:
    url: str
    format: str
    sample_rate: int = 16000  # pcm16 only; twilio is always 8 kHz
    headers: Dict[str, str] = field(default_factory=dict)
    session_query_param: str = DEFAULT_SESSION_PARAM
    send_start: bool = True  # pcm16 only
    open_timeout_s: float = 10.0
    # How long to wait for the bot to finish the closing handshake. A bot that stops reading never
    # sees our close frame, and the call should not hang on it.
    close_timeout_s: float = 2.0
    # Opens the WebSocket in place of websockets' own `connect`, called with the same arguments. A
    # hosted runner passes one that resolves the host, refuses non-public addresses and connects to
    # the address it pinned. It is code, not a setting, so it stays out of call.json and repr.
    connector: Optional[Callable[..., Awaitable[Any]]] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.format not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}, got {self.format!r}")
        scheme = urlsplit(self.url).scheme.lower()
        if scheme not in ("ws", "wss"):
            raise ValueError("the target url must start with ws:// or wss://")
        if self.format == "twilio":
            self.sample_rate = TWILIO_RATE
        elif int(self.sample_rate) not in SUPPORTED_RATES:
            raise ValueError(f"pcm16 sample_rate must be one of {SUPPORTED_RATES}")
        self.sample_rate = int(self.sample_rate)
        if len(self.headers) > MAX_HEADERS:
            raise ValueError(f"at most {MAX_HEADERS} extra headers")
        if not self.session_query_param:
            raise ValueError("session_query_param must not be empty")

    def __repr__(self) -> str:
        # Header values and query values are the bot's secrets; a repr ends up in logs and tracebacks.
        d = self.describe()
        return f"Target(url={d['url']!r}, format={self.format!r}, sample_rate={self.sample_rate}, " \
               f"header_names={d['header_names']!r})"

    @property
    def frame_samples(self) -> int:
        return frame_samples(self.sample_rate)

    def url_with_session(self, session: str) -> str:
        parts = urlsplit(self.url)
        query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)]
        query = [(k, v) for k, v in query if k != self.session_query_param]
        query.append((self.session_query_param, session))
        return urlunsplit(parts._replace(query=urlencode(query)))

    def describe(self) -> Dict[str, Any]:
        """The target for call.json: header names but never their values, and query values other
        than the session token redacted, since URLs often carry credentials."""
        parts = urlsplit(self.url)
        query = [
            (k, v if k == self.session_query_param else "REDACTED")
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
        ]
        url = urlunsplit(parts._replace(query=urlencode(query), netloc=parts.netloc.split("@")[-1]))
        out: Dict[str, Any] = {
            "url": url,
            "format": self.format,
            "sample_rate": self.sample_rate,
            "header_names": sorted(self.headers),
            "session_query_param": self.session_query_param,
        }
        if self.format == "pcm16":
            out["send_start"] = self.send_start
        return out


# Events a bot can send, after parsing ---------------------------------------------------------


@dataclass
class BotAudio:
    pcm: Any  # int16 at the wire rate
    nbytes: int  # payload bytes as received (mu-law bytes or PCM bytes)


@dataclass
class BotClear:
    pass


@dataclass
class BotMark:
    name: str


@dataclass
class BotOther:
    """A frame the format does not define; logged and ignored."""

    detail: str


@dataclass
class TransportClosed:
    code: Optional[int] = None
    reason: str = ""


BotEvent = Union[BotAudio, BotClear, BotMark, BotOther, TransportClosed]


class Transport:
    """One call's WebSocket client. The caller owns pacing; a transport only frames and parses.

    Lifecycle: `connect()`, `send_start()`, then `send_frame()` every 20 ms with `recv()` running
    concurrently in another task, `send_mark()` when playback reaches a bot's mark, `send_stop()`,
    `close()`.
    """

    format = ""

    def __init__(self, target: Target, session: str) -> None:
        self.target = target
        self.session = session
        self.rate = target.sample_rate
        self._ws = None
        self._frames_sent = 0

    @property
    def frames_sent(self) -> int:
        return self._frames_sent

    async def connect(self) -> None:
        connect = self.target.connector
        if connect is None:
            from websockets.asyncio.client import connect

        self._ws = await connect(
            self.target.url_with_session(self.session),
            additional_headers=dict(self.target.headers) or None,
            compression=None,  # deflate adds latency and CPU to every 20 ms frame
            max_size=MAX_MESSAGE_BYTES,
            open_timeout=self.target.open_timeout_s,
            close_timeout=self.target.close_timeout_s,
            ping_interval=None,  # the media stream itself keeps the socket busy
        )

    async def send_start(self) -> Optional[dict]:
        """Send the in-band start; returns the message sent, or None."""
        raise NotImplementedError

    async def send_frame(self, pcm, t_ms: int) -> None:
        raise NotImplementedError

    async def send_mark(self, name: str) -> bool:
        """Echo a mark; False when the format has no marks."""
        return False

    async def send_stop(self) -> Optional[dict]:
        return None

    def parse(self, message: Union[str, bytes]) -> BotEvent:
        raise NotImplementedError

    async def recv(self) -> BotEvent:
        from websockets.exceptions import ConnectionClosed

        try:
            message = await self._ws.recv()
        except ConnectionClosed as exc:
            rcvd = getattr(exc, "rcvd", None)
            return TransportClosed(
                code=getattr(rcvd, "code", None), reason=str(getattr(rcvd, "reason", "") or "")
            )
        return self.parse(message)

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # closing a dead socket is not an error worth surfacing
                log.debug("close failed", exc_info=True)

    async def _send(self, data: Union[str, bytes]) -> None:
        await self._ws.send(data)


def make_transport(target: Target, session: str) -> Transport:
    if target.format == "twilio":
        from .twilio import TwilioTransport

        return TwilioTransport(target, session)
    from .pcm16 import Pcm16Transport

    return Pcm16Transport(target, session)
