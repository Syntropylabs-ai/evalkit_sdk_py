"""The caller audio a run plan names (RUN_API section 3.3 ``audio``), downloaded once and kept on disk
under its sha256.

The plan's audio is synthesized on our side and addressed by content, so a file whose bytes hash to
the plan's sha256 is the right file whichever run asked for it: a second run, or a repeat of the same
scenario, reads it from the cache instead of downloading it again. A download is written beside its
final name and renamed only once its hash matches, so a cache entry is never partial or wrong.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from typing import Any, Dict, Mapping, Optional

from .net import HttpError, RetryPolicy, display_url, request, with_retries

log = logging.getLogger("evalkit.voicetest.audio")

CACHE_ENV = "EVALKIT_VOICETEST_CACHE"
# A caller line is seconds of audio; anything this large is not one, and must not fill the disk.
MAX_AUDIO_BYTES = 64 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AudioError(RuntimeError):
    """A line's audio could not be had: the download failed, or its bytes are not the plan's."""


def default_cache_dir() -> str:
    explicit = os.environ.get(CACHE_ENV)
    if explicit:
        return explicit
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "evalkit", "voicetest", "audio")


def spec_field(spec: Mapping[str, Any], camel: str, snake: str, default: Any = None) -> Any:
    if camel in spec:
        return spec[camel]
    return spec.get(snake, default)


def spec_sha256(spec: Mapping[str, Any]) -> str:
    value = spec.get("sha256")
    if not isinstance(value, str) or not _SHA256.match(value.strip().lower()):
        raise AudioError("the line's audio has no valid sha256")
    return value.strip().lower()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AudioCache:
    """Files named ``<sha256>.audio`` in `directory`, created on first use."""

    def __init__(self, directory: Optional[str] = None, retry: Optional[RetryPolicy] = None,
                 timeout_s: float = 60.0) -> None:
        self.directory = directory or default_cache_dir()
        self.retry = retry or RetryPolicy()
        self.timeout_s = timeout_s
        self.downloaded = 0
        self.hits = 0

    def path_for(self, sha256: str) -> str:
        return os.path.join(self.directory, sha256 + ".audio")

    def cached(self, sha256: str) -> Optional[str]:
        path = self.path_for(sha256)
        if not os.path.exists(path):
            return None
        if sha256_file(path) == sha256:
            return path
        # Corrupted on disk after it was verified: fetch it again rather than play the wrong audio.
        log.warning("cached audio %s does not match its sha256; downloading it again", sha256[:12])
        try:
            os.remove(path)
        except OSError:
            pass
        return None

    def fetch(self, spec: Mapping[str, Any]) -> str:
        """The local path of the line's audio, downloading and verifying it when it is not cached."""
        sha256 = spec_sha256(spec)
        path = self.cached(sha256)
        if path is not None:
            self.hits += 1
            return path
        url = spec.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise AudioError(f"audio {sha256[:12]} is not cached and has no http(s) url")
        os.makedirs(self.directory, exist_ok=True)
        try:
            with_retries(lambda: self._download(url, sha256), self.retry,
                         f"downloading audio {sha256[:12]}")
        except HttpError as exc:
            raise AudioError(f"audio {sha256[:12]} could not be downloaded from {display_url(url)}: "
                             f"{exc.message if exc.status is None else f'HTTP {exc.status}: {exc.message}'}") from None
        self.downloaded += 1
        return self.path_for(sha256)

    def _download(self, url: str, sha256: str) -> None:
        digest = hashlib.sha256()
        fd, tmp = tempfile.mkstemp(prefix=sha256[:12] + ".", suffix=".part", dir=self.directory)
        try:
            with os.fdopen(fd, "wb") as out:
                def sink(chunk: bytes) -> None:
                    digest.update(chunk)
                    out.write(chunk)

                # A signed URL authorises itself; our subscription key never goes to storage.
                request("GET", url, timeout=self.timeout_s, max_bytes=MAX_AUDIO_BYTES, sink=sink)
            got = digest.hexdigest()
            if got != sha256:
                raise AudioError(f"audio from {display_url(url)} has sha256 {got[:12]}..., the plan "
                                 f"says {sha256[:12]}...")
            os.replace(tmp, self.path_for(sha256))
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


def decode_audio(data: bytes, spec: Mapping[str, Any], rate: int, trim: bool = True):
    """The line's audio as int16 mono at `rate`, with near-silent edges trimmed (the caller logs its
    first and last sample as the exact speech boundaries).

    ``encoding``: ``pcm16`` (signed 16-bit little-endian mono at ``sampleRate``, or a WAV file), ``wav``,
    or ``mulaw`` (8-bit mu-law at ``sampleRate``, 8000 by default).
    """
    import numpy as np

    from .audio import line_audio, mulaw_decode, read_wav_bytes

    encoding = str(spec.get("encoding") or "pcm16").strip().lower()
    src_rate = spec_field(spec, "sampleRate", "sample_rate")
    is_wav = data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    if encoding in ("wav", "audio/wav", "audio/x-wav") or (encoding in ("pcm16", "linear16", "pcm_s16le") and is_wav):
        return read_wav_bytes(data, rate, trim=trim, name="the line's audio")
    if not isinstance(src_rate, int) or isinstance(src_rate, bool) or src_rate <= 0:
        if encoding in ("mulaw", "ulaw", "mu-law", "audio/x-mulaw"):
            src_rate = 8000
        else:
            raise AudioError("raw pcm16 audio needs its sampleRate")
    if encoding in ("pcm16", "linear16", "pcm_s16le"):
        if len(data) % 2:
            data = data[:-1]
        pcm = np.frombuffer(data, dtype="<i2").astype(np.int16)
    elif encoding in ("mulaw", "ulaw", "mu-law", "audio/x-mulaw"):
        pcm = mulaw_decode(data)
    else:
        raise AudioError(f"unsupported audio encoding {encoding!r}")
    return line_audio(pcm, int(src_rate), rate, trim=trim)


def describe(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """The audio spec as kept in call.json: everything but the signed URL's query."""
    out = {k: v for k, v in spec.items() if k != "url"}
    if isinstance(spec.get("url"), str):
        out["url"] = display_url(spec["url"])
    return out
