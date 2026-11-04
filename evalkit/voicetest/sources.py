"""Where a caller line's audio comes from.

Lines are rendered before the call starts, so producing audio can never delay the caller mid-call.
A `LineSource` turns one line's spec into PCM at the call's rate; the script names the source by
its `kind` key: `synth` (the deterministic speech-like signal), `wav` (a local file) and `audio` (the
synthesized speech a run plan names, downloaded and verified by sha256).
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .audio import SynthSpec, read_wav, synth_speech


class LineSource:
    kind = ""

    def render(self, spec: Any, rate: int, base_dir: str):
        """Return the line's audio as int16 mono at `rate`, with no leading or trailing silence:
        its first and last samples are the caller's exact speech boundaries."""
        raise NotImplementedError


class SynthSource(LineSource):
    """The deterministic speech-like signal (audio.synth_speech). Spec: {"ms", "seed", "f0", ...}."""

    kind = "synth"

    def render(self, spec: Any, rate: int, base_dir: str):
        return synth_speech(SynthSpec.from_json(spec), rate)


class WavSource(LineSource):
    """A PCM16 WAV file, any rate, resampled to the call's rate. Spec: a path relative to the
    script file, or {"path": ..., "trim": true}. Near-silent edges are trimmed by default."""

    kind = "wav"

    def render(self, spec: Any, rate: int, base_dir: str):
        if isinstance(spec, str):
            path, trim = spec, True
        elif isinstance(spec, dict) and isinstance(spec.get("path"), str):
            path, trim = spec["path"], bool(spec.get("trim", True))
        else:
            raise ValueError('a wav line needs a path, or {"path": ..., "trim": bool}')
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        pcm = read_wav(path, rate, trim=trim)
        if not len(pcm):
            raise ValueError(f"{path}: no audio above the trim level")
        return pcm


class PlanAudioSource(LineSource):
    """The audio a run plan names (RUN_API section 3.3): ``{"url", "encoding", "sampleRate",
    "sha256", "durationMs"}``, with an optional ``"trim": false``.

    `prepare()` downloads and decodes a line ahead of the call; `render()` then only reads memory, so
    a call never waits on the network or the decoder once it has connected. A line `prepare()` did
    not see is fetched by `render()` itself, which is what a single local call wants.
    """

    kind = "audio"

    def __init__(self, cache=None) -> None:
        from .audio_cache import AudioCache

        self.cache = cache if cache is not None else AudioCache()
        self._rendered: Dict[Tuple[str, int, bool], Any] = {}
        # A download that failed for good fails every later line that needs it at once, instead of
        # making each wait through the retries again.
        self._failed: Dict[str, str] = {}
        self._lock = threading.Lock()

    def prepare(self, spec: Any, rate: int):
        from .audio_cache import AudioError, decode_audio, spec_sha256

        if not isinstance(spec, Mapping):
            raise AudioError("an audio line needs an object: {url, encoding, sampleRate, sha256}")
        sha256 = spec_sha256(spec)
        trim = bool(spec.get("trim", True))
        key = (sha256, int(rate), trim)
        with self._lock:
            pcm = self._rendered.get(key)
            failed = self._failed.get(sha256)
        if pcm is not None:
            return pcm
        if failed is not None:
            raise AudioError(failed)
        try:
            path = self.cache.fetch(spec)
            with open(path, "rb") as fh:
                data = fh.read()
            try:
                pcm = decode_audio(data, spec, int(rate), trim=trim)
            except AudioError:
                raise
            except Exception as exc:  # wave.Error, a short file: bytes that are not the audio claimed
                raise AudioError(f"audio {sha256[:12]} cannot be decoded: {exc}") from None
            if not len(pcm):
                raise AudioError(f"audio {sha256[:12]} has no sound above the trim level")
        except (AudioError, OSError) as exc:
            with self._lock:
                self._failed[sha256] = str(exc)
            raise AudioError(str(exc)) from None
        with self._lock:
            self._rendered[key] = pcm
        return pcm

    def render(self, spec: Any, rate: int, base_dir: str):
        return self.prepare(spec, rate)


def default_sources(extra: Optional[Iterable[LineSource]] = None) -> Dict[str, LineSource]:
    sources: Dict[str, LineSource] = {s.kind: s for s in (WavSource(), SynthSource())}
    for s in extra or ():
        sources[s.kind] = s
    return sources


def render_lines(script, rate: int, sources: Optional[Mapping[str, LineSource]] = None) -> list:
    sources = sources or default_sources()
    out = []
    for line in script.lines:
        source = sources.get(line.source)
        if source is None:
            raise ValueError(f"line {line.index}: no source for {line.source!r}")
        out.append(source.render(line.spec, rate, script.base_dir))
    return out
