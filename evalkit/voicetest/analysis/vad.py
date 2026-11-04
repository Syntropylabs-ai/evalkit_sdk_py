"""Versioned speech detection for the post-call analysis (CONTRACT section 5).

A score is only comparable with another made by the same detector, so every parameter that can move a
boundary belongs to a named, frozen version, and the scorecard records which one it used. Changing a
parameter means adding a version, never editing one.
"""

from __future__ import annotations

import sys
from array import array
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

from .intervals import Interval

# webrtc-v2 is the default: the calibration showed webrtc-v1 misses the interruption target (CONTRACT section 5).
DEFAULT_VAD_VERSION = "webrtc-v2"


@dataclass(frozen=True)
class VadSpec:
    version: str
    aggressiveness: int
    frame_ms: int
    start_frames: int
    end_frames: int
    min_segment_ms: int
    # "forward" runs the detector once. "forward_and_reverse" also runs it on the time-reversed audio and
    # keeps a frame only when both passes call it voiced: WebRTC VAD holds its decision for a few frames
    # after speech stops (its hangover), and only extends forward in time, so the reverse pass cancels it.
    passes: str = "forward"
    # A bot segment detected to start less than this before the caller's line ends is taken to start
    # at the line end (a response, not talk-over): the detector places an onset only to within a
    # frame, so a bot answering at once would otherwise flip between the two.
    onset_tolerance_ms: int = 0

    def describe(self) -> Dict[str, object]:
        return asdict(self)


VAD_SPECS: Dict[str, VadSpec] = {
    "webrtc-v1": VadSpec(
        version="webrtc-v1",
        aggressiveness=2,
        frame_ms=20,
        start_frames=3,
        end_frames=15,
        min_segment_ms=100,
    ),
    # Proposed after the calibration showed webrtc-v1 ends every segment about 105 ms late (the VAD's
    # own hangover), which puts interruption_stop_ms outside the 50 ms target, and that a bot
    # answering within a frame of the line end is counted as talking over the caller.
    "webrtc-v2": VadSpec(
        version="webrtc-v2",
        aggressiveness=2,
        frame_ms=20,
        start_frames=3,
        end_frames=15,
        min_segment_ms=100,
        passes="forward_and_reverse",
        onset_tolerance_ms=20,
    ),
}


def get_spec(version: str) -> VadSpec:
    try:
        return VAD_SPECS[version]
    except KeyError:
        known = ", ".join(sorted(VAD_SPECS))
        raise ValueError(f"unknown vad version {version!r} (known: {known})") from None


def _samples(pcm16: bytes) -> array:
    samples = array("h")
    samples.frombytes(pcm16[: len(pcm16) - (len(pcm16) % 2)])
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def _to_bytes(samples: array) -> bytes:
    if sys.byteorder == "big":
        samples = array("h", samples)
        samples.byteswap()
    return samples.tobytes()


def _frame_flags(vad: object, samples: array, sample_rate: int, frame_len: int) -> List[bool]:
    buf = _to_bytes(samples)
    step = frame_len * 2
    return [bool(vad.is_speech(buf[i : i + step], sample_rate)) for i in range(0, len(buf), step)]  # type: ignore[attr-defined]


def voiced_frames(pcm16: bytes, sample_rate: int, spec: VadSpec) -> List[bool]:
    """One voiced/unvoiced decision per ``spec.frame_ms`` frame of mono little-endian PCM16."""
    import webrtcvad

    frame_len = sample_rate * spec.frame_ms // 1000
    samples = _samples(pcm16)
    remainder = len(samples) % frame_len
    if remainder:
        # Digital silence is never voiced, so padding the last partial frame cannot invent speech.
        samples.extend([0] * (frame_len - remainder))
    forward = _frame_flags(webrtcvad.Vad(spec.aggressiveness), samples, sample_rate, frame_len)
    if spec.passes == "forward":
        return forward
    if spec.passes != "forward_and_reverse":
        raise ValueError(f"unknown vad passes {spec.passes!r}")
    reversed_samples = array("h", samples)
    reversed_samples.reverse()
    backward = _frame_flags(webrtcvad.Vad(spec.aggressiveness), reversed_samples, sample_rate, frame_len)
    backward.reverse()
    return [f and b for f, b in zip(forward, backward)]


def segments_from_frames(flags: Sequence[bool], spec: VadSpec) -> List[Interval]:
    """Speech segments in ms from per-frame decisions.

    A segment starts at the first of ``start_frames`` consecutive voiced frames and ends at the end of
    the last voiced frame before ``end_frames`` consecutive unvoiced ones; shorter than
    ``min_segment_ms`` it is dropped.
    """
    frame_ms = spec.frame_ms
    segments: List[Interval] = []
    in_segment = False
    voiced_run = 0
    unvoiced_run = 0
    start = 0
    last_voiced = 0
    for k, voiced in enumerate(flags):
        if not in_segment:
            voiced_run = voiced_run + 1 if voiced else 0
            if voiced_run >= spec.start_frames:
                in_segment = True
                start = k - spec.start_frames + 1
                last_voiced = k
                unvoiced_run = 0
            continue
        if voiced:
            last_voiced = k
            unvoiced_run = 0
            continue
        unvoiced_run += 1
        if unvoiced_run >= spec.end_frames:
            segments.append((start * frame_ms, (last_voiced + 1) * frame_ms))
            in_segment = False
            voiced_run = 0
    if in_segment:
        segments.append((start * frame_ms, (last_voiced + 1) * frame_ms))
    return [(s, e) for s, e in segments if e - s >= spec.min_segment_ms]


def detect_segments(pcm16: bytes, sample_rate: int, spec: VadSpec) -> List[Interval]:
    return segments_from_frames(voiced_frames(pcm16, sample_rate, spec), spec)
