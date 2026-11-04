"""The playback buffer: how received bot audio becomes "heard" audio (CONTRACT section 2).

A bot may send ten seconds of speech in one second, so the moment audio arrives is not the moment a
listener would hear it. The buffer models a phone line or speaker: it plays queued audio at exactly
1x on the sample clock, starting a chunk at its arrival position or where the previous chunk ends,
whichever is later. `clear` drops everything not yet played at the moment it arrives, and a mark
is reached when playback gets to the point where the bot queued it.

All positions are absolute sample indices at the call's wire rate on the caller's sample clock
(0 = the WebSocket opened). The caller renders the heard channel one frame behind real time, so
an arrival is never placed in a stretch that was already rendered.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple


@dataclass
class Scheduled:
    """Where one pushed chunk will be heard."""

    start: int  # heard position of its first sample
    end: int  # heard position just past its last sample
    queued: int  # samples queued ahead of the play position right after the push, this chunk included


@dataclass
class MarkReached:
    name: str
    position: int  # heard position at which playback reached it
    cleared: bool = False  # reached because a clear emptied the queue in front of it


class PlaybackBuffer:
    def __init__(self, rate: int) -> None:
        self.rate = int(rate)
        self._head = 0  # rendered up to here (exclusive)
        self._end = 0  # end of the last scheduled audio
        self._segments: Deque[list] = deque()  # [start, pcm], in position order, non-overlapping
        self._marks: Deque[Tuple[int, str]] = deque()

    @property
    def head(self) -> int:
        return self._head

    @property
    def queue_end(self) -> int:
        return self._end

    def queued_samples(self, at: Optional[int] = None) -> int:
        """Audio still to be heard after position `at` (default: the play head)."""
        at = self._head if at is None else max(int(at), self._head)
        return max(0, self._end - at)

    def push(self, pcm, arrival: int) -> Scheduled:
        """Queue a chunk that arrived at sample-clock position `arrival`."""
        import numpy as np

        x = np.asarray(pcm, dtype=np.int16)
        arrival = max(int(arrival), self._head)
        start = max(arrival, self._end)
        if len(x):
            self._segments.append([start, x])
            self._end = start + len(x)
        return Scheduled(start=start, end=start + len(x), queued=self._end - arrival)

    def push_mark(self, name: str, arrival: int) -> int:
        """Queue a mark behind the audio already queued; returns the position it will be reached."""
        arrival = max(int(arrival), self._head)
        position = max(arrival, self._end)
        self._marks.append((position, name))
        return position

    def clear(self, arrival: int) -> Tuple[int, List[MarkReached]]:
        """Drop all audio after `arrival`. Returns the dropped sample count and the marks the clear
        released: like Twilio, marks still waiting behind dropped audio are echoed at once."""
        arrival = max(int(arrival), self._head)
        dropped = 0
        kept: Deque[list] = deque()
        for start, pcm in self._segments:
            end = start + len(pcm)
            if end <= arrival:
                kept.append([start, pcm])
            elif start >= arrival:
                dropped += len(pcm)
            else:
                cut = arrival - start
                kept.append([start, pcm[:cut]])
                dropped += len(pcm) - cut
        self._segments = kept
        self._end = min(self._end, arrival)
        released: List[MarkReached] = []
        remaining: Deque[Tuple[int, str]] = deque()
        for position, name in self._marks:
            if position > arrival:
                released.append(MarkReached(name=name, position=arrival, cleared=True))
            else:
                remaining.append((position, name))
        self._marks = remaining
        return dropped, released

    def render(self, until: int):
        """Play from the head up to `until` (exclusive). Returns (pcm, marks reached in that span)."""
        import numpy as np

        until = int(until)
        n = until - self._head
        if n <= 0:
            return np.zeros(0, dtype=np.int16), []
        out = np.zeros(n, dtype=np.int16)
        while self._segments:
            start, pcm = self._segments[0]
            if start >= until:
                break
            end = start + len(pcm)
            a = max(start, self._head)
            b = min(end, until)
            if b > a:
                out[a - self._head : b - self._head] = pcm[a - start : b - start]
            if end <= until:
                self._segments.popleft()
            else:
                break
        reached: List[MarkReached] = []
        while self._marks and self._marks[0][0] <= until:
            position, name = self._marks.popleft()
            reached.append(MarkReached(name=name, position=max(position, self._head)))
        self._head = until
        if self._end < self._head:
            self._end = self._head
        return out, reached
