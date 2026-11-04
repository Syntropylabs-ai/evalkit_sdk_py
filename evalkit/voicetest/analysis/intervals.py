"""Integer-millisecond interval arithmetic shared by the metrics and the calibration.

Intervals are half-open ``(start_ms, end_ms)`` pairs on the call's sample clock. Everything here is pure
Python so the analysis has no numeric dependency beyond the VAD itself.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

Interval = Tuple[int, int]


def normalize(intervals: Iterable[Sequence[int]]) -> List[Interval]:
    """Sorted, non-empty, with overlapping or touching intervals merged."""
    items = sorted((int(s), int(e)) for s, e in intervals if int(e) > int(s))
    merged: List[Interval] = []
    for start, end in items:
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def overlap(a: Sequence[int], b: Sequence[int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def total_overlap(intervals: Iterable[Sequence[int]], span: Sequence[int]) -> int:
    return sum(overlap(iv, span) for iv in intervals)


def clip(intervals: Iterable[Sequence[int]], lo: int, hi: int) -> List[Interval]:
    out: List[Interval] = []
    for s, e in intervals:
        s2, e2 = max(s, lo), min(e, hi)
        if e2 > s2:
            out.append((s2, e2))
    return out


def subtract(base: Iterable[Sequence[int]], cut: Iterable[Sequence[int]]) -> List[Interval]:
    """``base`` with every part covered by ``cut`` removed."""
    cuts = normalize(cut)
    out: List[Interval] = []
    for s, e in normalize(base):
        cursor = s
        for cs, ce in cuts:
            if ce <= cursor or cs >= e:
                continue
            if cs > cursor:
                out.append((cursor, cs))
            cursor = max(cursor, ce)
            if cursor >= e:
                break
        if cursor < e:
            out.append((cursor, e))
    return out


def gaps(intervals: Iterable[Sequence[int]], lo: int, hi: int) -> List[Interval]:
    """The parts of ``[lo, hi)`` that no interval covers."""
    return subtract([(lo, hi)], intervals)


def percentile(values: Iterable[float], p: float) -> Optional[float]:
    """Nearest-rank percentile: always one of the observed values, so "p95 = 1,840 ms" means at least
    95% of the values were at or under 1,840 ms. None for no values."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, int(math.ceil(p / 100.0 * len(ordered))))
    return ordered[min(rank, len(ordered)) - 1]
