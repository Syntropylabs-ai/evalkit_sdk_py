"""Small structures whose size and cost stay bounded however the application behaves."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable, Generic, Iterator, Optional, Tuple, TypeVar

K = TypeVar("K")
V = TypeVar("V")

DEFAULT_MAX_ENTRIES = 10_000
DEFAULT_TTL_S = 3600.0


class BoundedTTLMap(Generic[K, V]):
    """A map of at most ``max_entries`` entries, each forgotten ``ttl_s`` seconds after it was set.

    For state kept from the start of an operation to its end (a callback run, a tool call): an operation
    whose end never arrives, because it failed in a way no callback reports, must not keep its entry for
    the life of the process. The oldest entry goes first when the map is full.
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_s: float = DEFAULT_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = max(1, int(max_entries))
        self._ttl = float(ttl_s)
        self._clock = clock
        self._data: "OrderedDict[K, Tuple[float, V]]" = OrderedDict()
        self._lock = threading.Lock()

    def _expire(self, now: float) -> None:
        data = self._data
        while data:
            key, (stamp, _value) = next(iter(data.items()))
            if now - stamp < self._ttl:
                break
            data.popitem(last=False)

    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            now = self._clock()
            data = self._data
            data.pop(key, None)
            data[key] = (now, value)
            self._expire(now)
            while len(data) > self._max:
                data.popitem(last=False)

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return default
            if self._clock() - entry[0] >= self._ttl:
                self._data.pop(key, None)
                return default
            return entry[1]

    def __getitem__(self, key: K) -> V:
        sentinel = object()
        value = self.get(key, sentinel)  # type: ignore[arg-type]
        if value is sentinel:
            raise KeyError(key)
        return value  # type: ignore[return-value]

    def __contains__(self, key: object) -> bool:
        sentinel = object()
        return self.get(key, sentinel) is not sentinel  # type: ignore[arg-type]

    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            entry = self._data.pop(key, None)
        if entry is None or self._clock() - entry[0] >= self._ttl:
            return default
        return entry[1]

    def __delitem__(self, key: K) -> None:
        with self._lock:
            del self._data[key]

    def __len__(self) -> int:
        with self._lock:
            self._expire(self._clock())
            return len(self._data)

    def __iter__(self) -> Iterator[K]:
        with self._lock:
            self._expire(self._clock())
            return iter(list(self._data.keys()))

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def reset_after_fork(self) -> None:
        # A lock another thread held at the fork stays held in the child.
        self._lock = threading.Lock()


class RateLimiter:
    """At most ``rate`` events a second, with a burst of one second's worth.

    Lock-free on purpose: it runs on every traced call and every log record. Two threads that read the
    bucket at the same moment may both pass, which lets a few events over the rate and never blocks one.
    """

    __slots__ = ("_rate", "_tokens", "_last", "_clock")

    def __init__(self, rate: Optional[float], clock: Callable[[], float] = time.monotonic) -> None:
        self._rate = float(rate) if rate and rate > 0 else 0.0
        self._tokens = self._rate
        self._clock = clock
        self._last = clock()

    @property
    def unlimited(self) -> bool:
        return self._rate <= 0

    def allow(self) -> bool:
        rate = self._rate
        if rate <= 0:
            return True
        now = self._clock()
        tokens = min(rate, self._tokens + (now - self._last) * rate)
        self._last = now
        if tokens < 1.0:
            self._tokens = tokens
            return False
        self._tokens = tokens - 1.0
        return True
