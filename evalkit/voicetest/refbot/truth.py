"""refbot_truth.jsonl: every reference-bot action on its own sample timeline.

The timeline counts caller audio received: position `sample` is the number of caller samples
received before it, at the call's wire rate, and `ms = sample * 1000 / rate` exactly (a float). A
record's `ms` is where its subject sits on that timeline (the first sample of a reply, the end of
caller speech); `now_ms` is the start of the caller frame being processed when the bot acted.
README.md ("How the truth timeline maps to the caller's") gives the mapping to the caller's clock.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional


class TruthLog:
    def __init__(self, path: Optional[str] = None, keep: bool = True) -> None:
        self.path = path
        self._file = open(path, "a", encoding="utf-8") if path else None
        self._keep = keep
        self.records: List[Dict[str, Any]] = []

    def write(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("wall", round(time.monotonic() * 1000.0, 3))
        if self._file is not None:
            self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._file.flush()
        if self._keep:
            self.records.append(record)

    def close(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.close()


class CallTruth:
    """One connection's view of the shared truth log."""

    def __init__(self, log: TruthLog, call: int, rate: int, session: Optional[str]) -> None:
        self.log = log
        self.call = call
        self.rate = rate
        self.session = session
        self.now = 0  # sample position of the frame being processed

    def ms(self, sample: int) -> float:
        return round(sample * 1000.0 / self.rate, 3)

    def write(self, type_: str, sample: Optional[int] = None, **fields: Any) -> None:
        record: Dict[str, Any] = {"call": self.call, "session": self.session, "type": type_}
        if sample is not None:
            record["sample"] = int(sample)
            record["ms"] = self.ms(sample)
        record["now_ms"] = self.ms(self.now)
        record["rate"] = self.rate
        record.update(fields)
        self.log.write(record)
