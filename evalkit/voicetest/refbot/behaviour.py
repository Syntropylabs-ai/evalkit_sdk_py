"""The reference bot's behaviour file: per-turn delays, reply lengths and injected faults, in JSON.
The format is documented in README.md ("Reference bot behaviour")."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

MODES = ("realtime", "burst")
ON_INTERRUPT = ("stop", "clear", "ignore")


class BehaviourError(ValueError):
    pass


def _ms(obj: Dict[str, Any], key: str, default: int, where: str) -> int:
    value = obj.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise BehaviourError(f"{where}: {key} must be a non-negative number of ms")
    return int(round(value))


def _check(obj: Any, allowed: set, where: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise BehaviourError(f"{where}: must be an object")
    extra = set(obj) - allowed
    if extra:
        raise BehaviourError(f"{where}: unknown fields {sorted(extra)}")
    return obj


@dataclass(frozen=True)
class Speech:
    """One stretch of bot speech: the synthetic signal of `ms`, sent at 1x or as a burst."""

    ms: int
    mode: str = "realtime"
    seed: int = 1000
    f0: float = 180.0
    level_dbfs: float = -20.0
    on_interrupt: str = "stop"
    stop_delay_ms: int = 200

    _FIELDS = {"ms", "mode", "seed", "f0", "level_dbfs", "on_interrupt", "stop_delay_ms"}

    @classmethod
    def from_json(cls, obj: Any, where: str, seed: int, length_key: str = "ms") -> "Speech":
        allowed = (cls._FIELDS - {"ms"}) | {length_key}
        obj = _check(obj, allowed, where)
        if length_key not in obj:
            raise BehaviourError(f"{where}: {length_key} is required")
        mode = obj.get("mode", "realtime")
        if mode not in MODES:
            raise BehaviourError(f"{where}: mode must be one of {MODES}")
        on_interrupt = obj.get("on_interrupt", "stop")
        if on_interrupt not in ON_INTERRUPT:
            raise BehaviourError(f"{where}: on_interrupt must be one of {ON_INTERRUPT}")
        length = _ms(obj, length_key, 0, where)
        if length <= 0:
            raise BehaviourError(f"{where}: {length_key} must be positive")
        return cls(
            ms=length,
            mode=mode,
            seed=int(obj.get("seed", seed)),
            f0=float(obj.get("f0", 180.0)),
            level_dbfs=float(obj.get("level_dbfs", -20.0)),
            on_interrupt=on_interrupt,
            stop_delay_ms=_ms(obj, "stop_delay_ms", 200, where),
        )

    def synth_json(self) -> Dict[str, Any]:
        return {"ms": self.ms, "seed": self.seed, "f0": self.f0, "level_dbfs": self.level_dbfs}


@dataclass(frozen=True)
class TalkOver:
    at_ms: int  # after the caller's detected speech start
    speech: Speech


@dataclass(frozen=True)
class Turn:
    """What the bot does for one caller utterance."""

    delay_ms: int = 800  # from the caller's detected speech end to the reply's first sample
    reply: Optional[Speech] = None  # None: stay silent this turn
    talk_over: Optional[TalkOver] = None

    @classmethod
    def from_json(cls, obj: Any, where: str, seed: int) -> "Turn":
        allowed = {"delay_ms", "silent", "talk_over", "reply_ms"} | (Speech._FIELDS - {"ms"})
        obj = _check(obj, allowed, where)
        silent = bool(obj.get("silent", False))
        reply = None
        if not silent:
            spec = {k: v for k, v in obj.items() if k not in ("delay_ms", "silent", "talk_over")}
            spec.setdefault("reply_ms", 1500)
            reply = Speech.from_json(spec, where, seed, length_key="reply_ms")
        talk_over = None
        if obj.get("talk_over") is not None:
            t = _check(obj["talk_over"], {"at_ms"} | Speech._FIELDS, where + ".talk_over")
            if "at_ms" not in t:
                raise BehaviourError(f"{where}.talk_over: at_ms is required")
            sp = Speech.from_json(
                {k: v for k, v in t.items() if k != "at_ms"}, where + ".talk_over", seed + 500
            )
            talk_over = TalkOver(at_ms=_ms(t, "at_ms", 0, where + ".talk_over"), speech=sp)
        return cls(delay_ms=_ms(obj, "delay_ms", 800, where), reply=reply, talk_over=talk_over)


@dataclass(frozen=True)
class Greeting:
    after_ms: int  # from the first caller sample
    speech: Speech


@dataclass
class Behaviour:
    endpoint_ms: int = 400  # caller silence that ends an utterance
    onset_ms: int = 60  # caller voiced audio that starts one (whole 20 ms frames)
    vad_aggressiveness: int = 2
    lead_ms: int = 20  # realtime sends stay this far ahead of playback, to absorb jitter
    burst_chunk_ms: int = 200
    marks: bool = True  # twilio: a mark after each stretch of speech
    sample_rate: int = 16000  # pcm16, when the caller sends no start message
    send_unknown_frame: bool = False  # one undefined text frame on connect, a conformance check
    greeting: Optional[Greeting] = None
    turns: List[Turn] = field(default_factory=list)
    default_turn: Turn = field(default_factory=lambda: Turn(reply=Speech(ms=1500, seed=1999)))
    raw: Dict[str, Any] = field(default_factory=dict)

    def turn(self, index: int) -> Turn:
        if 0 <= index < len(self.turns):
            return self.turns[index]
        return self.default_turn

    def all_speech(self) -> List[Speech]:
        out: List[Speech] = []
        if self.greeting:
            out.append(self.greeting.speech)
        for t in list(self.turns) + [self.default_turn]:
            if t.reply:
                out.append(t.reply)
            if t.talk_over:
                out.append(t.talk_over.speech)
        return out

    @classmethod
    def from_json(cls, obj: Any) -> "Behaviour":
        top = {
            "endpoint_ms",
            "onset_ms",
            "vad_aggressiveness",
            "lead_ms",
            "burst_chunk_ms",
            "marks",
            "sample_rate",
            "send_unknown_frame",
            "greeting",
            "turns",
            "default_turn",
        }
        obj = _check(obj if obj is not None else {}, top, "behaviour")
        onset = _ms(obj, "onset_ms", 60, "behaviour")
        if onset < 20 or onset % 20:
            raise BehaviourError("behaviour: onset_ms must be a positive multiple of 20")
        aggressiveness = int(obj.get("vad_aggressiveness", 2))
        if aggressiveness not in (0, 1, 2, 3):
            raise BehaviourError("behaviour: vad_aggressiveness must be 0-3")
        burst_chunk = _ms(obj, "burst_chunk_ms", 200, "behaviour")
        if burst_chunk < 20:
            raise BehaviourError("behaviour: burst_chunk_ms must be at least 20")
        greeting = None
        if obj.get("greeting") is not None:
            g = _check(obj["greeting"], {"after_ms"} | Speech._FIELDS, "greeting")
            greeting = Greeting(
                after_ms=_ms(g, "after_ms", 0, "greeting"),
                speech=Speech.from_json({k: v for k, v in g.items() if k != "after_ms"}, "greeting", 999),
            )
        raw_turns = obj.get("turns", [])
        if not isinstance(raw_turns, list):
            raise BehaviourError("behaviour: turns must be a list")
        turns = [Turn.from_json(t, f"turns[{i}]", 1000 + i) for i, t in enumerate(raw_turns)]
        default = (
            Turn.from_json(obj["default_turn"], "default_turn", 1999)
            if obj.get("default_turn") is not None
            else Turn(reply=Speech(ms=1500, seed=1999))
        )
        return cls(
            endpoint_ms=_ms(obj, "endpoint_ms", 400, "behaviour"),
            onset_ms=onset,
            vad_aggressiveness=aggressiveness,
            lead_ms=_ms(obj, "lead_ms", 20, "behaviour"),
            burst_chunk_ms=burst_chunk,
            marks=bool(obj.get("marks", True)),
            sample_rate=int(obj.get("sample_rate", 16000)),
            send_unknown_frame=bool(obj.get("send_unknown_frame", False)),
            greeting=greeting,
            turns=turns,
            default_turn=default,
            raw=json.loads(json.dumps(obj)),
        )


def load_behaviour(path: str) -> Behaviour:
    with open(path, "r", encoding="utf-8") as f:
        return Behaviour.from_json(json.load(f))
