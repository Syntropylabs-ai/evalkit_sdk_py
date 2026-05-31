from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class SimContext:
    message: str
    session_id: str
    state: Dict[str, Any] = field(default_factory=dict)
    turn: int = 0


@dataclass
class AgentTurnResult:
    text: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_context: List[str] = field(default_factory=list)


def resolve_persona(persona: Any) -> Tuple[str, str]:
    if isinstance(persona, dict):
        name = str(persona.get("name", "custom"))
        desc = str(persona.get("traits") or persona.get("description") or name)
        return name, desc
    if isinstance(persona, str):
        return persona, persona
    return "EXPERT", "EXPERT"
