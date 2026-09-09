from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_ENDPOINT = "https://api.syntropylabs.ai/v1/otlp"
CONFIG_DIR = Path(".config") / "evalkit"
CONFIG_FILE = "config.env"
STATE_FILE = "state.json"


@dataclass
class SharedConfig:
    endpoint: str = DEFAULT_ENDPOINT
    key: str = ""
    capture: str = "minimal"
    tenant: str = ""

    def masked_key(self) -> str:
        if len(self.key) <= 8:
            return "••••" if self.key else "(not set)"
        return f"{self.key[:8]}…{self.key[-4:]}"


@dataclass
class State:
    vendors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    doctor: Dict[str, Any] = field(default_factory=dict)


class ConfigStore:
    def __init__(self, home: Optional[Path] = None) -> None:
        self.home = home or Path.home()
        self.dir = self.home / CONFIG_DIR

    def load(self) -> SharedConfig:
        cfg = SharedConfig()
        path = self.dir / CONFIG_FILE
        if path.exists():
            for line in path.read_text().splitlines():
                if "=" not in line or line.lstrip().startswith("#"):
                    continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip()
                if key == "EVALKIT_OTLP_ENDPOINT":
                    cfg.endpoint = value
                elif key == "EVALKIT_API_KEY":
                    cfg.key = value
                elif key == "EVALKIT_CONTENT_CAPTURE":
                    cfg.capture = value
                elif key == "EVALKIT_TENANT_ID":
                    cfg.tenant = value
        cfg.endpoint = os.getenv("EVALKIT_OTLP_ENDPOINT", cfg.endpoint)
        cfg.key = os.getenv("EVALKIT_API_KEY", cfg.key)
        cfg.capture = os.getenv("EVALKIT_CONTENT_CAPTURE", cfg.capture)
        cfg.tenant = os.getenv("EVALKIT_TENANT_ID", cfg.tenant)
        return cfg

    def save(self, cfg: SharedConfig) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / CONFIG_FILE
        body = (
            f"EVALKIT_OTLP_ENDPOINT={cfg.endpoint}\n"
            f"EVALKIT_API_KEY={cfg.key}\n"
            f"EVALKIT_CONTENT_CAPTURE={cfg.capture}\n"
            f"EVALKIT_TENANT_ID={cfg.tenant}\n"
        )
        path.write_text(body)
        os.chmod(path, 0o600)

    def load_state(self) -> State:
        path = self.dir / STATE_FILE
        if not path.exists():
            return State()
        raw = json.loads(path.read_text() or "{}")
        return State(vendors=dict(raw.get("vendors") or {}), doctor=dict(raw.get("doctor") or {}))

    def save_state(self, state: State) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / STATE_FILE
        body: Dict[str, Any] = {"vendors": state.vendors}
        if state.doctor:
            body["doctor"] = state.doctor
        path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
        os.chmod(path, 0o600)

    def purge(self) -> None:
        for name in (CONFIG_FILE, STATE_FILE):
            path = self.dir / name
            if path.exists():
                path.unlink()
        if self.dir.exists() and not any(self.dir.iterdir()):
            self.dir.rmdir()
