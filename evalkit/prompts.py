from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.parse
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

from .core.api import get_json
from .core.context import PromptRef, _current_prompt

Template = Union[str, List[Dict[str, Any]]]

_VARIABLE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}")
_cache: Dict[Tuple[str, Optional[str], Optional[int]], Tuple[float, "Prompt"]] = {}
_cache_lock = threading.Lock()


def _render_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, default=str)


def _substitute(text: str, variables: Mapping[str, Any], missing: List[str]) -> str:
    def replace(match: "re.Match[str]") -> str:
        name = match.group(1)
        if name not in variables:
            if name not in missing:
                missing.append(name)
            return match.group(0)
        return _render_value(variables[name])

    return _VARIABLE.sub(replace, text)


def compile_template(template: Template, variables: Mapping[str, Any]) -> Template:
    missing: List[str] = []
    if isinstance(template, str):
        rendered: Template = _substitute(template, variables, missing)
    else:
        rendered = [
            {**message, "content": _substitute(str(message.get("content", "")), variables, missing)}
            for message in template
        ]
    if missing:
        raise ValueError(f"missing prompt variables: {', '.join(missing)}")
    return rendered


def variables_of(template: Template) -> List[str]:
    texts = [template] if isinstance(template, str) else [str(m.get("content", "")) for m in template]
    seen: List[str] = []
    for text in texts:
        for name in _VARIABLE.findall(text):
            if name not in seen:
                seen.append(name)
    return seen


class Prompt:
    def __init__(self, payload: Mapping[str, Any]):
        self.name: str = str(payload["name"])
        self.version: int = int(payload["version"])
        self.type: str = str(payload.get("type") or "text")
        self.semver: Optional[str] = payload.get("semver")
        self.labels: List[str] = list(payload.get("labels") or [])
        self.template: Template = payload["template"]
        self.variables: List[str] = list(payload.get("variables") or variables_of(self.template))
        self.config: Dict[str, Any] = dict(payload.get("config") or {})
        self.compiled: Optional[Template] = payload.get("compiled")
        self._token = None

    @property
    def ref(self) -> PromptRef:
        return PromptRef(name=self.name, version=self.version)

    def compile(self, variables: Optional[Mapping[str, Any]] = None) -> Template:
        if variables is None and self.compiled is not None:
            return self.compiled
        return compile_template(self.template, variables or {})

    def hyperparameters(self) -> Dict[str, Any]:
        return {"promptName": self.name, "promptVersion": self.version}

    def stamp(self) -> "Prompt":
        self._token = _current_prompt.set(self.ref)
        return self

    def unstamp(self) -> None:
        if self._token is not None:
            _current_prompt.reset(self._token)
            self._token = None
        else:
            _current_prompt.set(None)

    @contextmanager
    def stamped(self) -> Iterator["Prompt"]:
        token = _current_prompt.set(self.ref)
        try:
            yield self
        finally:
            _current_prompt.reset(token)

    def __repr__(self) -> str:
        return f"Prompt({self.ref.ref})"


def get(
    name: str,
    *,
    label: Optional[str] = None,
    version: Optional[int] = None,
    variables: Optional[Mapping[str, Any]] = None,
    api_url: Optional[str] = None,
    cache_ttl: float = 60.0,
    timeout: float = 30.0,
) -> Prompt:
    from . import _get_instance

    if not name:
        raise ValueError("name is required")
    if label is not None and version is not None:
        raise ValueError("pass either label or version, not both")

    key = (name, label, version)
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and hit[0] > now:
            prompt = hit[1]
            if variables is None:
                return prompt
            return Prompt({**_payload_of(prompt), "compiled": compile_template(prompt.template, variables)})

    client = _get_instance()
    base = (api_url or os.getenv("EVALKIT_API_URL") or client._opts.api_url).rstrip("/")
    query: Dict[str, str] = {}
    if label is not None:
        query["label"] = label
    if version is not None:
        query["version"] = str(version)
    if variables is not None:
        query["compile"] = "true"
        query["variables"] = json.dumps(dict(variables), default=str)
    url = f"{base}/v1/prompts/{urllib.parse.quote(name, safe='')}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    payload = get_json(client, url, timeout=timeout, span_name=f"GET /v1/prompts/{name}")
    prompt = Prompt(payload)
    if variables is not None and prompt.compiled is None:
        prompt.compiled = compile_template(prompt.template, variables)

    if cache_ttl > 0:
        cached = Prompt({**_payload_of(prompt), "compiled": None})
        with _cache_lock:
            _cache[key] = (now + cache_ttl, cached)
            _cache[(name, None, prompt.version)] = (now + cache_ttl, cached)
    return prompt


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _payload_of(prompt: Prompt) -> Dict[str, Any]:
    return {
        "name": prompt.name,
        "version": prompt.version,
        "type": prompt.type,
        "semver": prompt.semver,
        "labels": list(prompt.labels),
        "template": prompt.template,
        "variables": list(prompt.variables),
        "config": dict(prompt.config),
    }
