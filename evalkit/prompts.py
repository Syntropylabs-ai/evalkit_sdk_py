from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

from .core.api import get_json
from .core.context import PromptRef, _current_prompt

Template = Union[str, List[Dict[str, Any]]]
_Key = Tuple[str, Optional[str], Optional[int]]

_log = logging.getLogger("evalkit")
_VARIABLE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}")
# After a refresh fails, the copy it would have replaced is served for this long before the next
# try, so callers do not each wait on an API that is not answering.
_STALE_RETRY_S = 10.0


class _Entry:
    __slots__ = ("prompt", "fresh_until", "stale_until")

    def __init__(self, prompt: "Prompt", fresh_until: float, stale_until: float) -> None:
        self.prompt = prompt
        self.fresh_until = fresh_until
        self.stale_until = stale_until


class _Flight:
    """One fetch of a prompt. Callers of the same key that arrive while it runs wait for it rather
    than fetch the same prompt again."""

    __slots__ = ("done", "error", "shareable")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.error: Optional[BaseException] = None
        # Whether a waiter may take the error as its own answer: not when it may be about the
        # fetching caller's variables (the server compiled them and refused).
        self.shareable = True


_cache: Dict[_Key, _Entry] = {}
_inflight: Dict[_Key, _Flight] = {}
_cache_lock = threading.Lock()


def _reset_lock_after_fork() -> None:
    # A lock another thread held at the fork stays held in the child, and a fetch another thread
    # ran never finishes there.
    global _cache_lock
    _cache_lock = threading.Lock()
    _inflight.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_lock_after_fork)


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


def _is_transient(exc: BaseException) -> bool:
    """No answer, or one the next try may not repeat (408, 429, 5xx): what a cached copy may stand
    in for. A refusal (401, 403, 404, 400) is the API's answer, and is raised."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (408, 429) or exc.code >= 500
    return isinstance(exc, Exception)


def _served(prompt: "Prompt", variables: Optional[Mapping[str, Any]]) -> "Prompt":
    if variables is None:
        return prompt
    return Prompt({**_payload_of(prompt), "compiled": compile_template(prompt.template, variables)})


def _fetch(name: str, label: Optional[str], version: Optional[int], variables: Optional[Mapping[str, Any]],
           api_url: Optional[str], timeout: float) -> "Prompt":
    from . import _get_instance

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
    return Prompt(get_json(client, url, timeout=timeout, span_name=f"GET /v1/prompts/{name}"))


def get(
    name: str,
    *,
    label: Optional[str] = None,
    version: Optional[int] = None,
    variables: Optional[Mapping[str, Any]] = None,
    api_url: Optional[str] = None,
    cache_ttl: float = 60.0,
    timeout: float = 30.0,
    stale_if_error: float = 86400.0,
) -> Prompt:
    if not name:
        raise ValueError("name is required")
    if label is not None and version is not None:
        raise ValueError("pass either label or version, not both")

    key: _Key = (name, label, version)
    caching = cache_ttl > 0
    mine: Optional[_Flight] = None
    while True:
        hit: Optional[Prompt] = None
        with _cache_lock:
            entry = _cache.get(key)
            if entry is not None and entry.fresh_until > time.monotonic():
                hit = entry.prompt
            else:
                flight = _inflight.get(key) if caching else None
                if flight is None and caching:
                    mine = _inflight[key] = _Flight()
        if hit is not None:
            return _served(hit, variables)
        if flight is None:
            break
        # Another caller is fetching this prompt: its answer serves this one too.
        if not flight.done.wait(timeout):
            with _cache_lock:
                entry = _cache.get(key)
                hit = entry.prompt if entry is not None and entry.stale_until > time.monotonic() else None
            if hit is not None:
                return _served(hit, variables)
            raise TimeoutError(f"prompt {name!r}: the fetch already under way did not finish within {timeout}s")
        if flight.error is not None and flight.shareable:
            raise flight.error
        # Fetched (the cache has it), or refused for that caller's variables (fetch on its own).

    try:
        prompt = _fetch(name, label, version, variables, api_url, timeout)
    except BaseException as exc:
        now = time.monotonic()
        with _cache_lock:
            entry = _cache.get(key)
            stale = entry is not None and entry.stale_until > now and _is_transient(exc)
            if stale:
                entry.fresh_until = min(entry.stale_until, now + min(_STALE_RETRY_S, cache_ttl))
            if mine is not None:
                if not stale:
                    mine.error = exc
                    mine.shareable = isinstance(exc, Exception) and (variables is None or _is_transient(exc))
                _inflight.pop(key, None)
                mine.done.set()
        if not stale:
            raise
        _log.warning("evalkit: prompt %r could not be refreshed (%s: %s); serving the cached version %d",
                     name, type(exc).__name__, exc, entry.prompt.version)
        return _served(entry.prompt, variables)

    now = time.monotonic()
    with _cache_lock:
        if caching:
            cached = _Entry(Prompt({**_payload_of(prompt), "compiled": None}), now + cache_ttl,
                            now + cache_ttl + max(0.0, stale_if_error))
            _cache[key] = cached
            _cache[(name, None, prompt.version)] = cached
        if mine is not None:
            _inflight.pop(key, None)
            mine.done.set()
    if variables is not None and prompt.compiled is None:
        prompt.compiled = compile_template(prompt.template, variables)
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
