"""init()'s options: what makes them invalid, and how the ones read from the environment resolve.

An invalid configuration turns the SDK into a no-op rather than an exception at startup, so each
problem is described in words that name the option and never repeat the subscription key.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from .secrets import scrub_url

DISABLE_ENV_VARS = ("EVALKIT_DISABLED", "OTEL_SDK_DISABLED")
_TRUTHY = ("1", "true", "yes", "on")


def disabled_by_env() -> Optional[str]:
    for name in DISABLE_ENV_VARS:
        if (os.environ.get(name) or "").strip().lower() in _TRUTHY:
            return name
    return None


def _number_problem(name: str, value: Any, *, minimum: float, integer: bool, allow_none: bool = False) -> Optional[str]:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{name} must be a number (got {type(value).__name__})"
    if value != value or value in (float("inf"), float("-inf")):
        return f"{name} must be a finite number"
    if integer and int(value) != value:
        return f"{name} must be a whole number (got {value!r})"
    if value < minimum:
        return f"{name} must be at least {minimum:g} (got {value!r})"
    return None


def config_problems(options: Dict[str, Any]) -> List[str]:
    """What is wrong with init()'s options, in words that never repeat the key."""
    problems: List[str] = []
    key = options["subscription_key"]
    if not isinstance(key, str) or not key.strip():
        problems.append("subscription_key is empty" if isinstance(key, str) or key is None
                        else f"subscription_key must be a string (got {type(key).__name__})")
    elif any(ord(c) < 32 or ord(c) == 127 for c in key.strip()):
        problems.append("subscription_key contains control characters")
    for name in ("base_url", "api_url"):
        url = options[name]
        if url is None and name == "api_url":
            continue
        if not isinstance(url, str) or not url.strip():
            problems.append(f"{name} is empty")
            continue
        try:
            parts = urlsplit(url.strip())
        except Exception:
            parts = None
        if parts is None or parts.scheme.lower() not in ("http", "https") or not parts.netloc:
            problems.append(f"{name} must be an http:// or https:// URL (got {scrub_url(str(url))!r})")
    checks = (
        ("max_export_batch_size", 1, True, False),
        ("scheduled_delay_ms", 1, True, False),
        ("max_queue_size", 1, True, False),
        ("max_queue_bytes", 0, True, False),
        ("max_body_bytes", 0, True, False),
        ("max_value_bytes", 64, True, True),
        ("function_rate_limit", 0, False, True),
        ("log_rate_limit", 0, False, True),
    )
    for name, minimum, integer, allow_none in checks:
        problem = _number_problem(name, options[name], minimum=minimum, integer=integer, allow_none=allow_none)
        if problem:
            problems.append(problem)
    rate = options["sample_rate"]
    if rate is not None:
        problem = _number_problem("sample_rate", rate, minimum=0, integer=False)
        if problem is None and rate > 1:
            problem = f"sample_rate must be between 0 and 1 (got {rate!r})"
        if problem:
            problems.append(problem)
    for name in ("in_app_include", "in_app_exclude", "trace_packages"):
        value = options[name]
        if value is not None and not isinstance(value, (list, tuple, set, frozenset, str)):
            problems.append(f"{name} must be a list of module names")
    return problems


def resolve_log_level(value: Any) -> int:
    """The log level option as a number: a level, a level's name, else EVALKIT_LOG_LEVEL, else WARNING."""
    if value is None:
        value = os.environ.get("EVALKIT_LOG_LEVEL") or logging.WARNING
    if isinstance(value, str):
        named = logging.getLevelName(value.strip().upper())
        if isinstance(named, int):
            return named
        try:
            return int(value)
        except ValueError:
            return logging.WARNING
    try:
        return int(value)
    except (TypeError, ValueError):
        return logging.WARNING


def _comparable(options: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(options)
    for name in ("trace_packages", "in_app_include", "in_app_exclude", "ignore_routes", "ignore_urls",
                 "ignore_functions"):
        value = out.get(name)
        if isinstance(value, (list, tuple, set, frozenset)):
            out[name] = tuple(value)
    return out


def changed_options(first: Dict[str, Any], later: Dict[str, Any]) -> List[str]:
    """The names of the options a later init() call set differently from the first."""
    old, new = _comparable(first), _comparable(later)
    return sorted(name for name in new if name in old and old[name] is not new[name] and old[name] != new[name])
