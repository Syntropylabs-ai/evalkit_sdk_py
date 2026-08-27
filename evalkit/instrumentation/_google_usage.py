from typing import Any, Dict, NamedTuple


class GoogleUsage(NamedTuple):
    tokens_in: int
    tokens_out: int
    total_tokens: int
    cached_tokens: int
    tool_use_tokens: int
    reasoning_tokens: int


EMPTY_GOOGLE_USAGE = GoogleUsage(0, 0, 0, 0, 0, 0)
_EMPTY = EMPTY_GOOGLE_USAGE


def _count(usage: Any, name: str) -> int:
    try:
        return int(getattr(usage, name, 0) or 0)
    except Exception:
        return 0


def extract_google_usage(usage: Any) -> GoogleUsage:
    if usage is None:
        return _EMPTY
    prompt = _count(usage, "prompt_token_count")
    candidates = _count(usage, "candidates_token_count") or _count(usage, "response_token_count")
    cached = _count(usage, "cached_content_token_count")
    tool_use = _count(usage, "tool_use_prompt_token_count")
    reasoning = _count(usage, "thoughts_token_count")
    tokens_in = prompt + tool_use
    tokens_out = candidates + reasoning
    total = _count(usage, "total_token_count") or (tokens_in + tokens_out)
    return GoogleUsage(tokens_in, tokens_out, total, cached, tool_use, reasoning)


def merge_google_usage(prev: GoogleUsage, cur: GoogleUsage) -> GoogleUsage:
    return GoogleUsage(*(c or p for p, c in zip(prev, cur)))


def google_usage_attrs(usage: GoogleUsage) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {"gen_ai.usage.total_tokens": usage.total_tokens}
    if usage.cached_tokens:
        attrs["gen_ai.usage.cached_tokens"] = usage.cached_tokens
    if usage.tool_use_tokens:
        attrs["gen_ai.usage.tool_use_prompt_tokens"] = usage.tool_use_tokens
    if usage.reasoning_tokens:
        attrs["gen_ai.usage.reasoning_tokens"] = usage.reasoning_tokens
    return attrs
