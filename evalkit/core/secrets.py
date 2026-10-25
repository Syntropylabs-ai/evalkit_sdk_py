from __future__ import annotations

import functools
import json
import re
from typing import Any, Optional
from urllib.parse import unquote_plus

from ..utils.headers import MASK, SENSITIVE_RE
from .types import TraceEnvelope

SECRET_KEY_RE = re.compile(
    r"(?:^|[._-])("
    r"api[_-]?key|apikey|secret|password|passwd|authorization|credentials?|"
    r"subscription[_-]?key|x-subscription-key|cookie|set-cookie|"
    r"private[_-]?key|access[_-]?key|client[_-]?secret"
    r")(?:$|[._-])",
    re.IGNORECASE,
)

SECRET_TOKEN_KEY_RE = re.compile(r"(?:^|[._-])token$", re.IGNORECASE)

# Keys that end in "token" because they measure or price tokens rather than hold
# one: OpenTelemetry's gen_ai.server.time_to_first_token and
# gen_ai.server.time_per_output_token on bridged spans, and per-token prices such
# as input_cost_per_token, input_cost_per_audio_token and
# output_cost_per_reasoning_token in LiteLLM's price sheet. A price says so before
# "per" (cost_per_..., price_per_...); a bare "per" is followed only by the kinds of
# token a model is billed for. Masking them blanks out time-to-first-token and cost
# and protects nothing. A per-something token outside those kinds (per_user_token)
# could hold a credential and is still masked. The TypeScript SDK decides by the same
# rule, so a key masks alike from either SDK.
TELEMETRY_TOKEN_KEY_RE = re.compile(
    r"(?:^|[._-])(?:"
    r"to[._-]first"
    r"|(?:cost|price)[._-]per(?:[._-][a-z0-9]+)*"
    r"|per(?:[._-](?:input|output|prompt|completion|generated|audio|text|image|video|cached|cache|read|write"
    r"|creation|reasoning|thinking|tool|character))*"
    r")[._-]token$",
    re.IGNORECASE,
)

SECRET_QUERY_KEY_RE = re.compile(r"(?:^|[._-])(key|sig|signature)$", re.IGNORECASE)

# The vocabulary in an assignment such as api_key='...' or "password": "...", searched for by the
# word itself so that scanning a long payload costs one pass; the full name is read back from the
# text before the match and decided by is_secret_key, so tokens= and max_tokens: stay untouched.
_ASSIGNED_SECRET_RE = re.compile(
    r"(?P<word>key|secret|passw(?:or)?d|token|credentials?|authorization|cookie)"
    r"(?P<name_quote>['\"]?)\s*[=:]\s*"
    r"(?P<quote>['\"])(?P<value>(?:[^'\"\\\n]|\\.){1,1024})(?P=quote)",
    re.IGNORECASE,
)

_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")


def _mask_assigned_secret(match: "re.Match[str]") -> str:
    text, start = match.string, match.start()
    begin = start
    while begin > 0 and start - begin < 64 and text[begin - 1] in _NAME_CHARS:
        begin -= 1
    if not is_secret_key(text[begin:start] + match.group("word")):
        return match.group(0)
    quote = match.group("quote")
    return match.group(0)[: match.start("value") - start] + MASK + quote


def _mask_url_query(match: "re.Match[str]") -> str:
    # A URL written into free text (an exception message, a log line) keeps its path, and its query
    # loses the values of parameters named like a credential (?api_key=..., &sig=...).
    url = match.group(0)
    base, sep, rest = url.partition("?")
    query, hash_sep, fragment = rest.partition("#")
    return base + sep + scrub_query(query) + hash_sep + (scrub_query(fragment) if fragment else "")


_URL_WITH_QUERY_RE = re.compile(r"\b(?:https?|wss?)://[^\s'\"<>?#]*\?[^\s'\"<>]*", re.IGNORECASE)


SECRET_PATTERNS = (
    # A whole PEM private key, or what is left of one when the text was cut inside it.
    (re.compile(
        r"-----BEGIN[A-Z ]*PRIVATE KEY-----.*?(?:-----END[A-Z ]*PRIVATE KEY-----|\Z)", re.DOTALL,
    ), MASK),
    (re.compile(r"tk_live_[0-9a-f]{8,}"), "tk_live_" + MASK),
    (re.compile(r"tk_test_[0-9a-f]{8,}"), "tk_test_" + MASK),
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "sk-" + MASK),
    # A JSON Web Token: three base64url segments, the first two encoding JSON objects.
    (re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), MASK),
    (re.compile(r"(Bearer)\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE), r"\1 " + MASK),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA" + MASK),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "ghp_" + MASK),
    # Google API keys (Gemini, Maps, Firebase).
    (re.compile(r"AIza[0-9A-Za-z_-]{30,}"), "AIza" + MASK),
    # The userinfo of a URL or DSN written into free text: postgres://user:pass@host, https://key@host.
    (re.compile(r"\b([A-Za-z][A-Za-z0-9+.-]{1,30}://)[^\s/?#@'\"<>]+@"), r"\1" + MASK + "@"),
    (_ASSIGNED_SECRET_RE, _mask_assigned_secret),
)

HEADER_ATTRIBUTES = frozenset({"http.request.headers", "http.response.headers"})

URL_ATTRIBUTES = frozenset({"http.url", "url.full", "http.target", "server.address"})

QUERY_ATTRIBUTES = frozenset({"url.query"})

URL_RE = re.compile(r"^(?:https?|wss?)://\S+$", re.IGNORECASE)

_MAX_DEPTH = 12

_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_FLAG_WORDS = frozenset({"has", "is", "needs", "requires", "uses", "use", "with", "no", "supports", "missing"})


def _is_secret_spelling(key: str) -> bool:
    if SECRET_KEY_RE.search(key):
        return True
    return bool(SECRET_TOKEN_KEY_RE.search(key) and not TELEMETRY_TOKEN_KEY_RE.search(key))


def is_secret_key(key: str, header_like: bool = False) -> bool:
    if _is_secret_spelling(key):
        return True
    # camelCase keys (accessToken, secretKey, authToken) split at their capitals, so they match
    # by segment as their snake_case spellings do; maxTokens and inputCostPerAudioToken stay exempt,
    # and so does a flag about a token (hasToken, isTokenValid), which holds a yes or no, not a token.
    split = _CAMEL_BOUNDARY_RE.sub("_", key)
    if split != key:
        if SECRET_KEY_RE.search(split):
            return True
        if (SECRET_TOKEN_KEY_RE.search(split) and not TELEMETRY_TOKEN_KEY_RE.search(split)
                and split.split("_", 1)[0].lower() not in _FLAG_WORDS):
            return True
    return bool(header_like and SENSITIVE_RE.match(key))


# Words that make a parameter or function name a credential's wherever they stand in it: a
# parameter is a single variable, so key, auth or dsn in its name is enough to withhold its value.
_SECRET_NAME_WORDS = frozenset({
    "key", "keys", "apikey", "secret", "secrets", "password", "passwords", "passwd", "pwd", "passphrase",
    "credential", "credentials", "creds", "auth", "oauth", "authorization", "authenticate",
    "authentication", "cookie", "cookies", "bearer", "jwt", "dsn",
})

_NAME_WORD_RE = re.compile(r"[a-z]+|[A-Z][a-z]*|[0-9]+")



@functools.lru_cache(maxsize=4096)
def is_secret_name(name: str) -> bool:
    """Whether a parameter, keyword or function name says its value is a credential.

    Broader than is_secret_key, which decides span attribute and payload keys: a parameter
    named key, auth or dsn holds nothing else, while a payload key named key often does.
    """
    if not name:
        return False
    if is_secret_key(name):
        return True
    return any(word.lower() in _SECRET_NAME_WORDS for word in _NAME_WORD_RE.findall(name))


def is_secret_query_key(key: str) -> bool:
    decoded = unquote_plus(key)
    return is_secret_key(decoded, True) or bool(SECRET_QUERY_KEY_RE.search(decoded))


def is_url(value: str) -> bool:
    return bool(URL_RE.match(value.strip()))


def scrub_string(value: str) -> str:
    out = value
    for pattern, replacement in SECRET_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


# Text written by the application's code about a failure, rather than content: exception and log
# messages, and stack traces, which quote them. A URL in such text is a request the code made, and a
# credential in its query string is the one thing in it not worth reading. A URL in a prompt or a
# completion is the conversation's content and keeps its query.
MESSAGE_ATTRIBUTES = frozenset({
    "exception.message",
    "exception.stacktrace",
    "error.message",
    "error.stack",
    "log.message",
    "evalkit.exception.causes",
})


def scrub_message(value: str) -> str:
    """Scrub a failure message: secrets by shape, and credential parameters of any URL in it."""
    return _URL_WITH_QUERY_RE.sub(_mask_url_query, scrub_string(value))


def scrub_query(query: str) -> str:
    parts = []
    for param in query.split("&"):
        key, sep, _ = param.partition("=")
        parts.append(key + sep + MASK if sep and is_secret_query_key(key) else param)
    return "&".join(parts)


def _scrub_userinfo(value: str) -> str:
    scheme = value.find("://")
    if scheme < 0:
        return value
    start = scheme + 3
    end = value.find("/", start)
    if end < 0:
        end = len(value)
    at = value.rfind("@", start, end)
    if at <= start:
        return value
    return value[:start] + MASK + value[at:]


def scrub_url(value: str) -> str:
    head, hash_sep, fragment = value.partition("#")
    base, query_sep, query = head.partition("?")
    out = _scrub_userinfo(base) + query_sep + scrub_query(query) + hash_sep + scrub_query(fragment)
    return scrub_string(out)


def _scrub_text(value: str) -> str:
    return scrub_url(value) if is_url(value) else scrub_string(value)


def _scrub_json_string(value: str, header_like: bool, depth: int) -> Optional[str]:
    trimmed = value.lstrip()
    if not trimmed or trimmed[0] not in "{[":
        return None
    try:
        parsed = json.loads(value)
    except Exception:
        return None
    if not isinstance(parsed, (dict, list)):
        return None
    scrubbed = scrub_value(parsed, header_like, depth + 1)
    if scrubbed == parsed:
        return None
    try:
        return json.dumps(scrubbed)
    except Exception:
        return None


def scrub_value(value: Any, header_like: bool = False, depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        return value
    if isinstance(value, str):
        as_json = _scrub_json_string(value, header_like, depth)
        return _scrub_text(value) if as_json is None else as_json
    if isinstance(value, dict):
        return {
            k: MASK if is_secret_key(str(k), header_like) else scrub_value(v, header_like, depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        scrubbed = [scrub_value(v, header_like, depth + 1) for v in value]
        return tuple(scrubbed) if isinstance(value, tuple) else scrubbed
    return value


def _scrub_attr(name: str, value: Any) -> Any:
    if is_secret_key(name):
        return MASK
    if isinstance(value, str):
        if name in URL_ATTRIBUTES:
            return scrub_url(value)
        if name in QUERY_ATTRIBUTES:
            return scrub_string(scrub_query(value))
        if name in MESSAGE_ATTRIBUTES:
            scrubbed = scrub_value(value)
            return _URL_WITH_QUERY_RE.sub(_mask_url_query, scrubbed) if isinstance(scrubbed, str) else scrubbed
    return scrub_value(value, name in HEADER_ATTRIBUTES)


def _scrub_attrs(attrs: Any) -> None:
    if not isinstance(attrs, dict):
        return
    for key in list(attrs.keys()):
        attrs[key] = _scrub_attr(str(key), attrs[key])


def scrub_secrets(envelope: TraceEnvelope) -> TraceEnvelope:
    if isinstance(envelope.operation, str):
        envelope.operation = scrub_string(envelope.operation)
    if isinstance(envelope.prompt, str):
        envelope.prompt = scrub_string(envelope.prompt)
    if isinstance(envelope.completion, str):
        envelope.completion = scrub_string(envelope.completion)
    if isinstance(envelope.status_message, str):
        envelope.status_message = scrub_message(envelope.status_message)
    _scrub_attrs(envelope.attributes)
    for event in (envelope.events or ()):
        _scrub_attrs(getattr(event, "attributes", None))
    return envelope
