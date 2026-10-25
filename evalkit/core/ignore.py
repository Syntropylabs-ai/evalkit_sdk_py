"""What init()'s ignore_routes, ignore_urls and ignore_functions keep out of traces.

The patterns are compiled once, when init() runs: matching runs on every request, outgoing call and
traced function, so each check is a walk over the compiled list and costs nothing when it is empty.
A pattern that cannot be read is skipped with one warning; init() never fails over one. The rules are
the TypeScript SDK's, so one list of patterns means the same in both.
"""
from __future__ import annotations

import inspect
import logging
import os
import re
from typing import Any, Callable, List, Optional, Tuple
from urllib.parse import urlsplit

from .secrets import scrub_url

_log = logging.getLogger("evalkit")

ROUTES_ENV = "EVALKIT_IGNORE_ROUTES"
URLS_ENV = "EVALKIT_IGNORE_URLS"
FUNCTIONS_ENV = "EVALKIT_IGNORE_FUNCTIONS"

_PatternType = type(re.compile(""))
_METHOD = re.compile(r"^([A-Za-z]+|\*)\s+(\S.*)$", re.S)
_ORIGIN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://[^/?#]*")
_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*|\*)://(.*)$", re.S)
_SLASHES = re.compile(r"/{2,}")
# A route template's parameter is one segment, whichever framework's spelling it uses: :id, {id}, <int:id>.
_PARAM = re.compile(r"^(?::.+|\{.*\}|<.*>)$")
_DEFAULT_PORTS = {"http": 80, "ws": 80, "https": 443, "wss": 443}
_SHOWN_MAX = 200


class RouteInfo:
    """What a callable in ``ignore_routes`` is given: the request's method (upper case), its path
    without the query string, and the framework's route template, or None when there is none.

    Some frameworks match the route only after the request is let in; there ``route`` is worked out
    when it is first read, so a callable that never reads it costs that request nothing.
    """

    __slots__ = ("method", "path", "_route", "_resolve")

    def __init__(self, method: str, path: str, route: Optional[str] = None,
                 resolve: Optional[Callable[[], Optional[str]]] = None) -> None:
        self.method = method
        self.path = path
        self._route = route
        self._resolve = resolve if route is None else None

    @property
    def route(self) -> Optional[str]:
        resolve, self._resolve = self._resolve, None
        if resolve is not None:
            try:
                found = resolve()
            except Exception:
                found = None
            self._route = found if isinstance(found, str) and found else None
        return self._route

    def __repr__(self) -> str:
        return f"RouteInfo(method={self.method!r}, path={self.path!r}, route={self._route!r})"


class UrlInfo:
    """What a callable in ``ignore_urls`` is given: the call's method (upper case, ``GET`` when the client
    did not say), its full URL, its host (lower case) and its path without the query string."""

    __slots__ = ("method", "url", "host", "path")

    def __init__(self, method: str, url: str, host: str, path: str) -> None:
        self.method = method
        self.url = url
        self.host = host
        self.path = path

    def __repr__(self) -> str:
        return f"UrlInfo(method={self.method!r}, url={self.url!r}, host={self.host!r}, path={self.path!r})"


class _Invalid(ValueError):
    pass


class _Callable:
    """A callable pattern. One that raises is a pattern that does not match, said once."""

    __slots__ = ("fn", "option", "warned")

    def __init__(self, fn: Callable[[Any], Any], option: str) -> None:
        self.fn = fn
        self.option = option
        self.warned = False

    def __call__(self, arg: Any) -> bool:
        try:
            answer = self.fn(arg)
            if inspect.isawaitable(answer):
                # The decision is needed now, before the request or call goes on: an async callable's
                # answer would come too late, and a coroutine is truthy whatever it would return.
                close = getattr(answer, "close", None)
                if callable(close):
                    close()
                self._warn("returned an awaitable, not a bool")
                return False
            return bool(answer)
        except Exception as exc:
            self._warn(f"raised {type(exc).__name__}")
            return False

    def _warn(self, what: str) -> None:
        if not self.warned:
            self.warned = True
            _log.warning("evalkit: a callable in %s %s; it is read as not matching.", self.option, what)


class _Rules:
    __slots__ = ("routes", "urls", "functions", "trace_health_checks")

    def __init__(self, routes: tuple = (), urls: tuple = (), functions: tuple = (),
                 trace_health_checks: bool = False) -> None:
        self.routes = routes
        self.urls = urls
        self.functions = functions
        self.trace_health_checks = trace_health_checks


_rules = _Rules()


def _bare_path(path: str) -> str:
    origin = _ORIGIN.match(path)
    if origin:
        path = path[origin.end():]
    for mark in ("?", "#"):
        cut = path.find(mark)
        if cut >= 0:
            path = path[:cut]
    return path


def _normalized(bare: str) -> str:
    """A path as patterns are matched against it: no repeated or trailing slash; the root is empty."""
    if "//" in bare:
        bare = _SLASHES.sub("/", bare)
    return bare.rstrip("/")


def _glob(text: str, within: str) -> str:
    if "*" not in text:
        return re.escape(text)
    return within.join(re.escape(piece) for piece in text.split("*"))


def _path_regex(path: str) -> str:
    """A path pattern as an anchored regex over a normalized path. ``*`` and a parameter are one segment,
    ``**`` any number of them, and ``*`` inside a segment any run of its characters."""
    out = []
    for segment in _SLASHES.sub("/", _bare_path(path)).split("/"):
        if not segment:
            continue
        if segment == "**":
            out.append("(?:/[^/]+)*")
        elif segment == "*" or _PARAM.match(segment):
            out.append("/[^/]+")
        else:
            out.append("/" + _glob(segment, "[^/]*"))
    return "".join(out)


def _dotted_regex(text: str, what: str) -> str:
    """A dotted pattern (a host's labels, a function's qualified name) as a regex over "." + the name."""
    out = []
    for part in text.split("."):
        if not part:
            raise _Invalid(f"has an empty {what}")
        if part == "**":
            out.append(r"(?:\.[^.]+)*")
        elif part == "*":
            out.append(r"\.[^.]+")
        else:
            out.append(r"\." + _glob(part, "[^.]*"))
    return "".join(out)


def _compile_route(text: str) -> Tuple[str, Optional[str], Any]:
    pattern = text.strip()
    if not pattern:
        raise _Invalid("is empty")
    method: Optional[str] = None
    if any(c.isspace() for c in pattern):
        found = _METHOD.match(pattern)
        if not found:
            raise _Invalid("must be a path, optionally after an HTTP method: \"GET /status\"")
        method = None if found.group(1) == "*" else found.group(1).upper()
        pattern = found.group(2).strip()
        if any(c.isspace() for c in pattern):
            raise _Invalid("has whitespace in its path")
    if pattern.startswith("*"):
        pattern = "/" + pattern
    if not pattern.startswith("/"):
        raise _Invalid("must start with /")
    return ("path", method, re.compile("^" + _path_regex(pattern) + "$"))


def _compile_url(text: str) -> Tuple[str, Any, Any]:
    pattern = text.strip()
    if not pattern:
        raise _Invalid("is empty")
    if any(c.isspace() for c in pattern):
        raise _Invalid("has whitespace in it")
    scheme: Optional[str] = None
    rest = pattern
    found = _SCHEME.match(pattern)
    if found:
        scheme = None if found.group(1) == "*" else found.group(1).lower()
        rest = found.group(2)
        if not rest or rest[0] in "/?#":
            raise _Invalid("has no host")
    elif "://" in pattern:
        raise _Invalid("has a malformed scheme")
    rest = rest.split("#", 1)[0].split("?", 1)[0]
    host_regex = None
    port: Optional[int] = None
    path: Optional[str] = rest if rest.startswith("/") else None
    if path is None:
        authority, slash, tail = rest.partition("/")
        path = "/" + tail if slash else None
        authority = authority.rsplit("@", 1)[-1]
        host, port_text = authority, ""
        if authority.startswith("["):
            close = authority.find("]")
            if close < 0:
                raise _Invalid("has an unclosed [ in its host")
            host, after = authority[:close + 1], authority[close + 1:]
            if after:
                if not after.startswith(":"):
                    raise _Invalid("has text after its host")
                port_text = after[1:]
        elif ":" in authority:
            host, _, port_text = authority.rpartition(":")
        if port_text and port_text != "*":
            if not port_text.isdigit() or not 0 < int(port_text) < 65536:
                raise _Invalid("has an invalid port")
            port = int(port_text)
        host = host.lower().rstrip(".")
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]
        if not host:
            raise _Invalid("has no host")
        if host not in ("*", "**"):
            host_regex = re.compile(_dotted_regex(host, "host label") + "$")
    path_regex = re.compile("^" + _path_regex(path) + "$") if path is not None else None
    return ("url", (scheme, host_regex, port), path_regex)


def _compile_function(text: str) -> Tuple[str, None, Any]:
    pattern = text.strip()
    if not pattern:
        raise _Invalid("is empty")
    if any(c.isspace() for c in pattern):
        raise _Invalid("has whitespace in it")
    return ("name", None, re.compile(_dotted_regex(pattern, "name part") + "$"))


def _shown(pattern: Any, option: str) -> str:
    try:
        text = pattern if isinstance(pattern, str) else repr(pattern)
        if option.endswith("urls") or option == URLS_ENV:
            text = scrub_url(text)
    except Exception:
        text = f"<{type(pattern).__name__}>"
    return repr(text if len(text) <= _SHOWN_MAX else text[:_SHOWN_MAX] + "...")


def _as_items(value: Any, option: str) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, _PatternType)) or callable(value):
        return [value]
    try:
        return list(value)
    except Exception:
        _log.warning("evalkit.init(): %s must be a list of patterns (got %s); it is ignored.",
                     option, type(value).__name__)
        return []


def _env_items(name: str) -> List[str]:
    raw = os.environ.get(name) or ""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _compile_all(value: Any, option: str, env: str, compile_text: Callable[[str], tuple]) -> tuple:
    compiled = []
    sources = [(item, option) for item in _as_items(value, option)]
    sources += [(item, env) for item in _env_items(env)]
    for item, source in sources:
        try:
            if isinstance(item, bytes):
                item = item.decode("utf-8")
            if isinstance(item, str):
                compiled.append(compile_text(item))
            elif isinstance(item, _PatternType):
                compiled.append(("regex", None, item))
            elif callable(item):
                compiled.append(("call", None, _Callable(item, option)))
            else:
                raise _Invalid("is not a string, a compiled regex or a callable")
        except Exception as exc:
            reason = str(exc) if isinstance(exc, _Invalid) else f"{type(exc).__name__}: {exc}"
            _log.warning("evalkit.init(): %s pattern %s is skipped: it %s.", source, _shown(item, source), reason)
    return tuple(compiled)


def configure(
    ignore_routes: Any = None,
    ignore_urls: Any = None,
    ignore_functions: Any = None,
    trace_health_checks: Any = False,
) -> None:
    """Compile the options and the EVALKIT_IGNORE_* variables, merged, as the rules in force."""
    global _rules
    _rules = _Rules(
        routes=_compile_all(ignore_routes, "ignore_routes", ROUTES_ENV, _compile_route),
        urls=_compile_all(ignore_urls, "ignore_urls", URLS_ENV, _compile_url),
        functions=_compile_all(ignore_functions, "ignore_functions", FUNCTIONS_ENV, _compile_function),
        trace_health_checks=bool(trace_health_checks),
    )


def reset() -> None:
    global _rules
    _rules = _Rules()


def traces_health_checks() -> bool:
    return _rules.trace_health_checks


def has_url_patterns() -> bool:
    return bool(_rules.urls)


def ignored_route(
    method: Any,
    path: Any,
    route: Optional[str] = None,
    resolve_route: Optional[Callable[[], Optional[str]]] = None,
) -> bool:
    """Whether an incoming request matches ``ignore_routes``.

    A path pattern matches the request's path, or the framework's route template when it is known; a
    regex is searched in the path; a callable decides from a RouteInfo.
    """
    patterns = _rules.routes
    if not patterns or not isinstance(path, str):
        return False
    verb = method.upper() if isinstance(method, str) and method else ""
    bare = _bare_path(path)
    normalized = _normalized(bare)
    known = _normalized(_bare_path(route)) if isinstance(route, str) and route else None
    info: Optional[RouteInfo] = None
    for kind, want, test in patterns:
        if kind == "path":
            if want is not None and want != verb:
                continue
            if test.match(normalized) or (known is not None and test.match(known)):
                return True
        elif kind == "regex":
            if test.search(bare) or (normalized != bare and test.search(normalized)):
                return True
        else:
            if info is None:
                info = RouteInfo(verb, bare, route if known is not None else None, resolve_route)
            if test(info):
                return True
    return False


def _url_parts(url: str) -> Optional[Tuple[str, str, Optional[int], str]]:
    try:
        parts = urlsplit(url)
        scheme = (parts.scheme or "").lower()
        host = (parts.hostname or "").lower()
        try:
            port = parts.port
        except ValueError:
            port = None
        return scheme, host, port or _DEFAULT_PORTS.get(scheme), parts.path or ""
    except Exception:
        return None


def ignored_url(url: Any, method: Any = None) -> bool:
    """Whether an outgoing call matches ``ignore_urls``: a URL pattern by scheme, host, port and path, a
    regex searched in the full URL, or a callable given a UrlInfo."""
    patterns = _rules.urls
    if not patterns:
        return False
    try:
        text = url.decode("utf-8", "replace") if isinstance(url, (bytes, bytearray)) else str(url)
    except Exception:
        return False
    parts = None
    for kind, want, test in patterns:
        if kind == "regex":
            if test.search(text):
                return True
            continue
        if parts is None:
            parts = _url_parts(text) or ("", "", None, _bare_path(text))
        scheme, host, port, path = parts
        if kind == "url":
            want_scheme, host_regex, want_port = want
            if want_scheme is not None and want_scheme != scheme:
                continue
            if host_regex is not None and not host_regex.match("." + host):
                continue
            if want_port is not None and want_port != port:
                continue
            if test is not None and not test.match(_normalized(path)):
                continue
            return True
        verb = method.upper() if isinstance(method, str) and method else "GET"
        if test(UrlInfo(verb, text, host, path)):
            return True
    return False


def qualified_name(fn: Any) -> str:
    """``module.qualname`` of a function, the name ``ignore_functions`` is matched against."""
    module = getattr(fn, "__module__", None) or ""
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or ""
    return f"{module}.{name}" if module and name else (module or name)


def ignored_function(name: Any) -> bool:
    """Whether a qualified function name matches ``ignore_functions``."""
    patterns = _rules.functions
    if not patterns or not isinstance(name, str) or not name:
        return False
    dotted = "." + name
    for kind, _want, test in patterns:
        if kind == "name":
            if test.match(dotted):
                return True
        elif kind == "regex":
            if test.search(name):
                return True
        elif test(name):
            return True
    return False


def ignored_callable(fn: Any) -> bool:
    if not _rules.functions:
        return False
    try:
        return ignored_function(qualified_name(fn))
    except Exception:
        return False
