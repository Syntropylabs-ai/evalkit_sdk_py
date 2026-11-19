"""urlopen for every request the SDK and its CLI send with a credential.

urllib follows a 30x with the request's headers, so a redirect from a misconfigured proxy or a
compromised host would hand the subscription key (X-Subscription-Key, or Authorization on OTLP) to
whatever server it names. A request that carries a credential follows a redirect only within its own
origin, or from http to https on the same host; one to anywhere else raises HTTPError with the 30x
status and a body saying it was refused, which each caller reports as it reports any refusal. A
request with no credential (a signed storage URL, which authorises itself) follows redirects as
before.

Standard library only, so the coding-agent hooks, which import it on every event, never load the
tracing runtime; the SDK's own HTTP paths use the same opener.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from typing import Any, Iterable, Optional, Tuple
from urllib.parse import urlsplit

CREDENTIAL_HEADERS = frozenset({
    "authorization", "proxy-authorization", "cookie", "x-subscription-key", "x-voice-run-token", "x-api-key",
})
_DEFAULT_PORTS = {"http": 80, "https": 443}


def _origin(url: str) -> Optional[Tuple[str, str, Optional[int]]]:
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    return scheme, (parts.hostname or "").lower(), port if port is not None else _DEFAULT_PORTS.get(scheme)


def same_origin(source: str, target: str) -> bool:
    """Whether a redirect from `source` to `target` stays with the same server: the same scheme, host
    and port, or http upgraded to https on the same host at the default ports."""
    a, b = _origin(source), _origin(target)
    if a is None or b is None:
        return False
    if a == b:
        return True
    return a[0] == "http" and b[0] == "https" and a[1] == b[1] and a[2] == 80 and b[2] == 443


def carries_credential(header_names: Iterable[str]) -> bool:
    return any(str(name).lower() in CREDENTIAL_HEADERS for name in header_names)


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any,
                         newurl: str) -> Optional[urllib.request.Request]:
        if not same_origin(req.full_url, newurl) and carries_credential(name for name, _ in req.header_items()):
            where = urlsplit(newurl)
            message = (f"refused a redirect to {where.scheme}://{where.netloc.rpartition('@')[2]}: the request "
                       "carries a credential, which goes only to the server it was sent to")
            try:
                fp.close()
            except Exception:
                pass
            raise urllib.error.HTTPError(req.full_url, code, message, headers, io.BytesIO(message.encode("utf-8")))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_SameOriginRedirectHandler)


def urlopen(req: urllib.request.Request, timeout: float) -> Any:
    """urllib.request.urlopen, with the redirect rule above."""
    return _opener.open(req, timeout=timeout)
