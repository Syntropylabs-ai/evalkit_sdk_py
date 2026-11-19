"""The voice test run endpoints (RUN_API sections 3.1 and 4), for the SDK run and the hosted runner.

Both runners report through the same endpoints and differ only in how they authenticate: the SDK
with the subscription key (``X-Subscription-Key``), the hosted runner with a token scoped to one run
(``X-Voice-Run-Token``). The credential is held privately and sent only to this API: never to the
signed storage URLs, never in a message or a log line.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import quote

from .net import HttpError, RetryPolicy, json_request, request, with_retries

log = logging.getLogger("evalkit.voicetest.api")

DEFAULT_API_URL = "https://api.syntropylabs.ai"
PREFIX = "/v1/voice-tests"


class VoiceTestApiError(RuntimeError):
    """A call to the API failed for good. `status` is the HTTP status, None when nothing answered;
    `server_message` is what the API said, which names what to fix."""

    def __init__(self, message: str, status: Optional[int] = None,
                 server_message: Optional[str] = None, retry_after_s: Optional[float] = None) -> None:
        super().__init__(message)
        self.status = status
        self.server_message = server_message
        # The server's Retry-After, for a caller that schedules its own retry.
        self.retry_after_s = retry_after_s


class VoiceTestApi:
    """`base_url` is the API's base (``https://api.syntropylabs.ai``, or a hosted runner's
    ``callbackBase``); `prefix` is where the voice test routes sit under it."""

    def __init__(
        self,
        base_url: str,
        auth_headers: Mapping[str, str],
        *,
        prefix: str = PREFIX,
        retry: Optional[RetryPolicy] = None,
        timeout_s: float = 30.0,
        create_timeout_s: float = 120.0,
        upload_timeout_s: float = 300.0,
    ) -> None:
        if not auth_headers:
            raise ValueError("the voice test API needs a credential header")
        self.base_url = base_url.rstrip("/")
        self.prefix = "/" + prefix.strip("/") if prefix.strip("/") else ""
        self._auth = dict(auth_headers)
        # Other credentials a request body carries (a provider key for the caller's voice): kept out of
        # every message, like the credential header.
        self._secrets: list = []
        self.retry = retry or RetryPolicy()
        self.timeout_s = timeout_s
        # A create synthesizes the caller's new lines before it answers; a repeat of it waits for that.
        self.create_timeout_s = create_timeout_s
        self.upload_timeout_s = upload_timeout_s

    @classmethod
    def with_subscription_key(cls, base_url: str, key: str, **kwargs: Any) -> "VoiceTestApi":
        if not key or not key.strip():
            raise ValueError("a subscription key is required")
        return cls(base_url, {"X-Subscription-Key": key.strip()}, **kwargs)

    @classmethod
    def with_run_token(cls, base_url: str, token: str, **kwargs: Any) -> "VoiceTestApi":
        if not token or not token.strip():
            raise ValueError("a run token is required")
        return cls(base_url, {"X-Voice-Run-Token": token.strip()}, **kwargs)

    def __repr__(self) -> str:
        return f"VoiceTestApi({self.base_url + self.prefix!r}, auth={sorted(self._auth)})"

    def _url(self, *parts: Any) -> str:
        return self.base_url + self.prefix + "".join("/" + quote(str(p), safe="") for p in parts)

    def redact(self, value: Optional[str]) -> None:
        """Keeps `value`, a credential some request body carries, out of every message from here on."""
        if value and value.strip() and value not in self._secrets:
            self._secrets.append(value)

    def _scrub(self, text: str) -> str:
        for value in [*self._auth.values(), *self._secrets]:
            if value:
                text = text.replace(value, "***")
        return text

    def _call(self, method: str, url: str, what: str, body: Optional[Dict[str, Any]] = None,
              idempotent: bool = True, headers: Optional[Mapping[str, str]] = None,
              timeout_s: Optional[float] = None) -> Dict[str, Any]:
        sent = {**self._auth, **(headers or {})}
        timeout = self.timeout_s if timeout_s is None else timeout_s

        def send() -> Dict[str, Any]:
            try:
                return json_request(method, url, headers=sent, body=body, timeout=timeout)
            except HttpError as exc:
                # Scrubbed before the retry loop logs it, not only when it gives up.
                raise HttpError(exc.method, exc.url, exc.status, self._scrub(exc.message),
                                retry_after_s=exc.retry_after_s, unsent=exc.unsent) from None

        try:
            return with_retries(send, self.retry, what, idempotent=idempotent)
        except HttpError as exc:
            raise VoiceTestApiError(self._scrub(f"{what} failed: {exc}"), exc.status,
                                    self._scrub(exc.message) if exc.status else None,
                                    exc.retry_after_s) from None

    # RUN_API 3.1 -------------------------------------------------------------------------------

    def create_run(self, body: Dict[str, Any], idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        """``POST /v1/voice-tests``: the plan. Every attempt carries the same ``Idempotency-Key`` (a
        fresh one per run unless given), and the API answers a repeat of it with the run it already
        made, so a create whose answer was lost is sent again without opening a second run."""
        key = idempotency_key or f"evalkit-{uuid.uuid4().hex}"
        return self._call("POST", self._url(), "creating the voice test run", body,
                          headers={"Idempotency-Key": key}, timeout_s=self.create_timeout_s)

    # RUN_API 4 ---------------------------------------------------------------------------------

    def recording_upload(self, run_id: str, call_index: int) -> Dict[str, Any]:
        """``POST .../calls/:callIndex/recording``: a signed upload for the call's recording."""
        return self._call("POST", self._url(run_id, "calls", call_index, "recording"),
                          f"asking where to upload call {call_index}'s recording")

    def upload_recording(self, upload: Mapping[str, Any], path: str) -> Tuple[int, str]:
        """PUTs the file to the signed URL with exactly the headers the API gave, and returns its
        byte count and sha256. Our credential is not among them: the URL authorises itself."""
        url = upload.get("uploadUrl")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise VoiceTestApiError("the recording upload has no uploadUrl")
        method = str(upload.get("method") or "PUT").upper()
        headers = {str(k): str(v) for k, v in (upload.get("headers") or {}).items()}
        size = os.path.getsize(path)
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        headers.setdefault("Content-Type", "audio/wav")
        headers["Content-Length"] = str(size)

        def put() -> None:
            with open(path, "rb") as fh:
                request(method, url, headers=headers, body=fh, timeout=self.upload_timeout_s)

        try:
            with_retries(put, self.retry, "uploading the recording")
        except HttpError as exc:
            raise VoiceTestApiError(f"uploading the recording failed: {exc}", exc.status) from None
        return size, digest.hexdigest()

    def post_result(self, run_id: str, call_index: int, body: Dict[str, Any]) -> Dict[str, Any]:
        """``POST .../calls/:callIndex/result``. A repeat of the same body changes nothing on the
        server, so it is retried like a read."""
        return self._call("POST", self._url(run_id, "calls", call_index, "result"),
                          f"sending call {call_index}'s result", body)

    def complete(self, run_id: str) -> Dict[str, Any]:
        """``POST .../complete``: the run summary."""
        return self._call("POST", self._url(run_id, "complete"), "completing the run", {})

    def get_run(self, run_id: str) -> Dict[str, Any]:
        """``GET /v1/voice-tests/:runId``: status, per-call summaries and evaluator results, ``passed``."""
        return self._call("GET", self._url(run_id), "reading the run")

    # RUN_API 4.1 -------------------------------------------------------------------------------

    def next_turn(self, run_id: str, call_index: int, turn: int, retake: int, bot_audio: Mapping[str, Any],
                  keys: Optional[Mapping[str, Optional[str]]] = None,
                  timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """``POST .../calls/:callIndex/turns``: a dynamic call's next caller line (RUN_API section 4.1).

        One attempt only: the caller retries by itself, sending silence meanwhile, while its call's
        budget lasts, and a failure raises VoiceTestApiError with the status and the Retry-After.
        `bot_audio` is ``{"data": <base64 WAV>, "durationMs"}``. `keys` are the request keys the run was
        created with (``caller``, ``voice``), sent again because the API keeps neither; each is kept out
        of every message from here on."""
        body: Dict[str, Any] = {"turn": int(turn), "retake": int(retake), "botAudio": dict(bot_audio)}
        sent = {name: value.strip() for name, value in (keys or {}).items()
                if isinstance(value, str) and value.strip()}
        for value in sent.values():
            self.redact(value)
        if sent:
            body["keys"] = sent
        url = self._url(run_id, "calls", call_index, "turns")
        what = f"asking for call {call_index}'s turn {turn}"
        try:
            return json_request("POST", url, headers=dict(self._auth), body=body,
                                timeout=self.timeout_s if timeout_s is None else timeout_s)
        except HttpError as exc:
            message = self._scrub(exc.message)
            raise VoiceTestApiError(self._scrub(f"{what} failed: {exc}"), exc.status,
                                    message if exc.status else None, exc.retry_after_s) from None
