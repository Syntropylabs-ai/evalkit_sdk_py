"""A local stand-in for the SyntropyLabs voice test API (RUN_API sections 3 and 4), plus the storage the
signed URLs point at and the trace ingest, so the runner can be tested end to end with no backend.

Every request is recorded with its headers, so a test can check what reached which endpoint.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

KEY = "tk_live_voicetest_fake_0123456789"


def synth_clip(ms: int, seed: int, encoding: str = "pcm16", rate: int = 16000) -> bytes:
    """A caller line as the API would serve it: raw PCM16 at `rate`, or a WAV file."""
    from evalkit.voicetest.audio import SynthSpec, synth_speech

    pcm = synth_speech(SynthSpec.from_json({"ms": ms, "seed": seed}), rate)
    # Silence around the speech, as text-to-speech leaves it: the caller trims it off.
    pad = [0] * (rate // 10)
    import numpy as np

    pcm = np.concatenate([np.array(pad, dtype=np.int16), pcm, np.array(pad, dtype=np.int16)])
    raw = pcm.astype("<i2").tobytes()
    if encoding == "wav":
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(raw)
        return buf.getvalue()
    return raw


def default_scenarios() -> List[Dict[str, Any]]:
    return [
        {"id": "sc-refund", "name": "refund for a duplicate charge", "lines": [
            {"text": "Hi, I was charged twice for my order.", "ms": 900, "seed": 1,
             "wait": {"rule": "fixed", "ms": 300}},
            {"text": "It was order 4471.", "ms": 700, "seed": 2},
        ]},
        {"id": "sc-status", "name": "order status", "lines": [
            {"text": "Where is my order?", "ms": 800, "seed": 3, "wait": {"rule": "fixed", "ms": 300}},
            {"text": "Thanks.", "ms": 500, "seed": 4},
        ]},
    ]


class FakeVoiceApi:
    def __init__(self, scenarios: Optional[List[Dict[str, Any]]] = None, *, encoding: str = "pcm16",
                 max_concurrent: int = 2, max_call_ms: int = 60000,
                 passed: Any = True, create_status: Optional[int] = None,
                 fail_once: Optional[Dict[str, int]] = None, statuses: Tuple[str, ...] = ("scoring", "completed"),
                 corrupt: Tuple[str, ...] = (), lose_create_answer: bool = False) -> None:
        self.scenarios = scenarios if scenarios is not None else default_scenarios()
        self.encoding = encoding
        self.max_concurrent = max_concurrent
        self.max_call_ms = max_call_ms
        self.passed = passed
        self.create_status = create_status
        self.fail_once = dict(fail_once or {})
        self.statuses = list(statuses)
        self.corrupt = set(corrupt)  # line texts whose audio is served with the wrong bytes
        # The first create is made but its answer lost on the way back (a 502 from a proxy).
        self.lose_create_answer = lose_create_answer
        self.by_idempotency_key: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.requests: List[Dict[str, Any]] = []
        self.created: List[Dict[str, Any]] = []
        self.plan: Optional[Dict[str, Any]] = None
        self.downloads: Dict[str, int] = {}
        self.upload_urls: List[Tuple[str, int]] = []
        self.uploads: Dict[Tuple[str, int], Tuple[bytes, Dict[str, str]]] = {}
        self.results: Dict[int, Dict[str, Any]] = {}
        self.result_posts = 0
        self.completed = 0
        self.polls = 0
        self.spans: List[Dict[str, Any]] = []
        self.failed: Dict[str, int] = {}
        self.audio: Dict[str, bytes] = {}
        self.server: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    # lifecycle ---------------------------------------------------------------------------------

    def __enter__(self) -> "FakeVoiceApi":
        api = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # quiet
                pass

            def _body(self) -> bytes:
                n = int(self.headers.get("Content-Length") or 0)
                data = self.rfile.read(n) if n else b""
                if self.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                return data

            def _send(self, status: int, body: Any = None, raw: Optional[bytes] = None, headers=None):
                data = raw if raw is not None else (b"" if body is None else json.dumps(body).encode())
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Type", "application/json" if raw is None else "application/octet-stream")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def _handle(self, method: str) -> None:
                body = self._body()
                path, _, query = self.path.partition("?")
                with api.lock:
                    api.requests.append({"method": method, "path": path, "query": query,
                                         "headers": {k.lower(): v for k, v in self.headers.items()},
                                         "body": body})
                    status = None
                    for pattern, code in list(api.fail_once.items()):
                        if re.fullmatch(pattern, f"{method} {path}") and not api.failed.get(pattern):
                            api.failed[pattern] = 1
                            status = code
                if status is not None:
                    self._send(status, {"message": "try again"})
                    return
                status, out, raw, headers = api.route(method, path, query, dict(self.headers), body)
                self._send(status, out, raw, headers)

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def do_PUT(self):
                self._handle("PUT")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()

    @property
    def url(self) -> str:
        assert self.server is not None
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    # what the tests read -----------------------------------------------------------------------

    def paths(self, prefix: str = "") -> List[str]:
        return [f"{r['method']} {r['path']}" for r in self.requests if r["path"].startswith(prefix)]

    def requests_to(self, pattern: str) -> List[Dict[str, Any]]:
        return [r for r in self.requests if re.fullmatch(pattern, f"{r['method']} {r['path']}")]

    # the API -----------------------------------------------------------------------------------

    def _plan(self, body: Dict[str, Any]) -> Dict[str, Any]:
        run_id = "run%03d" % (len(self.created))
        chosen = [s for s in self.scenarios if not body.get("scenarioIds") or s["id"] in body["scenarioIds"]]
        calls = []
        index = 0
        for repeat in range(int(body.get("repeats") or 1)):
            for sc in chosen:
                lines = []
                for line in sc["lines"]:
                    clip = synth_clip(line["ms"], line["seed"], self.encoding)
                    sha = hashlib.sha256(clip).hexdigest()
                    bad = clip[:4000] + bytes([clip[4000] ^ 0xFF]) + clip[4001:]
                    self.audio[sha] = bad if line["text"] in self.corrupt else clip
                    entry = {
                        "text": line["text"],
                        "audio": {"url": f"{self.url}/storage/audio/{sha}?X-Goog-Signature=s3cr3tsig",
                                  "encoding": "pcm16", "sampleRate": 16000, "sha256": sha,
                                  "durationMs": line["ms"] + 200},
                        "wait": line.get("wait") or {"rule": "bot_done", "silenceMs": 600, "maxMs": 6000},
                        "interruptAfterBotMs": line.get("interruptAfterBotMs"),
                        "futureField": {"ignored": True},
                    }
                    lines.append(entry)
                calls.append({
                    "callIndex": index, "scenarioId": sc["id"], "scenarioName": sc["name"], "repeat": repeat,
                    "sessionToken": f"{body.get('agentId') or 'sdk'}:{run_id}:{sc['id']}:{index}",
                    "script": {"maxCallMs": sc.get("maxCallMs", 300000), "lines": lines,
                               "end": {"afterLastLine": "bot_done", "silenceMs": 600, "maxMs": 5000}},
                })
                index += 1
        return {"runId": run_id, "consoleUrl": f"https://console.example/voice/{run_id}",
                "limits": {"maxCallMs": self.max_call_ms, "maxConcurrentCalls": self.max_concurrent},
                "calls": calls}

    def _summary(self, run_id: str, status: str) -> Dict[str, Any]:
        passed = self.passed(self.results) if callable(self.passed) else self.passed
        calls = []
        for index, result in sorted(self.results.items()):
            calls.append({"callIndex": index, "error": result.get("error"), "evaluators": [
                {"ruleId": "rule-1", "key": "voice_response_delay", "name": "voice_response_delay",
                 "score": 1.0 if passed else 0.5, "passed": bool(passed),
                 "figures": {"p95Ms": 840, "p50Ms": 610, "limitMs": 1200, "turnsOver": 0 if passed else 1,
                             "turns": 2, "unanswered": 0, "unscoredTurns": []}}]})
        out = {"runId": run_id, "status": status, "calls": calls}
        if status not in ("scoring", "running"):
            out["passed"] = passed
        return out

    def route(self, method: str, path: str, query: str, headers: Dict[str, str],
              body: bytes) -> Tuple[int, Any, Optional[bytes], Optional[Dict[str, str]]]:
        lower = {k.lower(): v for k, v in headers.items()}
        if path.startswith("/storage/"):
            if "x-subscription-key" in lower:
                return 400, {"message": "storage never takes the subscription key"}, None, None
            if method == "GET" and path.startswith("/storage/audio/"):
                sha = path.rsplit("/", 1)[-1]
                with self.lock:
                    self.downloads[sha] = self.downloads.get(sha, 0) + 1
                data = self.audio.get(sha)
                return (200, None, data, None) if data is not None else (404, {"message": "no such object"}, None, None)
            if method == "PUT" and path.startswith("/storage/upload/"):
                _, _, _, run_id, index = path.split("/")
                with self.lock:
                    self.uploads[(run_id, int(index))] = (body, lower)
                return 200, None, b"", None
            return 404, {"message": "not found"}, None, None
        if path == "/v1/ingest" and method == "POST":
            events = json.loads(body).get("events") or []
            with self.lock:
                self.spans.extend(events)
            return 200, {"accepted": len(events)}, None, None
        if lower.get("x-subscription-key") != KEY:
            return 401, {"message": "Invalid subscription key"}, None, None
        if path == "/v1/voice-tests" and method == "POST":
            if self.create_status:
                return self.create_status, {"message": "scenario set sc-missing not found"}, None, None
            req = json.loads(body)
            key = lower.get("idempotency-key")
            with self.lock:
                if key and key in self.by_idempotency_key:  # RUN_API 3.1: the first run, not a second
                    return 201, self.by_idempotency_key[key], None, None
                self.plan = self._plan(req)
                self.created.append(req)
                if key:
                    self.by_idempotency_key[key] = self.plan
                if self.lose_create_answer and len(self.created) == 1:
                    return 502, {"message": "bad gateway"}, None, None
            return 201, self.plan, None, None
        m = re.fullmatch(r"/v1/voice-tests/([^/]+)/calls/(\d+)/(recording|result)", path)
        if m and method == "POST":
            run_id, index, what = m.group(1), int(m.group(2)), m.group(3)
            if what == "recording":
                with self.lock:
                    self.upload_urls.append((run_id, index))
                return 200, {"uploadUrl": f"{self.url}/storage/upload/{run_id}/{index}?X-Goog-Signature=upsig",
                             "method": "PUT", "headers": {"Content-Type": "audio/wav", "x-goog-meta-run": run_id},
                             "objectPath": f"traces/tenant1/audio/{run_id}/{index}.wav",
                             "expiresAt": "2026-10-31T23:59:00Z"}, None, None
            with self.lock:
                self.results[index] = json.loads(body)
                self.result_posts += 1
            return 200, {"ok": True}, None, None
        m = re.fullmatch(r"/v1/voice-tests/([^/]+)/complete", path)
        if m and method == "POST":
            with self.lock:
                self.completed += 1
            return 200, self._summary(m.group(1), "scoring"), None, None
        m = re.fullmatch(r"/v1/voice-tests/([^/]+)", path)
        if m and method == "GET":
            with self.lock:
                self.polls += 1
                status = self.statuses[min(self.polls - 1, len(self.statuses) - 1)]
            return 200, self._summary(m.group(1), status), None, None
        return 404, {"message": f"no route {method} {path}"}, None, None


def run_in_thread(fn: Callable[[], Any]) -> threading.Thread:
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    return t
