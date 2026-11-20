"""A local stand-in for the SyntropyLabs voice test API (RUN_API sections 3 and 4), plus the storage the
signed URLs point at and the trace ingest, so the runner can be tested end to end with no backend.

Every request is recorded with its headers, so a test can check what reached which endpoint. Scenarios
with a goal, and callers written from ``generate``, are planned as dynamic calls, and the turn endpoint
(RUN_API section 4.1) answers them from a script of fake answers: it hears nothing, so ``heard`` names the
turn, and each line is synthetic speech.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import re
import threading
import time
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


# RUN_API section 3.3: the wait a line gets when it names none, line 0's by the target's firstSpeaker.
DEFAULT_LINE_WAIT = {"rule": "bot_done", "silenceMs": 700, "maxMs": 15000}
FIRST_LINE_WAIT = {
    "auto": {"rule": "bot_done", "startWithinMs": 3000, "silenceMs": 700, "maxMs": 15000},
    "bot": DEFAULT_LINE_WAIT,
    "caller": {"rule": "fixed", "ms": 0},
}
NO_VOICE_KEY = ("No key to synthesize the caller's lines with: this organisation has no Text-to-Speech "
                "model. Either add a provider connection and a Text-to-Speech model under Models in the "
                "console, or pass voice {provider, model, voice, apiKey} from the SDK "
                "(evalkit voice-test run --tts-provider openai with OPENAI_API_KEY set).")


# The wait after each bot answer before the next line is asked for, shorter than the API's so the tests run fast.
TURN_WAIT = {"rule": "bot_done", "silenceMs": 600, "maxMs": 6000}
CUT_IN_TEXT = "Sorry, quick question."


def _clip_for(text: str) -> Dict[str, Any]:
    return {"text": text, "ms": min(1200, 300 + 12 * len(text)), "seed": 100 + sum(text.encode()) % 800}


def _dynamic(sc: Dict[str, Any], first: Dict[str, Any], default_max: int) -> Dict[str, Any]:
    """A goal scenario as the fake plans it: the opening as line 0 and, for an interruption before turn 1,
    the cut-in as line 1 (CONTRACT section 4.1), and the loop's rules."""
    opening = sc.get("opening") or f"Hi, I need help: {sc['goal'][:60]}"
    lines = [dict(_clip_for(opening), wait=dict(first), interruptAfterBotMs=None)]
    interrupt = sc.get("interrupt")
    if interrupt and interrupt["turn"] == 1:
        lines.append(dict(_clip_for(CUT_IN_TEXT), wait=dict(DEFAULT_LINE_WAIT),
                          interruptAfterBotMs=interrupt["afterBotMs"]))
    turns = {"max": sc.get("maxTurns") or default_max, "wait": dict(TURN_WAIT)}
    if interrupt:
        turns["interrupt"] = dict(interrupt)
    return {"lines": lines, "mode": "dynamic", "turns": turns, "goal": sc["goal"]}


def inline_scenarios(run_id: str, body: Dict[str, Any], default_max: int = 2) -> List[Dict[str, Any]]:
    """The request's inline scenarios as the fake plans them: a clip per text, the waits as the API
    fills them in, and the scenario ids the API derives from the run and the index."""
    first = FIRST_LINE_WAIT[(body.get("target") or {}).get("firstSpeaker") or "auto"]
    out = []
    for i, sc in enumerate(body["scenarios"]):
        if sc.get("goal"):
            out.append(dict(_dynamic(sc, first, default_max), id=f"scn_{run_id}_{i}", name=sc["name"]))
            continue
        lines = []
        for j, line in enumerate(sc["lines"]):
            text = line["text"]
            interrupt = line.get("interruptAfterBotMs")
            wait = dict(line["wait"]) if line.get("wait") else None
            if wait and wait["rule"] == "bot_done":
                wait = {"silenceMs": 700, "maxMs": 15000, **wait}
            if wait is None:
                wait = dict(DEFAULT_LINE_WAIT if (j or interrupt is not None) else first)
            lines.append({"text": text, "ms": min(1200, 300 + 12 * len(text)), "seed": 100 + sum(text.encode()) % 800,
                          "wait": wait, "interruptAfterBotMs": interrupt})
        out.append({"id": f"scn_{run_id}_{i}", "name": sc["name"], "lines": lines})
    return out


def generated_scenarios(run_id: str, generate: Dict[str, Any]) -> List[Dict[str, Any]]:
    """`generate` as the fake writes it: `count` goal scenarios about the bot, the first impatient."""
    out = []
    for i in range(int(generate.get("count") or 5)):
        sc: Dict[str, Any] = {"name": f"generated caller {i}", "goal": f"Caller {i}: {generate['about'][:80]}",
                              "persona": "IMPATIENT" if i == 0 else "NOVICE"}
        if generate.get("maxTurns"):
            sc["maxTurns"] = generate["maxTurns"]
        out.append(sc)
    return out


def wav_b64(text: str) -> Dict[str, Any]:
    """A line's audio as the turn endpoint sends it inline: a base64 WAV, 16 kHz mono."""
    clip = _clip_for(text)
    data = synth_clip(clip["ms"], clip["seed"], "wav")
    return {"data": base64.b64encode(data).decode(), "encoding": "wav", "sampleRate": 16000,
            "durationMs": clip["ms"] + 200, "sha256": hashlib.sha256(data).hexdigest()}


def default_answer(call: Dict[str, Any], turn: int, retake: int, request: Dict[str, Any]) -> Dict[str, Any]:
    """The fake caller model: a line per turn up to turns.max, and the planned cut-in with the answer before
    its turn. Returns {"line"?: text, "done"?: bool, "cutIn"?: {text, afterBotMs}}."""
    turns = call["script"]["turns"]
    out: Dict[str, Any] = {"line": f"Here is my reply number {turn}."}
    interrupt = turns.get("interrupt")
    if interrupt and interrupt["turn"] == turn + 1 and turn + 1 <= turns["max"]:
        out["cutIn"] = {"text": CUT_IN_TEXT, "afterBotMs": interrupt["afterBotMs"]}
    return out


class FakeVoiceApi:
    def __init__(self, scenarios: Optional[List[Dict[str, Any]]] = None, *, encoding: str = "pcm16",
                 max_concurrent: int = 2, max_call_ms: int = 60000,
                 passed: Any = True, create_status: Optional[int] = None,
                 fail_once: Optional[Dict[str, int]] = None, statuses: Tuple[str, ...] = ("scoring", "completed"),
                 corrupt: Tuple[str, ...] = (), lose_create_answer: bool = False,
                 require_voice_key: bool = False,
                 answer: Optional[Callable[..., Dict[str, Any]]] = None, think_s: float = 0.3,
                 default_max_turns: int = 2, call_max_ms: Optional[int] = None,
                 caller: Optional[Dict[str, Any]] = None, platform_minutes_left: Any = None,
                 heard_before_cut_in: bool = False) -> None:
        # The turn endpoint (RUN_API 4.1): `answer(call, turn, retake, request)` writes each answer (see
        # `default_answer`); it may also return {"status", "message", "retryAfter"} to fail the request,
        # and {"lose": True} to store the answer but lose it on the way back. `think_s` is the model's time.
        # `heard_before_cut_in`: the first answer after a cut-in (a request with `cutInAtMs`) also says
        # what the bot said before the cut-in, as `heardBeforeCutIn`.
        self.answer = answer or default_answer
        self.heard_before_cut_in = heard_before_cut_in
        self.think_s = think_s
        self.default_max_turns = default_max_turns
        self.call_max_ms = call_max_ms
        self.caller = caller
        self.platform_minutes_left = platform_minutes_left
        self.turn_requests: List[Dict[str, Any]] = []
        self.turn_answers: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self.next_turn: Dict[int, Optional[int]] = {}
        self.last_turn: Dict[int, Tuple[int, int]] = {}
        self.final_turn: Dict[int, int] = {}
        self.model_calls = 0
        self.scenarios = scenarios if scenarios is not None else default_scenarios()
        # Answer a create that brings no voice key as the API does for an org with no TTS model.
        self.require_voice_key = require_voice_key
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
        if body.get("generate"):
            chosen = inline_scenarios(run_id, dict(body, scenarios=generated_scenarios(run_id, body["generate"])),
                                      self.default_max_turns)
        elif body.get("scenarios"):
            chosen = inline_scenarios(run_id, body, self.default_max_turns)
        else:
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
                script = {"maxCallMs": self.call_max_ms or sc.get("maxCallMs", 300000), "lines": lines,
                          "end": {"afterLastLine": "bot_done", "silenceMs": 600, "maxMs": 5000}}
                if sc.get("mode") == "dynamic":
                    script.update(mode="dynamic", turns=sc["turns"])
                    self.next_turn[index] = len(lines)
                calls.append({
                    "callIndex": index, "scenarioId": sc["id"], "scenarioName": sc["name"], "repeat": repeat,
                    "sessionToken": f"{body.get('agentId') or 'sdk'}:{run_id}:{sc['id']}:{index}",
                    "script": script,
                })
                index += 1
        dynamic = any(c["script"].get("mode") == "dynamic" for c in calls)
        caller = self.caller
        if caller is None and dynamic:
            given = body.get("caller") or {}
            caller = {"provider": given.get("provider") or "google", "model": given.get("model") or "gemini-3.8-flash",
                      "keySource": "request" if given.get("apiKey") else "organization"}
        return {"runId": run_id, "consoleUrl": f"https://console.example/voice/{run_id}",
                "limits": {"maxCallMs": self.max_call_ms, "maxConcurrentCalls": self.max_concurrent},
                "caller": caller if dynamic else None, "platformMinutesLeft": self.platform_minutes_left,
                "calls": calls}

    def _turn(self, index: int, req: Dict[str, Any], headers: Dict[str, str]):
        """RUN_API 4.1: the next caller line of call `index`."""
        turn, retake = req.get("turn"), req.get("retake", 0)
        audio = base64.b64decode((req.get("botAudio") or {}).get("data") or "")
        with wave.open(io.BytesIO(audio), "rb") as w:
            fmt = (w.getnchannels(), w.getsampwidth(), w.getframerate())
            audio_ms = int(round(w.getnframes() * 1000 / w.getframerate()))
        with self.lock:
            self.turn_requests.append({"call": index, "turn": turn, "retake": retake, "keys": req.get("keys"),
                                       "format": fmt, "audioMs": audio_ms,
                                       "durationMs": (req.get("botAudio") or {}).get("durationMs"),
                                       "cutInAtMs": (req.get("botAudio") or {}).get("cutInAtMs"),
                                       "headers": headers, "at": time.monotonic()})
            cut = (req.get("botAudio") or {}).get("cutInAtMs")
            if cut is not None and not (isinstance(cut, int) and 1 <= cut < audio_ms):
                return 400, {"message": f"botAudio.cutInAtMs is {cut} ms, past the end of the "
                                        f"{audio_ms} ms of audio sent"}, None, None
            stored = self.turn_answers.get((index, turn, retake))
            if stored is not None:  # a repeat: the stored answer, the model is not asked again
                return 200, stored, None, None
            last = self.last_turn.get(index)
            again = last is not None and turn == last[0] and retake == last[1] + 1 == 1
            if not again and (turn != self.next_turn.get(index) or retake != 0):
                return 409, {"message": f"turn {turn} is not the next turn of call {index} "
                                        f"(expected {self.next_turn.get(index)})"}, None, None
            call = next(c for c in self.plan["calls"] if c["callIndex"] == index)
            final = turn > call["script"]["turns"]["max"] or turn == self.final_turn.get(index)
        spec = {} if final else self.answer(call, turn, retake, req)
        if spec.get("status"):
            hdrs = {"Retry-After": str(spec["retryAfter"])} if spec.get("retryAfter") is not None else None
            return spec["status"], {"message": spec.get("message", "try again")}, None, hdrs
        delay = spec.get("delay", self.think_s)
        time.sleep(delay)
        answer: Dict[str, Any] = {"turn": turn, "heard": f"bot answer before turn {turn} ({audio_ms} ms)",
                                  "done": bool(final or spec.get("done")), "line": None, "cutIn": None,
                                  "thinkMs": int(delay * 1000)}
        if self.heard_before_cut_in and cut is not None:
            answer["heardBeforeCutIn"] = f"bot answer cut in at {cut} ms"
        if not final and spec.get("line"):
            answer["line"] = {"text": spec["line"], "audio": wav_b64(spec["line"])}
        if not answer["done"] and spec.get("cutIn"):
            cut = spec["cutIn"]
            answer["cutIn"] = {"text": cut["text"], "audio": wav_b64(cut["text"]), "afterBotMs": cut["afterBotMs"]}
        with self.lock:
            self.model_calls += 1
            self.turn_answers[(index, turn, retake)] = answer
            self.last_turn[index] = (turn, retake)
            if final or (answer["done"] and answer["line"] is None):
                self.next_turn[index] = None
            elif answer["done"]:
                self.next_turn[index] = self.final_turn[index] = turn + 1
            else:
                self.next_turn[index] = turn + (2 if answer["cutIn"] else 1)
        if spec.get("lose"):
            return 502, {"message": "bad gateway"}, None, None
        return 200, answer, None, None

    def _summary(self, run_id: str, status: str) -> Dict[str, Any]:
        passed = self.passed(self.results) if callable(self.passed) else self.passed
        calls = []
        for index, result in sorted(self.results.items()):
            evaluators = [
                {"ruleId": "rule-1", "key": "voice_response_delay", "name": "voice_response_delay",
                 "score": 1.0 if passed else 0.5, "passed": bool(passed),
                 "figures": {"p95Ms": 840, "p50Ms": 610, "limitMs": 1200, "turnsOver": 0 if passed else 1,
                             "turns": 2, "unanswered": 0, "unscoredTurns": []}}]
            if ((result.get("scorecard") or {}).get("call") or {}).get("mode") == "dynamic":
                evaluators.append({"ruleId": "rule-2", "key": "voice_task_success", "name": "Voice Task Success",
                                   "score": 0.8, "passed": True,
                                   "reason": "Met: the agent refunds the second charge."})
            calls.append({"callIndex": index, "error": result.get("error"), "evaluators": evaluators})
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
            if sum(bool(req.get(k)) for k in ("scenarios", "scenarioSetId", "generate")) != 1:
                return 400, {"statusCode": 400, "error": "Bad Request",
                             "message": "Send one of generate, scenarioSetId or scenarios"}, None, None
            if self.require_voice_key and not (req.get("voice") or {}).get("apiKey"):
                return 400, {"statusCode": 400, "error": "Bad Request", "message": NO_VOICE_KEY}, None, None
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
        m = re.fullmatch(r"/v1/voice-tests/([^/]+)/calls/(\d+)/turns", path)
        if m and method == "POST":
            return self._turn(int(m.group(2)), json.loads(body), lower)
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
