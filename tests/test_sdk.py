"""EvalKit Python SDK tests — no network, no real providers.

Run:
    cd apps/sdk-py && python -m unittest -v
"""
from __future__ import annotations

import asyncio
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import evalkit
from evalkit.core.context import _ambient_trace_id, _current_span


# ── fakes for google.genai ──────────────────────────────────────────────────

class _FC:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class _Part:
    def __init__(self, text=None, function_call=None):
        self.text = text
        self.function_call = function_call


class _Content:
    def __init__(self, parts):
        self.parts = parts


class _Candidate:
    def __init__(self, parts):
        self.content = _Content(parts)


class _Usage:
    def __init__(self, pin, pout):
        self.prompt_token_count = pin
        self.candidates_token_count = pout


class _Resp:
    def __init__(self, text="", tool=None, pin=3, pout=5):
        parts = []
        if text:
            parts.append(_Part(text=text))
        if tool:
            parts.append(_Part(function_call=_FC(tool[0], tool[1])))
        self.candidates = [_Candidate(parts)]
        self.usage_metadata = _Usage(pin, pout)
        self.text = text


# ── fakes for Anthropic (Claude) ─────────────────────────────────────────────

class _ABlock:
    def __init__(self, type, text=None, name=None, id=None, input=None):
        self.type = type
        self.text = text
        self.name = name
        self.id = id
        self.input = input


class _AUsage:
    def __init__(self, i, o):
        self.input_tokens = i
        self.output_tokens = o


class _AResp:
    def __init__(self, text="hi", tool=None):
        self.content = [_ABlock("text", text=text)] if text else []
        if tool:
            self.content.append(_ABlock("tool_use", name=tool[0], id="t1", input=tool[1]))
        self.usage = _AUsage(3, 5)
        self.model = "claude-3-5-sonnet"


def install_fake_anthropic(client, resp):
    from evalkit.instrumentation import auto as _auto

    class FakeMessages:
        def create(self, *a, **k):
            return resp

    _auto._patch_anthropic_class_sync(client, FakeMessages)
    return FakeMessages()


def install_fake_genai(client, *, response=None, stream_chunks=None):
    """Patch a fake google.genai Models class and return an instance."""
    from evalkit.instrumentation.llm import google_genai as gg

    resp = response or _Resp(text="hello")

    class FakeModels:
        def generate_content(self, *, model, contents, config=None, **k):
            return resp

        def generate_content_stream(self, *, model, contents, config=None, **k):
            return iter(stream_chunks or [])

    fake_mod = types.SimpleNamespace(Models=FakeModels, AsyncModels=None)
    gg._patch_sync_models(client, fake_mod)
    return FakeModels()


# ── helpers ─────────────────────────────────────────────────────────────────

_CLIENT = None


def fresh(capture):
    # Single client across tests so the global patches (requests/httpx/openai/…),
    # which bind to the first initialized client, route spans to the active capture.
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = evalkit.init(subscription_key="tk_test", base_url="http://127.0.0.1:1", service_name="test")
    _CLIENT._exporter.export = lambda batch: (capture.extend(batch), True)[1]
    return _CLIENT


def reset_context():
    _ambient_trace_id.set(None)
    _current_span.set(None)


# ── tests ────────────────────────────────────────────────────────────────────

class TestImport(unittest.TestCase):
    def test_version(self):
        self.assertEqual(evalkit.__version__, "0.1.17")

    def test_public_api(self):
        for name in ("init", "simulate_user", "generate_scenarios", "evaluate",
                     "patch_langchain", "patch_litellm", "patch_google_genai"):
            self.assertTrue(hasattr(evalkit, name), name)


class TestGoogleGenAI(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_single_trace_across_calls(self):
        """ADK fix: many calls with no active span share ONE trace id."""
        m = install_fake_genai(self.client)
        for i in range(6):
            m.generate_content(model="gemini-2.0-flash", contents=f"q{i}")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 6)
        self.assertEqual(len({e.trace_id for e in llm}), 1)
        self.assertEqual(llm[0].provider, "google-genai")
        self.assertEqual(llm[0].tokens_in, 3)
        self.assertEqual(llm[0].tokens_out, 5)

    def test_joins_active_trace(self):
        m = install_fake_genai(self.client)
        trace_id, end, ctx = self.client.start_trace("root")
        with evalkit.with_trace(ctx):
            m.generate_content(model="g", contents="hi")
        end("OK")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(llm[0].trace_id, trace_id)
        self.assertEqual(llm[0].parent_span_id, ctx.span_id)

    def test_tool_events_captured(self):
        m = install_fake_genai(self.client, response=_Resp(text="", tool=("get_weather", {"city": "london"})))
        m.generate_content(model="g", contents="weather?")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertTrue(llm.events)
        self.assertEqual(llm.events[0].attributes["gen_ai.tool.name"], "get_weather")

    def test_streaming(self):
        chunks = [_Resp(text="Hel"), _Resp(text="lo!", pin=0, pout=7)]
        m = install_fake_genai(self.client, stream_chunks=chunks)
        out = list(m.generate_content_stream(model="g", contents="hi"))
        self.assertEqual(len(out), 2)
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.completion, "Hello!")
        self.assertTrue(llm.attributes.get("gen_ai.streaming"))

    def test_tool_call_span_emitted(self):
        """A tool the model requests becomes a first-class tool_call span nested
        under the llm_call (emitted centrally in enqueue, all providers)."""
        m = install_fake_genai(self.client, response=_Resp(text="", tool=("get_weather", {"city": "london"})))
        m.generate_content(model="g", contents="weather?")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        tools = [e for e in self.captured if e.span_type == "tool_call"]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].operation, "get_weather")
        self.assertEqual(tools[0].parent_span_id, llm.span_id)
        self.assertEqual(tools[0].trace_id, llm.trace_id)


class TestAnthropicClaude(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_claude_llm_and_tool_span(self):
        m = install_fake_anthropic(self.client, _AResp(text="hi", tool=("get_weather", {"city": "london"})))
        m.create(model="claude-3-5-sonnet", system="be helpful",
                 messages=[{"role": "user", "content": "weather?"}])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].provider, "anthropic")
        self.assertEqual(llm[0].completion, "hi")
        self.assertEqual(llm[0].tokens_in, 3)
        self.assertEqual(llm[0].tokens_out, 5)
        tools = [e for e in self.captured if e.span_type == "tool_call"]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].operation, "get_weather")
        self.assertEqual(tools[0].parent_span_id, llm[0].span_id)

    def test_claude_simulate_user(self):
        m = install_fake_anthropic(self.client, _AResp(text="resolved", tool=("issue_refund", {"id": "1"})))

        def agent(ctx):
            m.create(model="claude-3-5-sonnet", messages=[{"role": "user", "content": ctx.message}])
            return "ok"

        report = evalkit.simulate_user(
            agent,
            [{"name": "refund", "starting_prompt": "charged twice", "expected_tools": ["issue_refund"]}],
        )
        self.assertEqual(report["results"][0]["status"], "OK")
        self.assertEqual(report["results"][0]["scores"].get("tool_correctness"), 1.0)


class TestGenerateScenarios(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_contract_and_span(self):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        recv = {}

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                ln = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(ln)
                recv["path"] = self.path
                recv["key"] = self.headers.get("X-Subscription-Key")
                recv["body"] = json.loads(body or b"{}")
                payload = json.dumps({"scenarios": [{"name": "s1", "starting_prompt": "hi", "expected_tools": ["t"]}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            scenarios = evalkit.generate_scenarios(
                "You are a billing agent.",
                tools=["issue_refund"],
                count=2,
                api_url=f"http://127.0.0.1:{port}",
            )
        finally:
            srv.shutdown()

        self.assertEqual(len(scenarios), 1)
        self.assertEqual(recv["path"], "/scenarios/generate")
        self.assertEqual(recv["key"], "tk_test")
        self.assertEqual(recv["body"]["agentInstructions"], "You are a billing agent.")
        self.assertEqual(recv["body"]["tools"], ["issue_refund"])
        self.assertEqual(recv["body"]["count"], 2)
        self.client.flush()
        spans = [e for e in self.captured if (e.operation or "").startswith("POST /scenarios/generate")]
        self.assertTrue(spans, "generate_scenarios should emit an http_call span")


class TestDB(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_sqlalchemy_db_query(self):
        try:
            from sqlalchemy import create_engine, text
        except ImportError:
            self.skipTest("sqlalchemy not installed")
        eng = create_engine("sqlite:///:memory:")
        evalkit.patch_sqlalchemy_engine(eng)
        tid, end, ctx = self.client.start_trace("req")
        with evalkit.with_trace(ctx):
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
        end("OK")
        self.client.flush()
        eng.dispose()
        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertTrue(db, "expected a db_query span")
        self.assertEqual(db[0].trace_id, tid)
        self.assertIn("db.statement", db[0].attributes)


class TestHTTP(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_requests_http_call(self):
        try:
            import requests
        except ImportError:
            self.skipTest("requests not installed")
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *a):
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            tid, end, ctx = self.client.start_trace("req")
            with evalkit.with_trace(ctx):
                requests.get(f"http://127.0.0.1:{port}/data")
            end("OK")
        finally:
            srv.shutdown()
        self.client.flush()
        http = [e for e in self.captured if e.span_type == "http_call" and "127.0.0.1" in (e.attributes or {}).get("http.url", "")]
        self.assertTrue(http, "expected an http_call span")
        self.assertEqual(http[0].trace_id, tid)
        # outgoing call carries a traceparent for cross-service propagation
        self.assertIn("traceparent", (http[0].attributes or {}).get("http.request.headers", ""))


class TestDBDrivers(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._injected = []

    def tearDown(self):
        for name in self._injected:
            sys.modules.pop(name, None)

    def _inject(self, name, mod):
        sys.modules[name] = mod
        self._injected.append(name)

    def test_start_db_span(self):
        from evalkit.instrumentation.db._common import start_db
        tid, end_tr, ctx = self.client.start_trace("req")
        with evalkit.with_trace(ctx):
            end = start_db(self.client, "postgresql", statement="SELECT * FROM users WHERE id=1")
            end("OK", rows=1)
        end_tr("OK")
        self.client.flush()
        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertTrue(db)
        a = db[0].attributes
        self.assertEqual(a["db.system"], "postgresql")
        self.assertEqual(a["db.operation"], "SELECT")
        self.assertEqual(a["db.sql.table"], "users")
        self.assertEqual(a["db.rows_affected"], 1)
        self.assertEqual(db[0].trace_id, tid)

    def test_redis_sync_and_async(self):
        redis_mod = types.ModuleType("redis")
        client_mod = types.ModuleType("redis.client")

        class FakeRedis:
            def execute_command(self, *args, **kw):
                return "OK"

        client_mod.Redis = FakeRedis
        redis_mod.client = client_mod

        asyncio_mod = types.ModuleType("redis.asyncio")
        asyncio_client_mod = types.ModuleType("redis.asyncio.client")

        class FakeAsyncRedis:
            async def execute_command(self, *a, **k):
                return "OK"

        asyncio_client_mod.Redis = FakeAsyncRedis
        asyncio_mod.client = asyncio_client_mod
        for n, m in (("redis", redis_mod), ("redis.client", client_mod),
                     ("redis.asyncio", asyncio_mod), ("redis.asyncio.client", asyncio_client_mod)):
            self._inject(n, m)

        from evalkit.instrumentation.db.redis import patch_redis
        patch_redis(self.client)

        FakeRedis().execute_command("GET", "mykey")
        asyncio.run(FakeAsyncRedis().execute_command("SET", "k", "v"))
        self.client.flush()
        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 2)
        ops = {e.attributes["db.operation"] for e in db}
        self.assertEqual(ops, {"GET", "SET"})
        self.assertTrue(all(e.attributes["db.system"] == "redis" for e in db))

    def test_asyncpg(self):
        asyncpg_mod = types.ModuleType("asyncpg")
        conn_mod = types.ModuleType("asyncpg.connection")

        class FakeConn:
            async def execute(self, query, *a, **k):
                return "INSERT 0 1"

        conn_mod.Connection = FakeConn
        asyncpg_mod.connection = conn_mod
        self._inject("asyncpg", asyncpg_mod)
        self._inject("asyncpg.connection", conn_mod)

        from evalkit.instrumentation.db.asyncpg import patch_asyncpg
        patch_asyncpg(self.client)
        asyncio.run(FakeConn().execute("SELECT * FROM accounts"))
        self.client.flush()
        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertTrue(db)
        self.assertEqual(db[0].attributes["db.system"], "postgresql")
        self.assertEqual(db[0].attributes["db.operation"], "SELECT")
        self.assertEqual(db[0].attributes["db.sql.table"], "accounts")

    def test_databases_no_crash_when_absent(self):
        from evalkit.instrumentation.auto import _try_patch_databases
        _try_patch_databases(self.client)  # redis/pymongo/asyncpg/psycopg/pymysql absent → no-op


class TestFrameworks(unittest.TestCase):
    """Every framework middleware/integration must be importable from the package."""

    def test_all_exports_present(self):
        for name in (
            "EvalKitMiddleware", "EvalKitStarletteMiddleware", "EvalKitDjangoMiddleware",
            "instrument_flask", "create_litestar_middleware",
            "patch_sqlalchemy_engine", "patch_openai_client", "patch_anthropic_client",
            "patch_bedrock_client", "patch_cohere_client", "patch_google_ai_model",
            "patch_google_genai", "patch_mistral_client", "patch_langchain", "patch_litellm",
            "start_trace", "start_span", "with_trace", "trace_function", "flush",
        ):
            self.assertTrue(hasattr(evalkit, name), f"missing export: {name}")


class TestSimulateUser(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_sync_tool_capture_and_tagging(self):
        """A sync agent that calls google.genai with a tool → tool metrics scored,
        spans tagged, one trace per scenario."""
        m = install_fake_genai(self.client, response=_Resp(text="", tool=("get_weather", {"city": "london"})))

        def agent(ctx):
            m.generate_content(model="g", contents=ctx.message)
            return "done"

        scenarios = [
            {"name": "s1", "starting_prompt": "weather in london?", "expected_tools": ["get_weather"]},
            {"name": "s2", "starting_prompt": "and tokyo?", "expected_tools": ["get_weather"]},
        ]
        report = evalkit.simulate_user(agent, scenarios)
        self.assertEqual(len(report["results"]), 2)
        for r in report["results"]:
            self.assertEqual(r["status"], "OK")
            # tool captured from llm_call events → tool metrics computed
            self.assertEqual(r["scores"].get("tool_correctness"), 1.0)

        sim_spans = [e for e in self.captured if e.is_simulation]
        self.assertTrue(sim_spans)
        self.assertTrue(all(e.simulation_id == report["simulation_id"] for e in sim_spans))
        # each scenario is its own trace
        scenario_roots = [e for e in self.captured if (e.operation or "").startswith("scenario:")]
        self.assertEqual(len({e.trace_id for e in scenario_roots}), 2)

    def test_async_entrypoint(self):
        m = install_fake_genai(self.client, response=_Resp(text="ok"))

        async def agent(ctx):
            m.generate_content(model="g", contents=ctx.message)
            return "async-done"

        scenarios = [{"name": "a1", "starting_prompt": "hi"}]
        report = evalkit.simulate_user(agent, scenarios)
        self.assertEqual(report["results"][0]["status"], "OK")
        self.assertEqual(report["results"][0]["turns"], 1)

    def test_thread_spawned_by_entrypoint_is_tagged(self):
        """If the agent does its work in a thread it spawns, the spans from that
        thread must STILL be simulation-tagged and on the scenario's trace
        (context is copied via the patched threading.Thread.start)."""
        import threading
        m = install_fake_genai(self.client, response=_Resp(text="", tool=("get_weather", {"city": "x"})))

        def agent(ctx):
            def work():
                m.generate_content(model="g", contents=ctx.message)
            t = threading.Thread(target=work)
            t.start()
            t.join()
            return "ok"

        report = evalkit.simulate_user(
            agent, [{"name": "s", "starting_prompt": "hi", "expected_tools": ["get_weather"]}]
        )
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertTrue(llm, "llm_call from spawned thread must be captured")
        self.assertTrue(all(e.is_simulation for e in llm), "thread spans must be simulation-tagged")
        root = [e for e in self.captured if (e.operation or "").startswith("scenario:")][0]
        self.assertTrue(all(e.trace_id == root.trace_id for e in llm), "thread spans must join the scenario trace")
        self.assertEqual(report["results"][0]["scores"].get("tool_correctness"), 1.0)

    def test_threadpool_spawned_by_entrypoint_is_tagged(self):
        """Same guarantee for concurrent.futures / asyncio.run_in_executor."""
        from concurrent.futures import ThreadPoolExecutor
        m = install_fake_genai(self.client, response=_Resp(text="ok"))

        pool = ThreadPoolExecutor(max_workers=2)

        def agent(ctx):
            futs = [pool.submit(m.generate_content, model="g", contents=ctx.message) for _ in range(3)]
            for f in futs:
                f.result()
            return "ok"

        report = evalkit.simulate_user(agent, [{"name": "s", "starting_prompt": "hi"}])
        pool.shutdown()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 3, "all 3 pooled calls captured")
        self.assertTrue(all(e.is_simulation for e in llm), "pool spans must be simulation-tagged")
        root = [e for e in self.captured if (e.operation or "").startswith("scenario:")][0]
        self.assertEqual(len({e.trace_id for e in llm}), 1, "all pooled spans on ONE trace")
        self.assertTrue(all(e.trace_id == root.trace_id for e in llm), "pooled spans join the scenario trace")

    def test_entrypoint_exception_is_isolated(self):
        def agent(ctx):
            raise RuntimeError("boom")

        report = evalkit.simulate_user(agent, [{"name": "x", "starting_prompt": "hi"}])
        self.assertEqual(report["results"][0]["status"], "ERROR")
        self.assertIn("boom", report["results"][0]["error"])


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_tool_metrics(self):
        scores = evalkit.evaluate(
            output="The answer covers AWS and Azure.",
            expected_tools=["search", "render"],
            tool_calls=[{"name": "search"}, {"name": "render"}],
            constraints={"required_terms": ["AWS", "Azure"]},
        )
        self.assertEqual(scores["tool_f1"], 1.0)
        self.assertEqual(scores["response_match"], 1.0)

    def test_eval_result_span_pushed(self):
        evalkit.evaluate(output="x", expected_tools=["a"], tool_calls=[{"name": "a"}])
        self.client.flush()
        evs = [e for e in self.captured if e.span_type == "eval_result"]
        self.assertTrue(evs)


class TestNoCrashWhenLibsAbsent(unittest.TestCase):
    """LangChain / LiteLLM / Groq are not installed here — must be no-ops."""

    def setUp(self):
        reset_context()
        self.client = fresh([])

    def test_langchain_noop(self):
        # absent → returns None, never raises
        self.assertIsNone(evalkit.patch_langchain())

    def test_litellm_noop(self):
        evalkit.patch_litellm()  # should not raise

    def test_auto_instrument_idempotent(self):
        from evalkit.instrumentation.auto import setup_auto_instrumentation
        setup_auto_instrumentation(self.client)  # second run must not raise


class TestASGIMiddleware(unittest.TestCase):
    """Drive EvalKitMiddleware (FastAPI / Starlette / ASGI) directly with a fake app."""

    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _run(self, app):
        mw = evalkit.EvalKitMiddleware(app, self.client)
        tp = b"00-" + b"a" * 32 + b"-" + b"b" * 16 + b"-01"
        scope = {
            "type": "http", "method": "POST", "path": "/api/chat", "query_string": b"",
            "headers": [(b"content-type", b"application/json"), (b"traceparent", tp)],
            "scheme": "http", "server": ("localhost", 80),
        }

        async def receive():
            return {"type": "http.request", "body": b'{"q":"hi"}', "more_body": False}

        sent = []

        async def send(m):
            sent.append(m)

        asyncio.run(mw(scope, receive, send))
        return sent

    def test_success_joins_upstream_and_injects_headers(self):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": b'{"ok":true}'})

        sent = self._run(app)
        self.client.flush()
        http = [e for e in self.captured if e.span_type == "http_call"]
        self.assertTrue(http)
        e = http[0]
        self.assertEqual(e.status, "OK")
        self.assertEqual(e.trace_id, "a" * 32, "should join upstream W3C traceparent")
        self.assertEqual(e.attributes["http.status_code"], 200)
        start = [m for m in sent if m["type"] == "http.response.start"][0]
        hdr_keys = {k for k, _ in start["headers"]}
        self.assertIn(b"x-trace-id", hdr_keys)
        self.assertIn(b"traceparent", hdr_keys)

    def test_error_path_records_error_span(self):
        async def app(scope, receive, send):
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            self._run(app)
        self.client.flush()
        http = [e for e in self.captured if e.span_type == "http_call"]
        self.assertTrue(http, "error must still produce an http_call span")
        self.assertEqual(http[0].status, "ERROR")


class TestLangChainHandler(unittest.TestCase):
    """Validate the LangChain handler maps runs → a correct span tree, using a
    fake langchain_core so the test runs without LangChain installed."""

    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

        base = types.ModuleType("langchain_core.callbacks.base")
        class BaseCallbackHandler:  # minimal stand-in
            pass
        base.BaseCallbackHandler = BaseCallbackHandler

        manager = types.ModuleType("langchain_core.callbacks.manager")
        manager.register_configure_hook = lambda *a, **k: None

        callbacks = types.ModuleType("langchain_core.callbacks")
        root = types.ModuleType("langchain_core")

        self._mods = {
            "langchain_core": root,
            "langchain_core.callbacks": callbacks,
            "langchain_core.callbacks.base": base,
            "langchain_core.callbacks.manager": manager,
        }
        sys.modules.update(self._mods)
        # Reset the module-global registration flag so the hook re-registers.
        import evalkit.instrumentation.llm.langchain as lc
        lc._registered = False
        lc._handler_var = None

    def tearDown(self):
        for name in self._mods:
            sys.modules.pop(name, None)
        import evalkit.instrumentation.llm.langchain as lc
        lc._registered = False
        lc._handler_var = None

    def test_span_tree(self):
        handler = evalkit.patch_langchain()
        self.assertIsNotNone(handler)

        handler.on_chain_start({"name": "graph"}, {}, run_id="A", parent_run_id=None)
        handler.on_chat_model_start(
            {"id": ["langchain", "chat_models", "openai", "ChatOpenAI"], "kwargs": {"model": "gpt-4o"}},
            [[types.SimpleNamespace(type="human", content="hi")]],
            run_id="B", parent_run_id="A",
        )
        handler.on_llm_end(
            types.SimpleNamespace(
                generations=[[types.SimpleNamespace(text="hello", message=None)]],
                llm_output={"token_usage": {"prompt_tokens": 4, "completion_tokens": 6}},
            ),
            run_id="B",
        )
        handler.on_tool_start({"name": "search"}, "query", run_id="C", parent_run_id="A")
        handler.on_tool_end("result", run_id="C")
        handler.on_chain_end({}, run_id="A")
        self.client.flush()

        by_type = {}
        for e in self.captured:
            by_type.setdefault(e.span_type, []).append(e)

        chain = by_type["function_call"][0]
        llm = by_type["llm_call"][0]
        tool = by_type["tool_call"][0]

        # one trace, correct parent/child wiring
        self.assertEqual(chain.trace_id, llm.trace_id)
        self.assertEqual(chain.trace_id, tool.trace_id)
        self.assertEqual(llm.parent_span_id, chain.span_id)
        self.assertEqual(tool.parent_span_id, chain.span_id)
        self.assertEqual(llm.model, "gpt-4o")
        self.assertEqual(llm.provider, "openai")
        self.assertEqual(llm.completion, "hello")
        self.assertEqual(llm.tokens_in, 4)
        self.assertEqual(llm.tokens_out, 6)
        self.assertEqual(tool.attributes.get("tool.name"), "search")


if __name__ == "__main__":
    unittest.main(verbosity=2)
