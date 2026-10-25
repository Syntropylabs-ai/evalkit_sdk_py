"""ignore_routes, ignore_urls, ignore_functions and evalkit.untraced keep things out of traces.

An ignored request is left out the way a health probe is, at every layer: its server span, what its
handler records (spans, traced functions, model calls, log lines, other libraries' spans) and the
traceparent of its outgoing calls. An ignored outgoing call has no span and sends no traceparent. An
ignored function is not wrapped, even by an explicit decorator. An untraced block records nothing, per
thread and per asyncio task, so a concurrent request is traced as usual. A pattern that cannot be read
is skipped with one warning; init() never fails over one.
"""
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import evalkit
from evalkit.core import ignore
from tests.helpers import OResp, fresh, install_fake_openai_sync, reset_context
from tests.test_framework_spans import _call

ROOT = os.path.join(os.path.dirname(__file__), "..")


class _Configured(unittest.TestCase):
    """Rules set for one test and cleared after it, whatever it did."""

    def configure(self, routes=None, urls=None, functions=None, trace_health_checks=False):
        ignore.configure(routes, urls, functions, trace_health_checks)
        self.addCleanup(ignore.reset)


class TestRoutePatterns(_Configured):
    def check(self, patterns, cases):
        self.configure(routes=patterns)
        for method, path, route, expected in cases:
            with self.subTest(method=method, path=path, route=route):
                self.assertIs(ignore.ignored_route(method, path, route), expected)

    def test_an_exact_path_ignores_a_trailing_slash_and_the_query_string(self):
        self.check(["/metrics"], [
            ("GET", "/metrics", None, True), ("GET", "/metrics/", None, True),
            ("GET", "/metrics?format=prom", None, True), ("GET", "/metrics#top", None, True),
            ("GET", "//metrics//", None, True), ("GET", "http://svc:8080/metrics?x=1", None, True),
            ("GET", "/metrics/extra", None, False), ("GET", "/metric", None, False),
            ("GET", "/api/metrics", None, False), ("GET", "/cb?next=/metrics", None, False),
        ])

    def test_paths_are_case_sensitive_and_methods_are_not(self):
        self.check(["GET /status", "post /hooks"], [
            ("GET", "/status", None, True), ("get", "/status", None, True), ("Get", "/status/", None, True),
            ("POST", "/status", None, False), ("HEAD", "/status", None, False), (None, "/status", None, False),
            ("GET", "/Status", None, False), ("POST", "/hooks", None, True), ("GET", "/hooks", None, False),
        ])

    def test_a_star_method_is_any_method(self):
        self.check(["* /any"], [("DELETE", "/any", None, True), ("GET", "/any", None, True)])

    def test_star_is_one_segment_and_double_star_any_number(self):
        self.check(["/internal/*", "/static/**", "**/debug"], [
            ("GET", "/internal/jobs", None, True), ("GET", "/internal/jobs/", None, True),
            ("GET", "/internal", None, False), ("GET", "/internal/jobs/7", None, False),
            ("GET", "/static", None, True), ("GET", "/static/app.js", None, True),
            ("GET", "/static/img/logo/x.png", None, True), ("GET", "/staticfiles", None, False),
            ("GET", "/debug", None, True), ("GET", "/a/b/debug", None, True), ("GET", "/debug/x", None, False),
        ])

    def test_a_star_inside_a_segment_stays_inside_it(self):
        self.check(["/assets/*.css", "/v*/ping"], [
            ("GET", "/assets/site.css", None, True), ("GET", "/assets/.css", None, True),
            ("GET", "/assets/site.js", None, False), ("GET", "/assets/a/site.css", None, False),
            ("GET", "/v1/ping", None, True), ("GET", "/v2/ping", None, True), ("GET", "/x1/ping", None, False),
        ])

    def test_a_route_template_matches_the_raw_path_and_the_frameworks_route(self):
        self.check(["GET /v1/orders/:id", "/users/{user_id}/avatar", "/files/<int:file_id>"], [
            ("GET", "/v1/orders/42", None, True), ("GET", "/v1/orders/42/", None, True),
            ("GET", "/v1/orders/42?expand=items", None, True), ("GET", "/v1/orders", None, False),
            ("GET", "/v1/orders/42/items", None, False), ("POST", "/v1/orders/42", None, False),
            ("GET", "/users/9/avatar", None, True), ("GET", "/files/3", None, True),
            # The framework's template, when it is known, is matched as well as the raw path.
            ("GET", "/srv/7", "/v1/orders/{order_id}", True), ("GET", "/srv/7", "/v1/orders/<int:order_id>/", True),
            ("GET", "/srv/7", "/v1/carts/{cart_id}", False),
        ])

    def test_a_compiled_regex_is_searched_in_the_path(self):
        self.check([re.compile(r"^/admin(/|$)"), re.compile(r"\.php$")], [
            ("GET", "/admin", None, True), ("GET", "/admin/users", None, True), ("GET", "/admin/", None, True),
            ("GET", "/administrator", None, False), ("GET", "/wp/login.php", None, True),
            ("GET", "/wp/login.php?x=1", None, True), ("GET", "/wp/login.php/", None, True),
            ("GET", "/wp/login.phpx", None, False),
        ])

    def test_a_callable_decides_from_method_path_and_route(self):
        seen = []

        def internal(info):
            seen.append((info.method, info.path, info.route))
            return info.method == "POST" and info.path.startswith("/internal/")

        self.configure(routes=[internal])
        self.assertTrue(ignore.ignored_route("post", "/internal/jobs?run=1"))
        self.assertFalse(ignore.ignored_route("GET", "/internal/jobs"))
        self.assertTrue(ignore.ignored_route("POST", "/internal/x", "/internal/{name}"))
        self.assertEqual(seen, [("POST", "/internal/jobs", None), ("GET", "/internal/jobs", None),
                                ("POST", "/internal/x", "/internal/{name}")])

    def test_a_callable_is_handed_a_route_worked_out_only_when_it_reads_it(self):
        resolved = []

        def resolve():
            resolved.append(1)
            return "/orders/{order_id}"

        self.configure(routes=[lambda info: info.method == "DELETE"])
        self.assertFalse(ignore.ignored_route("GET", "/orders/1", resolve_route=resolve))
        self.assertEqual(resolved, [], "a callable that does not read the route costs the request nothing")
        ignore.configure([lambda info: info.route == "/orders/{order_id}"])
        self.assertTrue(ignore.ignored_route("GET", "/orders/1", resolve_route=resolve))
        self.assertEqual(resolved, [1])

    def test_a_callable_that_raises_does_not_match_and_is_reported_once(self):
        def broken(info):
            raise RuntimeError("boom")

        self.configure(routes=[broken])
        with self.assertLogs("evalkit", "WARNING") as logs:
            self.assertFalse(ignore.ignored_route("GET", "/x"))
            self.assertFalse(ignore.ignored_route("GET", "/y"))
        self.assertEqual(len(logs.records), 1, logs.output)

    def test_an_async_callable_does_not_match_and_is_reported_once(self):
        async def decide(info):
            return True

        self.configure(routes=[decide])
        with self.assertLogs("evalkit", "WARNING") as logs:
            self.assertFalse(ignore.ignored_route("GET", "/x"))
            self.assertFalse(ignore.ignored_route("GET", "/y"))
        self.assertEqual(len(logs.records), 1, logs.output)
        self.assertIn("awaitable", logs.output[0])

    def test_the_root_and_nothing_configured(self):
        self.assertFalse(ignore.ignored_route("GET", "/metrics"))
        self.check(["/"], [("GET", "/", None, True), ("GET", "", None, True), ("GET", "/x", None, False)])


class TestUrlPatterns(_Configured):
    def check(self, patterns, cases):
        self.configure(urls=patterns)
        for url, expected in cases:
            with self.subTest(url=url):
                self.assertIs(ignore.ignored_url(url, "GET"), expected)

    def test_a_host_alone_is_every_call_to_it(self):
        self.check(["api.stripe.com"], [
            ("https://api.stripe.com/v1/charges", True), ("https://API.Stripe.com", True),
            ("http://api.stripe.com:8080/x", True), ("https://files.api.stripe.com/x", False),
            ("https://api.stripe.company/x", False),
        ])

    def test_host_and_path_globs(self):
        self.check(["*.internal.example.com/**", "**.cdn.example.com", "api.example.com/v1/orders/:id"], [
            ("http://jobs.internal.example.com/run?x=1", True), ("http://internal.example.com/run", False),
            ("http://a.b.internal.example.com/run", False), ("https://cdn.example.com/x.js", True),
            ("https://eu.west.cdn.example.com/x.js", True), ("https://api.example.com/v1/orders/9", True),
            ("https://api.example.com/v1/orders/9/", True), ("https://api.example.com/v1/orders", False),
        ])

    def test_scheme_and_port_narrow_the_match(self):
        self.check(["https://s3.amazonaws.com/bucket/*", "localhost:9000", "http://*:6333/**", "*://svc:443"], [
            ("https://s3.amazonaws.com/bucket/key", True), ("http://s3.amazonaws.com/bucket/key", False),
            ("https://s3.amazonaws.com/bucket/a/b", False), ("http://localhost:9000/minio", True),
            ("http://localhost:9001/minio", False), ("http://qdrant:6333/collections", True),
            ("http://qdrant:6334/collections", False), ("https://svc/x", True), ("http://svc/x", False),
        ])

    def test_a_path_alone_is_that_path_on_any_host(self):
        self.check(["/v1/telemetry/**"], [
            ("http://a.example/v1/telemetry", True), ("http://b.example/v1/telemetry/batch?x=1", True),
            ("http://b.example/v1/telemetrics", False),
        ])

    def test_ipv6_hosts(self):
        self.check(["[::1]:8080"], [("http://[::1]:8080/x", True), ("http://[::1]:8081/x", False)])

    def test_a_regex_is_searched_in_the_full_url_and_a_callable_gets_the_parts(self):
        seen = []

        def by_host(info):
            seen.append((info.method, info.host, info.path))
            return info.host.endswith(".corp")

        self.check([re.compile(r"[?&]nocache=1"), by_host], [
            ("https://api.example.com/x?nocache=1", True), ("https://git.corp/a/b?x=1", True),
            ("https://api.example.com/x", False),
        ])
        self.assertIn(("GET", "git.corp", "/a/b"), seen)


class TestFunctionPatterns(_Configured):
    def test_dotted_globs(self):
        self.configure(functions=["myapp.cache.*", "myapp.utils.slugify", "myapp.**.helpers", "*.debug_*",
                                  re.compile(r"\.test_"), lambda name: name.endswith("_quiet")])
        for name, expected in [
            ("myapp.cache.get", True), ("myapp.cache.Client.get", False), ("myapp.cache", False),
            ("myapp.utils.slugify", True), ("myapp.utils.slug", False), ("myapp.helpers", True),
            ("myapp.a.b.helpers", True), ("svc.debug_dump", True), ("svc.x.debug_dump", False),
            ("pkg.test_x", True), ("pkg.run_quiet", True), ("pkg.run", False), ("", False),
        ]:
            with self.subTest(name=name):
                self.assertIs(ignore.ignored_function(name), expected)

    def test_double_star_covers_a_modules_classes(self):
        self.configure(functions=["myapp.cache.**"])
        self.assertTrue(ignore.ignored_function("myapp.cache.Client.get"))
        self.assertTrue(ignore.ignored_function("myapp.cache.redis.get"))


class TestInvalidPatterns(_Configured):
    def test_each_invalid_pattern_warns_once_and_the_rest_apply(self):
        with self.assertLogs("evalkit", "WARNING") as logs:
            self.configure(
                routes=["metrics", "", "GET", "G3T /x", "GET /a b", 42, None, "/ok"],
                urls=["http://", "host:99999", "host:abc", "ok.example.com", "a..b"],
                functions=["bad name", "a..b", "", "pkg.ok"],
            )
        self.assertEqual(len(logs.records), 14, "\n".join(logs.output))
        self.assertTrue(ignore.ignored_route("GET", "/ok"))
        self.assertTrue(ignore.ignored_url("https://ok.example.com/x"))
        self.assertTrue(ignore.ignored_function("pkg.ok"))

    def test_a_single_pattern_and_an_unreadable_option(self):
        with self.assertLogs("evalkit", "WARNING") as logs:
            self.configure(routes="/metrics", urls=re.compile("x"), functions=12)
        self.assertTrue(ignore.ignored_route("GET", "/metrics"))
        self.assertTrue(ignore.ignored_url("http://x/"))
        self.assertEqual(len(logs.records), 1, logs.output)

    def test_a_url_with_credentials_is_not_repeated_in_the_warning(self):
        with self.assertLogs("evalkit", "WARNING") as logs:
            self.configure(urls=["https://user:hunter2-secret@:99999/x"])
        self.assertNotIn("hunter2-secret", "\n".join(logs.output))

    def test_init_with_invalid_patterns_warns_and_keeps_tracing(self):
        code = (
            "import evalkit\n"
            "c = evalkit.init(subscription_key='tk_test', base_url='http://127.0.0.1:1', function_tracing=False,"
            " ignore_routes=['nope', 42, '/metrics'], ignore_urls=['http://'], ignore_functions=['a b'],"
            " trace_health_checks='yes')\n"
            "from evalkit.core import ignore\n"
            "print(c.enabled, ignore.ignored_route('GET', '/metrics'), ignore.traces_health_checks())\n"
        )
        env = {k: v for k, v in os.environ.items() if not k.startswith("EVALKIT_IGNORE_")}
        env.update(EVALKIT_DISABLE_TRACING="true", PYTHONDONTWRITEBYTECODE="1")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
                             timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "True True True")
        self.assertEqual(out.stderr.count("is skipped"), 4, out.stderr)


class TestEnvironment(_Configured):
    def test_env_patterns_are_merged_with_the_options(self):
        env = {
            ignore.ROUTES_ENV: " /metrics , GET /v1/orders/:id,, ",
            ignore.URLS_ENV: "api.stripe.com,https://s3.amazonaws.com/**",
            ignore.FUNCTIONS_ENV: "myapp.cache.*",
        }
        with mock.patch.dict(os.environ, env):
            self.configure(routes=["/status"], urls=["metadata.google.internal"], functions=["myapp.utils.*"])
        self.assertTrue(ignore.ignored_route("GET", "/status"), "the option applies")
        self.assertTrue(ignore.ignored_route("GET", "/metrics"), "and so does the environment's list")
        self.assertTrue(ignore.ignored_route("GET", "/v1/orders/1"))
        self.assertFalse(ignore.ignored_route("POST", "/v1/orders/1"))
        self.assertTrue(ignore.ignored_url("http://metadata.google.internal/computeMetadata"))
        self.assertTrue(ignore.ignored_url("https://api.stripe.com/v1"))
        self.assertTrue(ignore.ignored_url("https://s3.amazonaws.com/b/k"))
        self.assertTrue(ignore.ignored_function("myapp.cache.get"))
        self.assertTrue(ignore.ignored_function("myapp.utils.slugify"))

    def test_an_invalid_env_pattern_names_the_variable(self):
        with mock.patch.dict(os.environ, {ignore.ROUTES_ENV: "metrics"}):
            with self.assertLogs("evalkit", "WARNING") as logs:
                self.configure()
        self.assertIn(ignore.ROUTES_ENV, logs.output[0])

    def test_init_reads_the_variables(self):
        code = (
            "import evalkit\n"
            "evalkit.init(subscription_key='tk_test', base_url='http://127.0.0.1:1', function_tracing=False,"
            " ignore_routes=['/status'])\n"
            "from evalkit.core import ignore\n"
            "print(ignore.ignored_route('GET', '/status'), ignore.ignored_route('GET', '/metrics'),"
            " ignore.ignored_url('https://api.stripe.com/x'), ignore.ignored_function('app.cache.get'))\n"
        )
        env = dict(os.environ, EVALKIT_DISABLE_TRACING="true", PYTHONDONTWRITEBYTECODE="1",
                   EVALKIT_IGNORE_ROUTES="/metrics", EVALKIT_IGNORE_URLS="api.stripe.com",
                   EVALKIT_IGNORE_FUNCTIONS="app.cache.*")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
                             timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "True True True True")


class _Server:
    """A local HTTP server that remembers the traceparent of each request it answers."""

    def __init__(self):
        received = self.received = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                received.append((self.path, self.headers.get("traceparent")))
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *args):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def traceparent_of(self, path):
        return dict(self.received).get(path)

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


class _Captured(_Configured):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self.client.flush()
        self.captured.clear()

    def exported(self):
        self.client.flush()
        try:
            from evalkit.core.otel_provider import flush_otel

            flush_otel()
        except Exception:
            pass
        self.client.flush()
        return list(self.captured)

    def operations(self):
        return sorted(e.operation or "" for e in self.exported())

    def http_urls(self):
        return [(e.attributes or {}).get("http.url", "") for e in self.exported() if e.span_type == "http_call"]


class TestOutgoingCalls(_Captured):
    def setUp(self):
        super().setUp()
        self.server = _Server()
        self.addCleanup(self.server.close)
        self.configure(urls=[f"127.0.0.1:{self.server.port}/ignored/**"])

    def test_requests(self):
        import requests

        requests.get(self.server.url("/ignored/a"))
        requests.get(self.server.url("/kept"))
        urls = self.http_urls()
        self.assertEqual([u.rsplit("/", 1)[-1] for u in urls], ["kept"], urls)
        self.assertIsNone(self.server.traceparent_of("/ignored/a"), "an ignored call propagates no context")
        self.assertIsNotNone(self.server.traceparent_of("/kept"))

    def test_httpx_sync_and_async(self):
        import httpx

        with httpx.Client() as http:
            http.get(self.server.url("/ignored/sync"))
            http.get(self.server.url("/kept-sync"))

        async def run():
            async with httpx.AsyncClient() as http:
                await http.get(self.server.url("/ignored/async"))
                await http.get(self.server.url("/kept-async"))

        asyncio.run(run())
        urls = sorted(u.rsplit("/", 1)[-1] for u in self.http_urls())
        self.assertEqual(urls, ["kept-async", "kept-sync"])
        self.assertIsNone(self.server.traceparent_of("/ignored/sync"))
        self.assertIsNone(self.server.traceparent_of("/ignored/async"))
        self.assertIsNotNone(self.server.traceparent_of("/kept-sync"))

    def test_aiohttp(self):
        try:
            import aiohttp
        except ImportError:
            self.skipTest("aiohttp not installed")

        async def run():
            async with aiohttp.ClientSession() as session:
                async with session.get(self.server.url("/ignored/aio")) as resp:
                    await resp.read()
                async with session.get(self.server.url("/kept-aio")) as resp:
                    await resp.read()

        asyncio.run(run())
        urls = [u.rsplit("/", 1)[-1] for u in self.http_urls()]
        self.assertEqual(urls, ["kept-aio"])
        self.assertIsNone(self.server.traceparent_of("/ignored/aio"))

    def test_urllib3_through_its_opentelemetry_instrumentation(self):
        try:
            import urllib3
            from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor
        except ImportError:
            self.skipTest("urllib3 instrumentation not installed")
        from evalkit.core.otel_provider import otel_active

        if not otel_active() or not URLLib3Instrumentor().is_instrumented_by_opentelemetry:
            self.skipTest("the SDK did not register the urllib3 instrumentation in this process")
        pool = urllib3.PoolManager()
        end, ctx = self.client.start_span("outer")
        with evalkit.with_trace(ctx):
            pool.request("GET", self.server.url("/ignored/u3"))
            pool.request("GET", self.server.url("/kept-u3"))
        end("OK")
        urls = [u for u in self.http_urls() if "u3" in u]
        self.assertEqual([u.rsplit("/", 1)[-1] for u in urls], ["kept-u3"], urls)
        self.assertIsNone(self.server.traceparent_of("/ignored/u3"), "the dropped span hands no context on")
        self.assertIsNotNone(self.server.traceparent_of("/kept-u3"))

    def test_an_outgoing_health_probe_is_traced_with_trace_health_checks(self):
        import requests

        requests.get(self.server.url("/healthz"))
        self.assertEqual(self.http_urls(), [])
        self.configure(urls=[], trace_health_checks=True)
        requests.get(self.server.url("/healthz"))
        self.assertEqual([u.rsplit("/", 1)[-1] for u in self.http_urls()], ["healthz"])


class TestFunctions(_Captured):
    def test_an_ignored_function_is_not_wrapped_even_by_an_explicit_decorator(self):
        module = types.ModuleType("shop.cache")
        exec("def get(key):\n    return key\n\ndef put(key):\n    return key\n", module.__dict__)
        self.configure(functions=["shop.cache.get"])
        self.assertIs(self.client.trace_function()(module.get), module.get, "the ignore wins")
        self.assertIs(self.client.trace_tool()(module.get), module.get)
        traced_put = self.client.trace_function()(module.put)
        self.assertIsNot(traced_put, module.put)
        traced_put("k")
        self.assertEqual(self.operations(), ["put"])

    def test_function_tracing_leaves_ignored_and_untraced_functions_alone(self):
        module = types.ModuleType("shop.utils")
        sys.modules["shop.utils"] = module
        self.addCleanup(sys.modules.pop, "shop.utils", None)
        exec("def slugify(s):\n    return s\n\ndef price(p):\n    return p\n", module.__dict__)

        @evalkit.untraced
        def quiet(x):
            return x

        quiet.__module__ = "shop.utils"
        module.quiet = quiet
        self.configure(functions=["shop.utils.slugify"])
        self.assertEqual(evalkit.trace_module(module), 1, "only price is wrapped")
        module.slugify("a")
        module.quiet("b")
        module.price(1)
        self.assertEqual(self.operations(), ["utils.price"])


class TestUntraced(_Captured):
    def record_everything(self):
        """What a block can record: a span, a traced function, another library's span, a model call, a
        log line and an outgoing call."""
        from opentelemetry import trace

        end, _ = self.client.start_span("work.span")
        end("OK")
        self.client.trace_function("work.function")(lambda: None)()
        with trace.get_tracer("app.lib").start_as_current_span("work.otel"):
            pass
        install_fake_openai_sync(self.client, OResp("hi")).create(model="gpt-4o-mini", messages=[])
        logging.getLogger("work.test").warning("work log line")
        import requests

        requests.get(self.server.url("/work-call"))

    def setUp(self):
        super().setUp()
        self.server = _Server()
        self.addCleanup(self.server.close)

    def test_a_block_records_nothing_and_the_code_after_it_is_traced(self):
        end, ctx = self.client.start_span("outer")
        with evalkit.with_trace(ctx):
            with evalkit.untraced():
                self.record_everything()
            after, _ = self.client.start_span("after.block")
            after("OK")
        end("OK")
        self.assertEqual(self.operations(), ["after.block", "outer"])
        self.assertIsNone(self.server.traceparent_of("/work-call"), "an untraced call sends no traceparent")

    def test_a_block_ends_however_it_ends(self):
        from evalkit.core.context import tracing_suppressed

        with self.assertRaises(KeyError):
            with evalkit.untraced():
                self.assertTrue(tracing_suppressed())
                raise KeyError("x")
        self.assertFalse(tracing_suppressed())
        with evalkit.untraced():
            with evalkit.untraced():
                pass
            self.assertTrue(tracing_suppressed(), "an inner block leaves the outer one in force")
        self.assertFalse(tracing_suppressed())

    def test_the_decorator_on_sync_async_and_generator_functions(self):
        from evalkit.core.context import tracing_suppressed

        @evalkit.untraced
        def sync_fn(x):
            self.record_everything()
            return x * 2

        @evalkit.untraced()
        async def async_fn(x):
            await asyncio.sleep(0)
            self.record_everything()
            return x + 1

        @evalkit.untraced
        def gen_fn():
            seen = []
            for i in range(2):
                seen.append(tracing_suppressed())
                got = yield i
                seen.append(got)
            return seen

        @evalkit.untraced
        async def agen_fn():
            for i in range(2):
                await asyncio.sleep(0)
                yield (i, tracing_suppressed())

        self.assertEqual(sync_fn(2), 4)
        self.assertEqual(asyncio.run(async_fn(1)), 2)
        gen = gen_fn()
        self.assertEqual(next(gen), 0)
        self.assertFalse(tracing_suppressed(), "the consumer's code between items is traced")
        self.assertEqual(gen.send("a"), 1)
        with self.assertRaises(StopIteration) as stop:
            gen.send("b")
        self.assertEqual(stop.exception.value, [True, "a", True, "b"])

        async def consume():
            out = []
            async for item in agen_fn():
                out.append((item, tracing_suppressed()))
            return out

        self.assertEqual(asyncio.run(consume()), [((0, True), False), ((1, True), False)])
        self.assertEqual(self.operations(), [], "nothing the functions did was recorded")
        self.assertTrue(getattr(sync_fn, "_evalkit_untraced", False))
        self.assertIs(self.client.trace_function()(sync_fn), sync_fn, "an untraced function stays untraced")

    def test_async_with_and_a_concurrent_task(self):
        async def untraced_request():
            async with evalkit.untraced():
                for _ in range(3):
                    await asyncio.sleep(0.01)
                    end, _ = self.client.start_span("untraced.request")
                    end("OK")

        async def traced_request():
            for _ in range(3):
                await asyncio.sleep(0.01)
                end, _ = self.client.start_span("traced.request")
                end("OK")

        async def both():
            await asyncio.gather(untraced_request(), traced_request())

        asyncio.run(both())
        self.assertEqual(self.operations(), ["traced.request"] * 3)

    def test_a_concurrent_thread_is_unaffected_and_a_thread_started_inside_stays_untraced(self):
        entered, release = threading.Event(), threading.Event()

        def other_request():
            entered.wait(5)
            end, _ = self.client.start_span("other.thread")
            end("OK")
            release.set()

        other = threading.Thread(target=other_request)
        other.start()
        with evalkit.untraced():
            inside = threading.Thread(target=lambda: self.client.start_span("started.inside")[0]("OK"))
            inside.start()
            entered.set()
            release.wait(5)
            inside.join(5)
        other.join(5)
        self.assertEqual(self.operations(), ["other.thread"])

    def test_a_task_started_inside_stays_untraced_after_the_block(self):
        async def run():
            gate = asyncio.Event()

            async def later():
                await gate.wait()
                self.client.start_span("task.after.block")[0]("OK")

            async with evalkit.untraced():
                task = asyncio.create_task(later())
            gate.set()
            await task
            self.client.start_span("caller.after.block")[0]("OK")

        asyncio.run(run())
        self.assertEqual(self.operations(), ["caller.after.block"])

    def test_untraced_works_before_and_without_a_client(self):
        self.assertIsInstance(evalkit.untraced(), type(evalkit.untraced()))
        self.assertEqual(evalkit.untraced(len)("abc"), 3)
        with evalkit.untraced("not a function"):
            pass


class _IgnoredRouteCase(_Captured):
    """The same app in each framework: /internal/jobs/<id> is ignored, /orders is not."""

    def setUp(self):
        super().setUp()
        self.server = _Server()
        self.addCleanup(self.server.close)
        self.configure(routes=["POST /internal/jobs/:id", "/metrics"])

    def handler_work(self, label):
        from opentelemetry import trace

        end, _ = self.client.start_span(f"{label}.span")
        end("OK")
        self.client.trace_function(f"{label}.function")(lambda: None)()
        with trace.get_tracer("app.lib").start_as_current_span(f"{label}.otel"):
            pass
        install_fake_openai_sync(self.client, OResp("hi")).create(model="gpt-4o-mini", messages=[])
        logging.getLogger("route.test").warning("%s log line", label)
        import requests

        requests.get(self.server.url(f"/{label}-call"))

    def assert_only_the_kept_request(self):
        spans = self.exported()
        servers = [e for e in spans if e.span_kind == "SERVER"]
        self.assertEqual(len(servers), 1, [e.operation for e in spans])
        self.assertIn("/orders", servers[0].attributes["http.url"])
        self.assertEqual(sorted(e.operation for e in spans if (e.operation or "").startswith("ignored")), [],
                         "nothing the ignored request did was recorded")
        kept = {e.operation for e in spans}
        self.assertTrue({"kept.span", "kept.function", "kept.otel"} <= kept, kept)
        self.assertEqual(len([e for e in spans if e.span_type == "llm_call"]), 1, "only the kept model call")
        self.assertEqual(len([e for e in spans if e.span_type == "log"]), 1, "only the kept log line")
        self.assertIsNone(self.server.traceparent_of("/ignored-call"))
        self.assertIsNotNone(self.server.traceparent_of("/kept-call"))


class TestFastAPIIgnoredRoute(_IgnoredRouteCase):
    def setUp(self):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            self.skipTest("fastapi not installed")
        super().setUp()

    def app(self):
        from fastapi import FastAPI

        app = FastAPI()
        app.add_middleware(evalkit.EvalKitMiddleware, client=self.client)
        case = self

        @app.post("/internal/jobs/{job_id}")
        def job(job_id: str):
            case.handler_work("ignored")
            return {"job": job_id}

        @app.get("/orders")
        def orders():
            case.handler_work("kept")
            return {"orders": []}

        return app

    def test_an_ignored_route_is_answered_untraced(self):
        app = self.app()
        sent = _call(app, "POST", "/internal/jobs/7")
        start = next(m for m in sent if m["type"] == "http.response.start")
        self.assertEqual(start["status"], 200)
        self.assertNotIn(b"x-trace-id", [k for k, _ in start.get("headers", [])])
        self.assertIn(b'"job":"7"', b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"))
        _call(app, "GET", "/orders")
        self.assert_only_the_kept_request()

    def test_the_method_restricts_the_match(self):
        app = self.app()
        _call(app, "GET", "/internal/jobs/7")
        servers = [e for e in self.exported() if e.span_kind == "SERVER"]
        self.assertEqual(len(servers), 1, "GET is not the ignored method; the 405 is traced")

    def test_a_callable_reads_the_route_starlette_will_match(self):
        self.configure(routes=[lambda info: info.route == "/internal/jobs/{job_id}"])
        app = self.app()
        _call(app, "POST", "/internal/jobs/7")
        _call(app, "GET", "/orders")
        self.assert_only_the_kept_request()

    def test_concurrent_requests_keep_their_own_spans(self):
        from fastapi import FastAPI

        app = FastAPI()
        app.add_middleware(evalkit.EvalKitMiddleware, client=self.client)
        case = self

        @app.post("/internal/jobs/{job_id}")
        async def job(job_id: str):
            for _ in range(3):
                await asyncio.sleep(0.01)
                case.client.start_span("ignored.step")[0]("OK")
            return {}

        @app.get("/orders")
        async def orders():
            for _ in range(3):
                await asyncio.sleep(0.01)
                case.client.start_span("kept.step")[0]("OK")
            async with evalkit.untraced():
                await asyncio.sleep(0.01)
                case.client.start_span("kept.untraced.step")[0]("OK")
            return {}

        async def request(method, path):
            scope = {
                "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
                "path": path, "raw_path": path.encode(), "root_path": "", "query_string": b"",
                "headers": [(b"host", b"testserver")], "scheme": "http", "server": ("testserver", 80),
            }
            messages = [{"type": "http.request", "body": b"", "more_body": False}]

            async def receive():
                return messages.pop(0) if messages else {"type": "http.disconnect"}

            async def send(message):
                pass

            await app(scope, receive, send)

        async def both():
            await asyncio.gather(request("POST", "/internal/jobs/1"), request("GET", "/orders"),
                                 request("POST", "/internal/jobs/2"))

        asyncio.run(both())
        spans = self.exported()
        self.assertEqual(sorted(e.operation for e in spans if e.span_kind != "SERVER"), ["kept.step"] * 3)
        self.assertEqual([e.attributes["http.url"].rsplit("/", 1)[-1] for e in spans if e.span_kind == "SERVER"],
                         ["orders"])

    def test_a_probe_is_traced_with_trace_health_checks(self):
        from fastapi import FastAPI

        app = FastAPI()
        app.add_middleware(evalkit.EvalKitMiddleware, client=self.client)
        app.get("/healthz")(lambda: {"ok": True})
        _call(app, "GET", "/healthz")
        self.assertEqual([e for e in self.exported() if e.span_kind == "SERVER"], [])
        self.configure(trace_health_checks=True)
        _call(app, "GET", "/healthz")
        self.assertEqual(len([e for e in self.exported() if e.span_kind == "SERVER"]), 1)


class TestFlaskIgnoredRoute(_IgnoredRouteCase):
    def setUp(self):
        try:
            import flask  # noqa: F401
        except ImportError:
            self.skipTest("flask not installed")
        super().setUp()

    def test_an_ignored_route_is_answered_untraced(self):
        from flask import Flask

        from evalkit.frameworks.flask import instrument_flask

        app = Flask(__name__)
        instrument_flask(app, self.client)
        case = self

        @app.post("/internal/jobs/<job_id>")
        def job(job_id):
            case.handler_work("ignored")
            return {"job": job_id}

        @app.get("/orders")
        def orders():
            case.handler_work("kept")
            return {"orders": []}

        http = app.test_client()
        response = http.post("/internal/jobs/7")
        self.assertEqual(response.get_json(), {"job": "7"})
        self.assertNotIn("X-Trace-Id", response.headers)
        http.get("/orders")
        self.assert_only_the_kept_request()

    def test_a_pattern_in_flasks_own_template_syntax(self):
        from flask import Flask

        from evalkit.frameworks.flask import instrument_flask

        self.configure(routes=["/reports/<int:report_id>/pdf"])
        app = Flask(__name__)
        instrument_flask(app, self.client)
        app.get("/reports/<int:report_id>/pdf")(lambda report_id: "pdf")
        app.test_client().get("/reports/3/pdf")
        self.assertEqual([e for e in self.exported() if e.span_kind == "SERVER"], [])


class TestDjangoIgnoredRoute(_IgnoredRouteCase):
    def setUp(self):
        try:
            import django
            from django.conf import settings
        except ImportError:
            self.skipTest("django not installed")
        if not settings.configured:
            settings.configure(
                DEBUG=False, ALLOWED_HOSTS=["*"], ROOT_URLCONF="tests.test_framework_spans", SECRET_KEY="test-only",
                MIDDLEWARE=["evalkit.frameworks.django.EvalKitDjangoMiddleware"],
            )
            django.setup()
        super().setUp()

    def test_an_ignored_route_is_answered_untraced(self):
        from django.http import JsonResponse
        from django.test import RequestFactory

        from evalkit.frameworks.django import EvalKitDjangoMiddleware

        def view(request):
            self.handler_work("ignored" if request.path.startswith("/internal/") else "kept")
            return JsonResponse({"ok": True})

        middleware = EvalKitDjangoMiddleware(view)
        middleware._client = self.client
        response = middleware(RequestFactory().post("/internal/jobs/7"))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.has_header("X-Trace-Id"))
        middleware(RequestFactory().get("/orders"))
        self.assert_only_the_kept_request()


class TestLitestarIgnoredRoute(_IgnoredRouteCase):
    def setUp(self):
        try:
            import litestar  # noqa: F401
        except ImportError:
            self.skipTest("litestar not installed")
        super().setUp()

    def test_an_ignored_route_is_answered_untraced(self):
        from litestar import Litestar, get, post

        case = self

        @post("/internal/jobs/{job_id:str}", sync_to_thread=False)
        def job(job_id: str) -> dict:
            case.handler_work("ignored")
            return {"job": job_id}

        @get("/orders", sync_to_thread=False)
        def orders() -> dict:
            case.handler_work("kept")
            return {"orders": []}

        app = Litestar(route_handlers=[job, orders], middleware=[evalkit.create_litestar_middleware(self.client)])
        sent = _call(app, "POST", "/internal/jobs/7")
        start = next(m for m in sent if m["type"] == "http.response.start")
        self.assertEqual(start["status"], 201)
        self.assertNotIn(b"x-trace-id", [k for k, _ in start.get("headers", [])])
        _call(app, "GET", "/orders")
        self.assert_only_the_kept_request()

    def test_a_pattern_in_litestars_template_syntax(self):
        from litestar import Litestar, get

        self.configure(routes=["/reports/{report_id:int}/pdf"])

        @get("/reports/{report_id:int}/pdf", sync_to_thread=False)
        def pdf(report_id: int) -> str:
            return "pdf"

        app = Litestar(route_handlers=[pdf], middleware=[evalkit.create_litestar_middleware(self.client)])
        _call(app, "GET", "/reports/3/pdf")
        self.assertEqual([e for e in self.exported() if e.span_kind == "SERVER"], [])


class TestQuietPropagator(unittest.TestCase):
    def test_a_dropped_span_injects_nothing_and_others_inject_as_before(self):
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

        from evalkit.core import otel_provider

        quiet = otel_provider._quiet_propagator(TraceContextTextMapPropagator())
        tracer = TracerProvider().get_tracer("t")
        dropped, kept = tracer.start_span("dropped"), tracer.start_span("kept")
        otel_provider._note_suppressed(dropped)
        self.addCleanup(otel_provider._take_suppressed, dropped)
        carrier = {}
        quiet.inject(carrier, context=trace.set_span_in_context(dropped))
        self.assertEqual(carrier, {})
        quiet.inject(carrier, context=trace.set_span_in_context(kept))
        self.assertIn("traceparent", carrier)
        self.assertEqual(quiet.fields, TraceContextTextMapPropagator().fields)
        ctx = quiet.extract({"traceparent": carrier["traceparent"]})
        self.assertEqual(trace.get_current_span(ctx).get_span_context().span_id,
                         kept.get_span_context().span_id)


if __name__ == "__main__":
    unittest.main()
