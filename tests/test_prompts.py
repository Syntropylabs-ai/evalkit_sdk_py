import json
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import evalkit
from evalkit import prompts
from evalkit.core.context import _current_prompt
from tests.helpers import OResp, fresh, install_fake_openai_sync, reset_context

PAYLOAD = {
    "name": "support_agent_system",
    "type": "chat",
    "version": 3,
    "semver": "0.3.0",
    "labels": ["production"],
    "template": [{"role": "system", "content": "You help {{customer}} on plan {{ plan }}."}],
    "variables": ["customer", "plan"],
    "config": {"model": "gpt-4o-mini", "temperature": 0.2},
}


def _server(payload=PAYLOAD, status=200):
    recv = {"count": 0, "paths": []}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            recv["count"] += 1
            recv["paths"].append(self.path)
            recv["key"] = self.headers.get("X-Subscription-Key")
            parsed = urllib.parse.urlparse(self.path)
            query = dict(urllib.parse.parse_qsl(parsed.query))
            body = dict(payload)
            if query.get("compile") == "true":
                variables = json.loads(query["variables"])
                body["compiled"] = [{"role": "system", "content": f"SERVER {variables['customer']}/{variables['plan']}"}]
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, recv, f"http://127.0.0.1:{srv.server_address[1]}"


class TestPrompts(unittest.TestCase):
    def setUp(self):
        reset_context()
        _current_prompt.set(None)
        prompts.clear_cache()
        self.captured = []
        self.client = fresh(self.captured)

    def test_get_fetches_with_key_and_label_and_emits_http_span(self):
        srv, recv, url = _server()
        try:
            prompt = prompts.get("support_agent_system", label="production", api_url=url)
        finally:
            srv.shutdown()
        self.assertEqual(recv["key"], "tk_test")
        self.assertEqual(recv["paths"], ["/v1/prompts/support_agent_system?label=production"])
        self.assertEqual(prompt.version, 3)
        self.assertEqual(prompt.semver, "0.3.0")
        self.assertEqual(prompt.variables, ["customer", "plan"])
        self.assertEqual(prompt.config["model"], "gpt-4o-mini")
        self.client.flush()
        http = [e for e in self.captured if (e.attributes or {}).get("http.url", "").startswith(url + "/v1/prompts")]
        self.assertEqual(len(http), 1)
        self.assertEqual(http[0].status, "OK")
        self.assertEqual(http[0].attributes["http.method"], "GET")

    def test_cache_serves_repeat_reads_until_ttl_expires(self):
        srv, recv, url = _server()
        clock = [100.0]
        try:
            with mock.patch("evalkit.prompts.time.monotonic", side_effect=lambda: clock[0]):
                prompts.get("support_agent_system", api_url=url, cache_ttl=60)
                clock[0] = 130.0
                prompts.get("support_agent_system", api_url=url, cache_ttl=60)
                self.assertEqual(recv["count"], 1)
                clock[0] = 170.0
                prompts.get("support_agent_system", api_url=url, cache_ttl=60)
                self.assertEqual(recv["count"], 2)
        finally:
            srv.shutdown()

    def test_version_read_is_cached_by_version_too(self):
        srv, recv, url = _server()
        try:
            prompts.get("support_agent_system", version=3, api_url=url)
            prompts.get("support_agent_system", version=3, api_url=url)
        finally:
            srv.shutdown()
        self.assertEqual(recv["count"], 1)
        self.assertIn("version=3", recv["paths"][0])

    def test_label_and_version_together_is_an_error(self):
        with self.assertRaises(ValueError):
            prompts.get("x", label="production", version=2)

    def test_local_compile_renders_chat_and_reports_missing(self):
        prompt = prompts.Prompt(PAYLOAD)
        rendered = prompt.compile({"customer": "Ada", "plan": "pro"})
        self.assertEqual(rendered, [{"role": "system", "content": "You help Ada on plan pro."}])
        with self.assertRaises(ValueError) as err:
            prompt.compile({"customer": "Ada"})
        self.assertIn("plan", str(err.exception))
        self.assertNotIn("customer", str(err.exception))

    def test_text_compile_renders_numbers_booleans_and_objects(self):
        prompt = prompts.Prompt({**PAYLOAD, "type": "text", "template": "n={{n}} ok={{ok}} obj={{obj}}"})
        self.assertEqual(prompt.compile({"n": 3, "ok": True, "obj": {"a": 1}}), 'n=3 ok=true obj={"a": 1}')

    def test_server_compiled_is_preferred_and_not_cached(self):
        srv, recv, url = _server()
        try:
            compiled = prompts.get("support_agent_system", variables={"customer": "Ada", "plan": "pro"}, api_url=url)
            plain = prompts.get("support_agent_system", api_url=url)
        finally:
            srv.shutdown()
        self.assertEqual(compiled.compiled[0]["content"], "SERVER Ada/pro")
        self.assertEqual(compiled.compile(), compiled.compiled)
        self.assertIsNone(plain.compiled)
        self.assertEqual(recv["count"], 1)

    def test_stamped_context_marks_the_next_llm_span_only(self):
        prompt = prompts.Prompt(PAYLOAD)
        fc = install_fake_openai_sync(self.client, OResp("hello"))
        with prompt.stamped():
            fc.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
            end, _ = self.client.start_span("helper", {"evalkit.span_type": "function_call"})
            end("OK")
        fc.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "later"}])
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"]
        self.assertEqual(len(llm), 2)
        self.assertEqual(llm[0].attributes["prompt.ref"], "support_agent_system@3")
        self.assertEqual(llm[0].attributes["prompt.name"], "support_agent_system")
        self.assertEqual(llm[0].attributes["prompt.version"], 3)
        self.assertNotIn("prompt.ref", llm[1].attributes or {})
        helper = [e for e in self.captured if e.operation == "helper"][0]
        self.assertNotIn("prompt.ref", helper.attributes or {})

    def test_stamp_and_unstamp(self):
        prompt = prompts.Prompt(PAYLOAD)
        prompt.stamp()
        self.assertEqual(_current_prompt.get().ref, "support_agent_system@3")
        prompt.unstamp()
        self.assertIsNone(_current_prompt.get())

    def test_explicit_prompt_attributes_win_over_the_ambient_stamp(self):
        prompt = prompts.Prompt(PAYLOAD)
        with prompt.stamped():
            end, _ = self.client.start_span("call", {"evalkit.span_type": "llm_call", "prompt.ref": "other@1"})
            end("OK")
        self.client.flush()
        llm = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(llm.attributes["prompt.ref"], "other@1")

    def test_hyperparameters_match_the_run_convention(self):
        prompt = prompts.Prompt(PAYLOAD)
        self.assertEqual(prompt.hyperparameters(), {"promptName": "support_agent_system", "promptVersion": 3})
        self.assertEqual(evalkit.Prompt, prompts.Prompt)


def _scripted_server(answers, delay=0.0):
    """A prompt API that answers each GET with the next (status, body) of `answers`, the last one
    repeating, after `delay` seconds; a body of None sends PAYLOAD."""
    import time as _time

    recv = {"count": 0, "paths": []}
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            with lock:
                recv["count"] += 1
                recv["paths"].append(self.path)
                status, body = answers[min(recv["count"], len(answers)) - 1]
            if delay:
                _time.sleep(delay)
            if callable(body):
                status, body = body(self.path)
            raw = json.dumps(PAYLOAD if body is None else body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, recv, f"http://127.0.0.1:{srv.server_address[1]}"


class TestPromptCacheResilience(unittest.TestCase):
    def setUp(self):
        reset_context()
        _current_prompt.set(None)
        prompts.clear_cache()
        self.captured = []
        self.client = fresh(self.captured)
        self.clock = [1000.0]
        patcher = mock.patch("evalkit.prompts.time.monotonic", side_effect=lambda: self.clock[0])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_failed_refresh_serves_the_cached_copy_and_waits_before_trying_again(self):
        srv, recv, url = _scripted_server([(200, None), (503, {"message": "down"})])
        try:
            first = prompts.get("support_agent_system", api_url=url, cache_ttl=60)
            self.clock[0] += 61
            with self.assertLogs("evalkit", level="WARNING") as logs:
                stale = prompts.get("support_agent_system", api_url=url, cache_ttl=60)
            self.assertEqual(stale.version, first.version)
            self.assertIn("serving the cached version 3", logs.output[0])
            self.assertEqual(recv["count"], 2)
            # Inside the retry window the cached copy is served without another request.
            self.clock[0] += 5
            prompts.get("support_agent_system", api_url=url, cache_ttl=60)
            self.assertEqual(recv["count"], 2)
            self.clock[0] += 6
            with self.assertLogs("evalkit", level="WARNING"):
                prompts.get("support_agent_system", api_url=url, cache_ttl=60)
            self.assertEqual(recv["count"], 3)
        finally:
            srv.shutdown()

    def test_no_answer_at_all_also_serves_the_cached_copy(self):
        srv, recv, url = _scripted_server([(200, None)])
        prompts.get("support_agent_system", api_url=url, cache_ttl=60)
        srv.shutdown()
        srv.server_close()
        self.clock[0] += 61
        with self.assertLogs("evalkit", level="WARNING"):
            prompt = prompts.get("support_agent_system", api_url=url, cache_ttl=60, timeout=2)
        self.assertEqual(prompt.version, 3)

    def test_a_refusal_is_raised_even_with_a_cached_copy(self):
        srv, recv, url = _scripted_server([(200, None), (404, {"message": "Prompt not found"})])
        try:
            prompts.get("support_agent_system", api_url=url, cache_ttl=60)
            self.clock[0] += 61
            with self.assertRaises(evalkit.ApiError) as err:
                prompts.get("support_agent_system", api_url=url, cache_ttl=60)
            self.assertEqual(err.exception.code, 404)
        finally:
            srv.shutdown()

    def test_the_cached_copy_is_served_only_within_stale_if_error(self):
        srv, recv, url = _scripted_server([(200, None), (503, {"message": "down"})])
        try:
            prompts.get("support_agent_system", api_url=url, cache_ttl=60, stale_if_error=100)
            self.clock[0] += 161
            with self.assertRaises(evalkit.ApiError):
                prompts.get("support_agent_system", api_url=url, cache_ttl=60, stale_if_error=100)
        finally:
            srv.shutdown()

    def test_stale_if_error_zero_turns_it_off(self):
        srv, recv, url = _scripted_server([(200, None), (503, {"message": "down"})])
        try:
            prompts.get("support_agent_system", api_url=url, cache_ttl=60, stale_if_error=0)
            self.clock[0] += 61
            with self.assertRaises(evalkit.ApiError):
                prompts.get("support_agent_system", api_url=url, cache_ttl=60, stale_if_error=0)
        finally:
            srv.shutdown()

    def test_variables_on_a_cached_copy_are_compiled_locally(self):
        srv, recv, url = _scripted_server([(200, None), (502, {"message": "bad gateway"})])
        try:
            prompts.get("support_agent_system", api_url=url, cache_ttl=60)
            self.clock[0] += 61
            with self.assertLogs("evalkit", level="WARNING"):
                prompt = prompts.get("support_agent_system", api_url=url, variables={"customer": "Ada", "plan": "pro"})
            self.assertEqual(prompt.compiled, [{"role": "system", "content": "You help Ada on plan pro."}])
        finally:
            srv.shutdown()


class TestPromptSingleFlight(unittest.TestCase):
    def setUp(self):
        reset_context()
        _current_prompt.set(None)
        prompts.clear_cache()
        self.captured = []
        self.client = fresh(self.captured)

    def _together(self, calls):
        """Run each zero-argument call in its own thread at once; return (results, errors) in order."""
        results, errors = [None] * len(calls), [None] * len(calls)
        start = threading.Barrier(len(calls))

        def run(i, call):
            start.wait()
            try:
                results[i] = call()
            except BaseException as exc:  # noqa: BLE001 - the test inspects it
                errors[i] = exc

        threads = [threading.Thread(target=run, args=(i, c)) for i, c in enumerate(calls)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        return results, errors

    def test_concurrent_reads_of_one_prompt_make_one_request(self):
        srv, recv, url = _scripted_server([(200, None)], delay=0.3)
        try:
            results, errors = self._together([lambda: prompts.get("support_agent_system", api_url=url)] * 8)
        finally:
            srv.shutdown()
        self.assertEqual(errors, [None] * 8)
        self.assertEqual(recv["count"], 1)
        self.assertEqual({r.version for r in results}, {3})

    def test_waiters_with_variables_compile_the_fetched_template(self):
        srv, recv, url = _scripted_server([(200, None)], delay=0.3)
        calls = [lambda: prompts.get("support_agent_system", api_url=url)] + [
            (lambda n=n: prompts.get("support_agent_system", api_url=url, variables={"customer": f"c{n}", "plan": "p"}))
            for n in range(4)
        ]
        try:
            results, errors = self._together(calls)
        finally:
            srv.shutdown()
        self.assertEqual(errors, [None] * 5)
        self.assertEqual(recv["count"], 1)
        compiled = sorted(r.compiled[0]["content"] for r in results if r.compiled)
        self.assertEqual(len(compiled), 4)

    def test_a_failure_is_every_waiters_answer(self):
        srv, recv, url = _scripted_server([(500, {"message": "boom"})], delay=0.3)
        try:
            _, errors = self._together([lambda: prompts.get("support_agent_system", api_url=url)] * 6)
        finally:
            srv.shutdown()
        self.assertEqual(recv["count"], 1)
        self.assertTrue(all(isinstance(e, evalkit.ApiError) and e.code == 500 for e in errors), errors)

    def test_a_refusal_of_one_callers_variables_is_not_another_callers_answer(self):
        def answer(path):
            query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(path).query))
            if query.get("compile") == "true" and "plan" not in json.loads(query["variables"]):
                return 400, {"message": "Missing variables: plan"}
            return 200, None

        srv, recv, url = _scripted_server([(200, answer)], delay=0.3)
        first_started = threading.Event()
        out = {}

        def bad():
            first_started.set()
            try:
                prompts.get("support_agent_system", api_url=url, variables={"customer": "Ada"})
            except evalkit.ApiError as exc:
                out["bad"] = exc

        def good():
            first_started.wait()
            import time as _time

            _time.sleep(0.1)
            out["good"] = prompts.get("support_agent_system", api_url=url)

        threads = [threading.Thread(target=bad), threading.Thread(target=good)]
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join(10)
        finally:
            srv.shutdown()
        self.assertEqual(out["bad"].code, 400)
        self.assertEqual(out["good"].version, 3)
        self.assertEqual(recv["count"], 2)

    def test_a_cache_ttl_of_zero_fetches_every_time(self):
        srv, recv, url = _scripted_server([(200, None)])
        try:
            prompts.get("support_agent_system", api_url=url, cache_ttl=0)
            prompts.get("support_agent_system", api_url=url, cache_ttl=0)
        finally:
            srv.shutdown()
        self.assertEqual(recv["count"], 2)


if __name__ == "__main__":
    unittest.main()
