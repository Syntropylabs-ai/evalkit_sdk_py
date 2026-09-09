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
        try:
            with mock.patch("evalkit.prompts.time.monotonic", side_effect=[100.0, 130.0, 170.0]):
                prompts.get("support_agent_system", api_url=url, cache_ttl=60)
                prompts.get("support_agent_system", api_url=url, cache_ttl=60)
                self.assertEqual(recv["count"], 1)
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


if __name__ == "__main__":
    unittest.main()
