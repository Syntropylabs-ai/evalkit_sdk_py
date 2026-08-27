import json
import os
import subprocess
import sys
import unittest

from evalkit.core.redact import (
    resolve_capture_content,
    resolve_capture_sensitive_headers,
    resolve_capture_stream_bodies,
    strip_content,
)
from evalkit.core.types import TraceEnvelope
from evalkit.utils.headers import (
    capture_sensitive_headers,
    sanitize_headers,
    set_capture_sensitive_headers,
)

SENSITIVE = {
    "Authorization": "Bearer sk-proj-secret",
    "Cookie": "session=abc",
    "X-Api-Key": "key-123",
    "x-subscription-key": "tk_live_1",
    "Proxy-Authorization": "Basic zzz",
    "Content-Type": "application/json",
    "traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01",
}


class TestSensitiveHeaderCapture(unittest.TestCase):
    def setUp(self):
        self.saved = capture_sensitive_headers()

    def tearDown(self):
        set_capture_sensitive_headers(self.saved)

    def test_sensitive_headers_are_kept_by_default(self):
        set_capture_sensitive_headers(True)
        out = sanitize_headers(SENSITIVE)
        self.assertEqual(
            out.get("Authorization"), "Bearer sk-proj-secret",
            "the SDK is meant to capture everything by default so an auth failure is debuggable from the trace alone; "
            "stripping the header silently is what made 401s impossible to diagnose",
        )
        self.assertEqual(
            set(out), set(SENSITIVE),
            "no header is dropped when capture is on, otherwise the recorded request is not the request that was sent",
        )

    def test_the_opt_out_strips_every_sensitive_header(self):
        set_capture_sensitive_headers(False)
        out = sanitize_headers(SENSITIVE)
        self.assertNotIn("Authorization", out)
        self.assertNotIn("Cookie", out)
        self.assertNotIn("X-Api-Key", out)
        self.assertNotIn("x-subscription-key", out)
        self.assertNotIn("Proxy-Authorization", out)
        self.assertEqual(
            out, {"Content-Type": "application/json", "traceparent": SENSITIVE["traceparent"]},
            "opting out has to leave the harmless headers behind - traceparent especially, since dropping it would "
            "break parent linkage in the very mode a security-conscious user picks",
        )

    def test_matching_is_case_insensitive_and_anchored(self):
        set_capture_sensitive_headers(False)
        out = sanitize_headers({
            "AUTHORIZATION": "x",
            "authorization": "y",
            "x-authorization-scheme": "not-a-secret",
            "authorization-type": "also-not",
        })
        self.assertNotIn("AUTHORIZATION", out)
        self.assertNotIn("authorization", out)
        self.assertEqual(
            set(out), {"x-authorization-scheme", "authorization-type"},
            "the pattern is anchored on purpose: an unanchored match would eat descriptive headers that merely "
            "mention authorization and quietly shrink the captured request",
        )

    def test_non_dict_header_pairs_still_work(self):
        set_capture_sensitive_headers(False)
        out = sanitize_headers([("Authorization", "x"), ("Accept", "*/*")])
        self.assertEqual(
            out, {"Accept": "*/*"},
            "httpx and aiohttp hand over multidicts and plain pair sequences, so the sanitizer cannot assume .items()",
        )

    def test_empty_headers_are_tolerated(self):
        self.assertEqual(sanitize_headers(None), {})
        self.assertEqual(sanitize_headers({}), {})

    def test_capture_content_off_still_removes_captured_headers(self):
        set_capture_sensitive_headers(True)
        env = TraceEnvelope(
            trace_id="t", span_id="s", span_type="http_call", status="OK",
            operation="GET /v1/chat", service_name="svc", created_at="2026-08-26T00:00:00Z",
            attributes={
                "http.request.headers": json.dumps(sanitize_headers(SENSITIVE)),
                "http.response.headers": json.dumps({"Set-Cookie": "a=b"}),
                "http.url": "https://api.openai.com/v1/chat",
            },
        )
        strip_content(env)
        self.assertNotIn(
            "http.request.headers", env.attributes,
            "turning header capture on by default is only defensible because capture_content=False is still a total "
            "kill switch; if headers survived it, there would be no way to keep secrets out of the backend",
        )
        self.assertNotIn("http.response.headers", env.attributes)
        self.assertIn(
            "http.url", env.attributes,
            "stripping content must not strip the metadata that makes a span identifiable",
        )


class TestFlagResolution(unittest.TestCase):
    def setUp(self):
        self.saved = {
            k: os.environ.get(k) for k in (
                "EVALKIT_CAPTURE_CONTENT",
                "EVALKIT_CAPTURE_SENSITIVE_HEADERS",
                "EVALKIT_CAPTURE_STREAM_BODIES",
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
            )
        }
        for k in self.saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_all_three_flags_default_to_capturing(self):
        self.assertTrue(resolve_capture_content(None))
        self.assertTrue(
            resolve_capture_sensitive_headers(None),
            "capture-everything-by-default is the product decision; a default of False would mean the common case is "
            "the degraded one",
        )
        self.assertTrue(resolve_capture_stream_bodies(None))

    def test_an_explicit_argument_beats_the_environment(self):
        os.environ["EVALKIT_CAPTURE_SENSITIVE_HEADERS"] = "1"
        os.environ["EVALKIT_CAPTURE_STREAM_BODIES"] = "1"
        self.assertFalse(
            resolve_capture_sensitive_headers(False),
            "code in init() is more specific than a deploy-wide env var, so passing the option must win or the "
            "argument is decorative",
        )
        self.assertFalse(resolve_capture_stream_bodies(False))

    def test_the_environment_can_turn_each_flag_off(self):
        for value in ("0", "false", "FALSE", "no", "off"):
            os.environ["EVALKIT_CAPTURE_SENSITIVE_HEADERS"] = value
            os.environ["EVALKIT_CAPTURE_STREAM_BODIES"] = value
            self.assertFalse(
                resolve_capture_sensitive_headers(None),
                f"{value!r} must read as off; an operator who cannot disable header capture without a code change has "
                "no opt-out at all",
            )
            self.assertFalse(resolve_capture_stream_bodies(None), f"{value!r} must read as off")

    def test_the_environment_can_turn_each_flag_on(self):
        for value in ("1", "true", "TRUE", "yes", "on"):
            os.environ["EVALKIT_CAPTURE_SENSITIVE_HEADERS"] = value
            os.environ["EVALKIT_CAPTURE_STREAM_BODIES"] = value
            self.assertTrue(resolve_capture_sensitive_headers(None), f"{value!r} must read as on")
            self.assertTrue(resolve_capture_stream_bodies(None), f"{value!r} must read as on")

    def test_an_unparseable_value_falls_back_to_the_default(self):
        os.environ["EVALKIT_CAPTURE_STREAM_BODIES"] = "maybe"
        self.assertTrue(
            resolve_capture_stream_bodies(None),
            "a typo in a deploy env var must not silently disable capture; it warns and keeps the default",
        )

    def test_the_three_flags_are_independent(self):
        os.environ["EVALKIT_CAPTURE_SENSITIVE_HEADERS"] = "0"
        self.assertTrue(
            resolve_capture_content(None),
            "sharing one resolver between the flags must not share the env var; turning off header capture cannot "
            "take prompts and completions down with it",
        )
        self.assertTrue(resolve_capture_stream_bodies(None))


class TestOptionsPlumbing(unittest.TestCase):
    def test_the_options_dataclass_defaults_to_capturing(self):
        from evalkit.core.types import EvalKitOptions

        opts = EvalKitOptions(subscription_key="tk", base_url="http://x")
        self.assertTrue(opts.capture_content)
        self.assertTrue(opts.capture_sensitive_headers)
        self.assertTrue(opts.capture_stream_bodies)

    def test_init_accepts_and_forwards_both_new_flags(self):
        import inspect

        import evalkit

        params = inspect.signature(evalkit.init).parameters
        for name in ("capture_sensitive_headers", "capture_stream_bodies"):
            self.assertIn(
                name, params,
                f"{name} is the documented opt-out, so it has to be a real init() keyword and not only an env var",
            )
            self.assertIsNone(
                params[name].default,
                f"{name} must default to None so resolve_* can tell 'not specified' from an explicit False and let "
                "the env var have its turn",
            )


_PROBE = """
import evalkit
from evalkit.utils.headers import capture_sensitive_headers, sanitize_headers
from evalkit.instrumentation.http import requests as rq
from evalkit.instrumentation.http import httpx as hx
from evalkit.instrumentation.http import aiohttp as ah

evalkit.init(subscription_key="tk_test", base_url="http://127.0.0.1:1", service_name="probe",
             function_tracing=False, {kwargs})
print(repr([
    capture_sensitive_headers(),
    "Authorization" in sanitize_headers({{"Authorization": "Bearer x"}}),
    rq._STREAM_BODIES,
    hx._STREAM_BODIES,
    ah._STREAM_BODIES,
]))
"""


def _probe(kwargs="", env=None):
    environ = dict(os.environ)
    environ.pop("EVALKIT_CAPTURE_SENSITIVE_HEADERS", None)
    environ.pop("EVALKIT_CAPTURE_STREAM_BODIES", None)
    environ.update(env or {})
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(kwargs=kwargs)],
        capture_output=True, text=True, env=environ, cwd=os.path.dirname(os.path.dirname(__file__)),
    )
    if out.returncode != 0:
        raise AssertionError(f"probe failed: {out.stderr}")
    return eval(out.stdout.strip().splitlines()[-1])


class TestInitWiring(unittest.TestCase):
    def test_a_default_init_turns_both_captures_on(self):
        keep, kept_header, rq, hx, ah = _probe()
        self.assertTrue(keep)
        self.assertTrue(
            kept_header,
            "the module global in headers.py is the only thing sanitize_headers reads, so init has to write it; "
            "leaving the write out would make the option silently do nothing",
        )
        self.assertTrue(rq and hx and ah)

    def test_the_env_var_reaches_the_header_global_through_init(self):
        keep, kept_header, _, _, _ = _probe(env={"EVALKIT_CAPTURE_SENSITIVE_HEADERS": "0"})
        self.assertFalse(keep)
        self.assertFalse(
            kept_header,
            "env var to resolver to options to set_capture_sensitive_headers is a four-link chain; any broken link "
            "leaves an operator unable to keep bearer tokens out of the backend",
        )

    def test_an_explicit_init_argument_reaches_the_header_global(self):
        keep, kept_header, _, _, _ = _probe(kwargs="capture_sensitive_headers=False")
        self.assertFalse(keep)
        self.assertFalse(kept_header)

    def test_the_env_var_reaches_all_three_http_modules(self):
        _, _, rq, hx, ah = _probe(env={"EVALKIT_CAPTURE_STREAM_BODIES": "0"})
        self.assertFalse(
            rq, "patch_requests must copy the option into its module global, which is what the emit branch reads")
        self.assertFalse(hx, "patch_httpx must copy the option too")
        self.assertFalse(
            ah,
            "aiohttp defers emission unconditionally when the tee is on, so it is the module where a missed opt-out "
            "would change span timing for every single call",
        )

    def test_an_explicit_init_argument_reaches_all_three_http_modules(self):
        _, _, rq, hx, ah = _probe(kwargs="capture_stream_bodies=False")
        self.assertFalse(rq)
        self.assertFalse(hx)
        self.assertFalse(ah)


if __name__ == "__main__":
    unittest.main()
