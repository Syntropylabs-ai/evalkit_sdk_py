import json
import os
import subprocess
import sys
import unittest

from evalkit.core.redact import (
    resolve_capture_content,
    resolve_capture_secrets,
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

    def test_sensitive_header_values_are_masked_by_default(self):
        set_capture_sensitive_headers(False)
        out = sanitize_headers(SENSITIVE)
        self.assertEqual(
            out.get("Authorization"), "***",
            "a bearer token in a span attribute is a credential leaving the process, so masked is the only defensible "
            "default; the previous default shipped it to the backend in clear text",
        )
        self.assertEqual(out.get("x-subscription-key"), "***")
        self.assertEqual(out.get("Cookie"), "***")
        self.assertEqual(out.get("X-Api-Key"), "***")
        self.assertEqual(out.get("Proxy-Authorization"), "***")
        self.assertEqual(
            set(out), set(SENSITIVE),
            "masking replaces the value and keeps the key, so the recorded request still shows which headers were "
            "sent; dropping them outright is what made 401s impossible to diagnose",
        )
        self.assertEqual(
            out.get("Content-Type"), "application/json",
            "only credential headers are masked - the rest of the request is what makes the span debuggable",
        )
        self.assertEqual(
            out.get("traceparent"), SENSITIVE["traceparent"],
            "traceparent especially has to survive, since losing it breaks parent linkage in the very mode a "
            "security-conscious user picks",
        )

    def test_the_opt_in_keeps_every_sensitive_header_verbatim(self):
        set_capture_sensitive_headers(True)
        out = sanitize_headers(SENSITIVE)
        self.assertEqual(
            out, dict(SENSITIVE),
            "EVALKIT_CAPTURE_SECRETS=1 exists to debug an auth failure from the trace alone, which only works if the "
            "header is recorded exactly as it was sent",
        )

    def test_matching_is_case_insensitive_and_anchored(self):
        set_capture_sensitive_headers(False)
        out = sanitize_headers({
            "AUTHORIZATION": "x",
            "authorization": "y",
            "x-authorization-scheme": "not-a-secret",
            "authorization-type": "also-not",
        })
        self.assertEqual(out["AUTHORIZATION"], "***")
        self.assertEqual(out["authorization"], "***")
        self.assertEqual(
            {out["x-authorization-scheme"], out["authorization-type"]}, {"not-a-secret", "also-not"},
            "the pattern is anchored on purpose: an unanchored match would blank out descriptive headers that merely "
            "mention authorization and were never secret",
        )

    def test_non_dict_header_pairs_still_work(self):
        set_capture_sensitive_headers(False)
        out = sanitize_headers([("Authorization", "x"), ("Accept", "*/*")])
        self.assertEqual(
            out, {"Authorization": "***", "Accept": "*/*"},
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
            "capture_content=False is the total kill switch that has to hold even for a user who opted back in to "
            "capturing credentials; if headers survived it there would be no way to keep secrets out of the backend",
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
                "EVALKIT_CAPTURE_SECRETS",
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

    def test_content_and_stream_bodies_default_to_capturing_but_secrets_do_not(self):
        self.assertTrue(resolve_capture_content(None))
        self.assertTrue(resolve_capture_stream_bodies(None))
        self.assertFalse(
            resolve_capture_secrets(None),
            "capture-everything-by-default is the product decision for content, but a credential is never the "
            "customer's data to lose, so secrets are masked unless someone opts in",
        )
        self.assertFalse(
            resolve_capture_sensitive_headers(None),
            "the old header flag is an alias of the same switch, so its default has to move with it",
        )

    def test_either_env_var_opts_back_in(self):
        os.environ["EVALKIT_CAPTURE_SECRETS"] = "1"
        self.assertTrue(resolve_capture_secrets(None))
        del os.environ["EVALKIT_CAPTURE_SECRETS"]
        os.environ["EVALKIT_CAPTURE_SENSITIVE_HEADERS"] = "1"
        self.assertTrue(
            resolve_capture_secrets(None),
            "operators already set EVALKIT_CAPTURE_SENSITIVE_HEADERS, so it stays as an alias rather than silently "
            "doing nothing after the rename",
        )

    def test_the_specific_variable_wins_when_both_are_set(self):
        os.environ["EVALKIT_CAPTURE_SECRETS"] = "0"
        os.environ["EVALKIT_CAPTURE_SENSITIVE_HEADERS"] = "1"
        self.assertFalse(
            resolve_capture_secrets(None),
            "EVALKIT_CAPTURE_SECRETS is the current name and the broader switch, so it has to beat the alias",
        )

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
    def test_the_options_dataclass_defaults_to_capturing_content_but_masking_secrets(self):
        from evalkit.core.types import EvalKitOptions

        opts = EvalKitOptions(subscription_key="tk", base_url="http://x")
        self.assertTrue(opts.capture_content)
        self.assertTrue(opts.capture_stream_bodies)
        self.assertFalse(opts.capture_secrets)
        self.assertFalse(
            opts.capture_sensitive_headers,
            "capture_sensitive_headers is now a read-only view of capture_secrets, so the two can never disagree and "
            "leave the header global out of step with the scrubber",
        )

    def test_init_accepts_and_forwards_both_new_flags(self):
        import inspect

        import evalkit

        params = inspect.signature(evalkit.init).parameters
        for name in ("capture_secrets", "capture_sensitive_headers", "capture_stream_bodies"):
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
    sanitize_headers({{"Authorization": "Bearer x"}}).get("Authorization"),
    rq._STREAM_BODIES,
    hx._STREAM_BODIES,
    ah._STREAM_BODIES,
]))
"""


def _probe(kwargs="", env=None):
    environ = dict(os.environ)
    environ.pop("EVALKIT_CAPTURE_SECRETS", None)
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
    def test_a_default_init_masks_credentials_and_keeps_stream_bodies(self):
        keep, header_value, rq, hx, ah = _probe()
        self.assertFalse(keep)
        self.assertEqual(
            header_value, "***",
            "the module global in headers.py is the only thing sanitize_headers reads, so init has to write it; "
            "leaving the write out would ship bearer tokens from a default install",
        )
        self.assertTrue(rq and hx and ah)

    def test_the_secrets_env_var_reaches_the_header_global_through_init(self):
        keep, header_value, _, _, _ = _probe(env={"EVALKIT_CAPTURE_SECRETS": "1"})
        self.assertTrue(keep)
        self.assertEqual(
            header_value, "Bearer x",
            "env var to resolver to options to set_capture_sensitive_headers is a four-link chain; any broken link "
            "leaves an operator unable to get the header back when a 401 needs diagnosing",
        )

    def test_the_legacy_env_var_still_reaches_the_header_global(self):
        keep, header_value, _, _, _ = _probe(env={"EVALKIT_CAPTURE_SENSITIVE_HEADERS": "1"})
        self.assertTrue(keep)
        self.assertEqual(header_value, "Bearer x")

    def test_an_explicit_init_argument_reaches_the_header_global(self):
        keep, header_value, _, _, _ = _probe(kwargs="capture_secrets=True")
        self.assertTrue(keep)
        self.assertEqual(header_value, "Bearer x")
        keep, header_value, _, _, _ = _probe(kwargs="capture_sensitive_headers=True")
        self.assertTrue(keep, "the old keyword is an alias, so it must still reach the global")
        self.assertEqual(header_value, "Bearer x")

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
