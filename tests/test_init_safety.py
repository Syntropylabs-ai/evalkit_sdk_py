"""init() never breaks the application: a bad option or the kill switch make the SDK a no-op, an
integration that fails to install is skipped, and calling it again says what it did not apply.

Each case runs in a fresh interpreter, as init() configures the process once.
"""
import json
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PROBE = r"""
import json, logging, sys, threading
logging.basicConfig(level=logging.WARNING, format="%(levelname)s|%(name)s|%(message)s", stream=sys.stderr)
original_call_handlers = logging.Logger.callHandlers
{setup}
import evalkit
client = evalkit.init(**{kwargs})
{after}
def plain(x):
    return x
traced = client.trace_function("plain")(plain)
print(json.dumps({{
    "enabled": client.enabled,
    "threads": sorted(t.name for t in threading.enumerate() if t.name.startswith("evalkit")),
    "logging_patched": logging.Logger.callHandlers is not original_call_handlers,
    "decorator_is_identity": traced is plain,
    "flush": evalkit.flush(),
    "stats": evalkit.stats(),
    "extra": globals().get("EXTRA"),
}}))
"""


def _run(kwargs, *, setup="", after="", env=None):
    script = _PROBE.format(setup=setup, kwargs=kwargs, after=after)
    full_env = {**os.environ, "PYTHONPATH": ROOT, "LITELLM_LOCAL_MODEL_COST_MAP": "True", "EVALKIT_FUNCTION_TRACE": "false"}
    for name in ("EVALKIT_DISABLED", "OTEL_SDK_DISABLED", "OTEL_TRACES_SAMPLER", "OTEL_TRACES_SAMPLER_ARG"):
        full_env.pop(name, None)
    full_env.update(env or {})
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=full_env, timeout=120)
    if out.returncode != 0:
        raise AssertionError(out.stderr)
    return json.loads(out.stdout.strip().splitlines()[-1]), out.stderr


_OK = 'dict(subscription_key="tk_test", base_url="http://127.0.0.1:1", service_name="probe", function_tracing=False)'


def _warnings(stderr):
    return [line for line in stderr.splitlines() if line.startswith("WARNING|evalkit|")]


class InvalidConfigTest(unittest.TestCase):
    def test_a_bad_base_url_makes_the_sdk_a_noop_with_one_warning(self):
        result, stderr = _run('dict(subscription_key="tk_test", base_url="localhost:8085", function_tracing=False)')
        self.assertFalse(result["enabled"])
        self.assertEqual(result["threads"], [], "no export thread")
        self.assertFalse(result["logging_patched"], "nothing is patched")
        self.assertTrue(result["decorator_is_identity"])
        self.assertTrue(result["flush"])
        self.assertEqual(result["stats"]["circuit_state"], "disabled")
        warnings = _warnings(stderr)
        self.assertEqual(len(warnings), 1, stderr)
        self.assertIn("base_url must be an http:// or https:// URL", warnings[0])

    def test_every_problem_is_named_and_the_key_is_never_repeated(self):
        result, stderr = _run(
            'dict(subscription_key="tk_live_secret\\n_value", base_url="ftp://x", max_queue_size=0, '
            'sample_rate=2, scheduled_delay_ms="soon", function_tracing=False)'
        )
        self.assertFalse(result["enabled"])
        [warning] = _warnings(stderr)
        for problem in ("subscription_key contains control characters", "base_url", "max_queue_size",
                        "sample_rate", "scheduled_delay_ms"):
            self.assertIn(problem, warning)
        self.assertNotIn("tk_live_secret", stderr)

    def test_an_empty_key_is_a_noop(self):
        result, stderr = _run('dict(subscription_key="", function_tracing=False)')
        self.assertFalse(result["enabled"])
        self.assertIn("subscription_key is empty", stderr)


class KillSwitchTest(unittest.TestCase):
    def test_evalkit_disabled_and_otel_sdk_disabled_turn_everything_off_quietly(self):
        for name in ("EVALKIT_DISABLED", "OTEL_SDK_DISABLED"):
            with self.subTest(name=name):
                result, stderr = _run(_OK, env={name: "true"})
                self.assertFalse(result["enabled"])
                self.assertEqual(result["threads"], [])
                self.assertFalse(result["logging_patched"])
                self.assertEqual(_warnings(stderr), [])

    def test_explicit_patch_calls_do_nothing_when_disabled(self):
        result, _ = _run(_OK, env={"EVALKIT_DISABLED": "1"}, after=(
            "import types\n"
            "fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **k: 1)))\n"
            "original = fake.chat.completions.create\n"
            "evalkit.patch_openai_client(fake)\n"
            "EXTRA = fake.chat.completions.create is original\n"
        ))
        self.assertTrue(result["extra"])


class StartupTest(unittest.TestCase):
    def test_a_valid_config_starts_as_before(self):
        result, stderr = _run(_OK)
        self.assertTrue(result["enabled"])
        self.assertEqual(result["threads"], ["evalkit-export"])
        self.assertTrue(result["logging_patched"])
        self.assertEqual(_warnings(stderr), [])

    def test_an_integration_that_raises_on_import_does_not_stop_init(self):
        setup = (
            "import importlib.abc\n"
            "class Broken(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'openai' or name.startswith('openai.'):\n"
            "            raise RuntimeError('a broken install raises something other than ImportError')\n"
            "        return None\n"
            "sys.modules.pop('openai', None)\n"
            "sys.meta_path.insert(0, Broken())\n"
        )
        after = (
            "import anthropic.resources.messages as m\n"
            "EXTRA = bool(getattr(m.Messages.create, '_evalkit', False))\n"
        )
        result, stderr = _run(_OK, setup=setup, after=after)
        self.assertTrue(result["enabled"], stderr)
        self.assertTrue(result["extra"], "the integrations after the broken one are still installed")

    def test_calling_init_again_with_other_options_warns_once_and_never_shows_values(self):
        after = (
            "again = evalkit.init(subscription_key='tk_other_secret', base_url='http://127.0.0.1:1', service_name='other')\n"
            "evalkit.init(subscription_key='tk_other_secret', service_name='third')\n"
            "evalkit.init(**" + _OK + ")\n"
            "EXTRA = again is client\n"
        )
        result, stderr = _run(_OK, after=after)
        self.assertTrue(result["extra"], "the first client stays")
        warnings = _warnings(stderr)
        self.assertEqual(len(warnings), 1, stderr)
        self.assertIn("service_name", warnings[0])
        self.assertIn("subscription_key", warnings[0])
        self.assertNotIn("tk_other_secret", stderr)

    def test_sampling_keeps_model_calls_tool_calls_and_errors(self):
        after = (
            "captured = []\n"
            "client._exporter.export = lambda batch: (captured.extend(batch), True)[1]\n"
            "from evalkit.core.types import TraceEnvelope\n"
            "for i in range(400):\n"
            "    tid = __import__('secrets').token_hex(16)\n"
            "    client.enqueue(TraceEnvelope(trace_id=tid, span_id=f'{i:016x}', created_at='t', span_type='function_call'))\n"
            "    client.enqueue(TraceEnvelope(trace_id=tid, span_id=f'{i:016x}', created_at='t', span_type='llm_call'))\n"
            "    client.enqueue(TraceEnvelope(trace_id=tid, span_id=f'{i:016x}', created_at='t', span_type='log', status='ERROR'))\n"
            "client.flush()\n"
            "kinds = {}\n"
            "for e in captured:\n"
            "    kinds[e.span_type] = kinds.get(e.span_type, 0) + 1\n"
            "EXTRA = kinds\n"
        )
        result, _ = _run(_OK, after=after, env={"OTEL_TRACES_SAMPLER": "parentbased_traceidratio",
                                                "OTEL_TRACES_SAMPLER_ARG": "0.25"})
        kinds = result["extra"]
        self.assertEqual(kinds["llm_call"], 400)
        self.assertEqual(kinds["log"], 400)
        self.assertLess(kinds.get("function_call", 0), 200)
        self.assertGreater(kinds.get("function_call", 0), 30)
        self.assertEqual(result["stats"]["dropped_sampled"], 400 - kinds.get("function_call", 0))


class SamplerTest(unittest.TestCase):
    def test_the_rate_resolves_from_the_option_then_opentelemetry(self):
        from unittest import mock

        from evalkit.core.sampling import Sampler, resolve_sample_rate

        with mock.patch.dict(os.environ, {"OTEL_TRACES_SAMPLER": "always_off"}):
            self.assertEqual(resolve_sample_rate(None), 0.0)
            self.assertEqual(resolve_sample_rate(0.5), 0.5, "the option wins")
        with mock.patch.dict(os.environ, {"OTEL_TRACES_SAMPLER": "traceidratio", "OTEL_TRACES_SAMPLER_ARG": "0.1"}):
            self.assertEqual(resolve_sample_rate(None), 0.1)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OTEL_TRACES_SAMPLER", None)
            self.assertEqual(resolve_sample_rate(None), 1.0, "the default keeps everything")
        sampler = Sampler(0.5)
        trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
        self.assertEqual(sampler.trace_kept(trace_id), sampler.trace_kept(trace_id), "one answer per trace")
        self.assertTrue(Sampler(0.0).trace_kept("not-hex"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
