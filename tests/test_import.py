import unittest
from tests.helpers import evalkit


class TestImport(unittest.TestCase):
    def test_version(self):
        from evalkit._version import SDK_VERSION
        # Keep the asserted version sourced from the single source of truth so it
        # can't drift behind pyproject.toml / _version.py again.
        self.assertEqual(evalkit.__version__, SDK_VERSION)

    def test_public_api(self):
        for name in (
            "init", "flush", "current_trace_id",
            "simulate_user", "generate_scenarios", "evaluate",
            "patch_langchain", "patch_litellm", "patch_google_genai",
            "patch_openai_client", "patch_anthropic_client",
            "patch_bedrock_client", "patch_cohere_client",
            "patch_mistral_client", "patch_claude_agent_sdk",
            "start_trace", "start_span", "start_http_trace",
            "with_trace", "trace_function",
            "EvalKitMiddleware", "EvalKitLoggingHandler",
        ):
            self.assertTrue(hasattr(evalkit, name), f"missing export: {name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
