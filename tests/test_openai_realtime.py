import asyncio
import types
import unittest

from tests.helpers import fresh, reset_context


def _event(**kwargs):
    return types.SimpleNamespace(**kwargs)


class FakeConnection:
    def __init__(self, script):
        self._script = script

    async def __aiter__(self):
        for e in self._script:
            yield e

    async def close(self):
        pass


class FakeConnectCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


def install_fake_realtime(client, conn):
    from evalkit.instrumentation.llm.openai_realtime import trace_openai_realtime

    class FakeRealtime:
        def connect(self, *, model=None):
            return FakeConnectCM(conn)

    class FakeOpenAI:
        class beta:
            realtime = FakeRealtime()

    return trace_openai_realtime(client, FakeOpenAI(), model="gpt-realtime")


class TestOpenAIRealtime(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _run(self, coro):
        return asyncio.run(coro)

    def test_turn_captures_prompt_completion_tokens_and_body(self):
        script = [
            _event(type="conversation.item.input_audio_transcription.completed", transcript="what is 6*7?"),
            _event(type="response.audio_transcript.delta", delta="4"),
            _event(type="response.audio_transcript.delta", delta="2"),
            _event(type="response.done", response=types.SimpleNamespace(
                output=[],
                usage=types.SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30),
            )),
        ]
        conn_cm = install_fake_realtime(self.client, FakeConnection(script))

        async def scenario():
            async with conn_cm as conn:
                async for _ in conn:
                    pass

        self._run(scenario())
        self.client.flush()

        llm = [e for e in self.captured if e.operation == "voice.turn"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].provider, "openai-realtime")
        self.assertEqual(llm[0].prompt, "what is 6*7?")
        self.assertEqual(llm[0].completion, "42")
        self.assertEqual(llm[0].tokens_in, 10)
        self.assertEqual(llm[0].tokens_out, 20)
        self.assertIn("gen_ai.response.body", llm[0].attributes)

        sessions = [e for e in self.captured if e.operation == "voice.session"]
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].trace_id, llm[0].trace_id)

    def test_turn_captures_audio_token_breakdown(self):
        script = [
            _event(type="conversation.item.input_audio_transcription.completed", transcript="what is 6*7?"),
            _event(type="response.audio_transcript.delta", delta="42"),
            _event(type="response.done", response=types.SimpleNamespace(
                output=[],
                usage=types.SimpleNamespace(
                    input_tokens=10, output_tokens=20, total_tokens=30,
                    input_token_details=types.SimpleNamespace(audio_tokens=6, cached_tokens=0),
                    output_token_details=types.SimpleNamespace(audio_tokens=20),
                ),
            )),
        ]
        conn_cm = install_fake_realtime(self.client, FakeConnection(script))

        async def scenario():
            async with conn_cm as conn:
                async for _ in conn:
                    pass

        self._run(scenario())
        self.client.flush()

        llm = [e for e in self.captured if e.operation == "voice.turn"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].tokens_in, 10)
        self.assertEqual(llm[0].tokens_out, 20)
        self.assertEqual(llm[0].attributes["gen_ai.usage.audio_input_tokens"], 6)
        self.assertEqual(llm[0].attributes["gen_ai.usage.audio_output_tokens"], 20)

    def test_audio_only_turn_with_no_transcript_emits_no_span(self):
        script = [_event(type="response.done", response=types.SimpleNamespace(output=[], usage=None))]
        conn_cm = install_fake_realtime(self.client, FakeConnection(script))

        async def scenario():
            async with conn_cm as conn:
                async for _ in conn:
                    pass

        self._run(scenario())
        self.client.flush()

        llm = [e for e in self.captured if e.operation == "voice.turn"]
        self.assertEqual(len(llm), 0)

    def test_function_call_emits_tool_span_and_event(self):
        script = [
            _event(type="response.audio_transcript.delta", delta="done"),
            _event(
                type="response.function_call_arguments.done",
                name="get_weather", arguments='{"city":"sf"}', call_id="call_1",
            ),
            _event(type="response.done", response=types.SimpleNamespace(output=[], usage=None)),
        ]
        conn_cm = install_fake_realtime(self.client, FakeConnection(script))

        async def scenario():
            async with conn_cm as conn:
                async for _ in conn:
                    pass

        self._run(scenario())
        self.client.flush()

        tools = [e for e in self.captured if e.span_type == "tool_call"]
        self.assertEqual(len(tools), 1)
        llm = [e for e in self.captured if e.operation == "voice.turn"]
        self.assertTrue(llm[0].events)
        self.assertEqual(llm[0].events[0].attributes["gen_ai.tool.name"], "get_weather")


if __name__ == "__main__":
    unittest.main(verbosity=2)
