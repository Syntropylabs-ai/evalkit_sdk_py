import types
import unittest
from contextlib import asynccontextmanager

from tests.helpers import fresh, reset_context


def _msg(**kwargs):
    return types.SimpleNamespace(**kwargs)


def _server_content(**kwargs):
    defaults = dict(
        input_transcription=None, model_turn=None, output_transcription=None,
        turn_complete=False, interrupted=False,
    )
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def _usage(prompt=10, response=20, total=30, thoughts=0, prompt_tokens_details=None, response_tokens_details=None):
    return types.SimpleNamespace(
        prompt_token_count=prompt, response_token_count=response,
        total_token_count=total, thoughts_token_count=thoughts,
        prompt_tokens_details=prompt_tokens_details, response_tokens_details=response_tokens_details,
    )


def _modality_count(modality, count):
    return types.SimpleNamespace(modality=modality, token_count=count)


class FakeAsyncSession:
    def __init__(self, script):
        self._script = script
        self.sent = []

    async def send_client_content(self, **kwargs):
        self.sent.append(("send_client_content", kwargs))

    async def send_realtime_input(self, **kwargs):
        self.sent.append(("send_realtime_input", kwargs))

    async def send_tool_response(self, **kwargs):
        self.sent.append(("send_tool_response", kwargs))

    def receive(self):
        async def _gen():
            for m in self._script:
                yield m
        return _gen()

    async def close(self):
        pass


def install_fake_live(client, session):
    from evalkit.instrumentation.llm.google_live import patch_google_genai_live
    import sys

    live_mod = types.ModuleType("google.genai.live")

    class AsyncLive:
        async def connect(self, *, model=None, config=None):
            return session

    original_connect = AsyncLive.connect

    @asynccontextmanager
    async def connect_cm(self, *, model=None, config=None):
        s = await original_connect(self, model=model, config=config)
        yield s

    AsyncLive.connect = connect_cm
    live_mod.AsyncLive = AsyncLive

    genai_mod = sys.modules.get("google.genai") or types.ModuleType("google.genai")
    genai_mod.live = live_mod
    sys.modules["google.genai"] = genai_mod
    sys.modules["google.genai.live"] = live_mod

    patch_google_genai_live(client)
    return AsyncLive()


class TestGoogleLive(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _run(self, coro):
        import asyncio
        return asyncio.run(coro)

    def test_turn_captures_prompt_completion_tokens_and_body(self):
        turn_complete_msg = _msg(
            server_content=_server_content(
                output_transcription=types.SimpleNamespace(text="42"),
                turn_complete=True,
            ),
            usage_metadata=_usage(prompt=10, response=20, total=30),
            tool_call=None,
        )
        session = FakeAsyncSession([turn_complete_msg])
        live = install_fake_live(self.client, session)

        async def scenario():
            async with live.connect(model="gemini-2.0-flash-live") as s:
                await s.send_client_content(
                    turns=[{"role": "user", "parts": [{"text": "what is 6*7?"}]}],
                    turn_complete=True,
                )
                async for _ in s.receive():
                    pass

        self._run(scenario())
        self.client.flush()

        llm = [e for e in self.captured if e.span_type == "llm_call" and e.operation == "voice.turn"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].provider, "google-live")
        self.assertEqual(llm[0].model, "gemini-2.0-flash-live")
        self.assertEqual(llm[0].prompt, "user: what is 6*7?")
        self.assertEqual(llm[0].completion, "42")
        self.assertEqual(llm[0].tokens_in, 10)
        self.assertEqual(llm[0].tokens_out, 20)
        self.assertIn("gen_ai.request.body", llm[0].attributes)
        self.assertIn("gen_ai.response.body", llm[0].attributes)

        sessions = [e for e in self.captured if e.operation == "voice.session"]
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].trace_id, llm[0].trace_id)
        self.assertEqual(sessions[0].span_id, llm[0].parent_span_id)

    def test_turn_captures_audio_token_breakdown(self):
        turn_complete_msg = _msg(
            server_content=_server_content(
                output_transcription=types.SimpleNamespace(text="42"),
                turn_complete=True,
            ),
            usage_metadata=_usage(
                prompt=10, response=20, total=30,
                prompt_tokens_details=[_modality_count("TEXT", 4), _modality_count("AUDIO", 6)],
                response_tokens_details=[_modality_count("AUDIO", 20)],
            ),
            tool_call=None,
        )
        session = FakeAsyncSession([turn_complete_msg])
        live = install_fake_live(self.client, session)

        async def scenario():
            async with live.connect(model="gemini-2.0-flash-live") as s:
                await s.send_client_content(
                    turns=[{"role": "user", "parts": [{"text": "what is 6*7?"}]}],
                    turn_complete=True,
                )
                async for _ in s.receive():
                    pass

        self._run(scenario())
        self.client.flush()

        llm = [e for e in self.captured if e.span_type == "llm_call" and e.operation == "voice.turn"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].tokens_in, 10)
        self.assertEqual(llm[0].tokens_out, 20)
        self.assertEqual(llm[0].attributes["gen_ai.usage.audio_input_tokens"], 6)
        self.assertEqual(llm[0].attributes["gen_ai.usage.audio_output_tokens"], 20)

    def test_input_transcription_also_captured_as_prompt(self):
        turn_complete_msg = _msg(
            server_content=_server_content(
                input_transcription=types.SimpleNamespace(text="what is 6 times 7"),
                output_transcription=types.SimpleNamespace(text="42"),
                turn_complete=True,
            ),
            usage_metadata=None,
            tool_call=None,
        )
        session = FakeAsyncSession([turn_complete_msg])
        live = install_fake_live(self.client, session)

        async def scenario():
            async with live.connect(model="gemini-2.0-flash-live") as s:
                async for _ in s.receive():
                    pass

        self._run(scenario())
        self.client.flush()

        llm = [e for e in self.captured if e.operation == "voice.turn"]
        self.assertEqual(llm[0].prompt, "what is 6 times 7")
        self.assertEqual(llm[0].completion, "42")

    def test_thinking_captured(self):
        turn_complete_msg = _msg(
            server_content=_server_content(
                model_turn=types.SimpleNamespace(parts=[
                    types.SimpleNamespace(text="let me think...", thought=True),
                    types.SimpleNamespace(text="42", thought=False),
                ]),
                turn_complete=True,
            ),
            usage_metadata=None,
            tool_call=None,
        )
        session = FakeAsyncSession([turn_complete_msg])
        live = install_fake_live(self.client, session)

        async def scenario():
            async with live.connect(model="gemini-2.0-flash-live") as s:
                async for _ in s.receive():
                    pass

        self._run(scenario())
        self.client.flush()

        llm = [e for e in self.captured if e.operation == "voice.turn"]
        self.assertEqual(llm[0].completion, "42")
        self.assertEqual(llm[0].attributes.get("gen_ai.response.thinking"), "let me think...")

    def test_audio_only_turn_with_no_transcript_emits_no_span(self):
        turn_complete_msg = _msg(
            server_content=_server_content(turn_complete=True),
            usage_metadata=None,
            tool_call=None,
        )
        session = FakeAsyncSession([turn_complete_msg])
        live = install_fake_live(self.client, session)

        async def scenario():
            async with live.connect(model="gemini-2.0-flash-live") as s:
                async for _ in s.receive():
                    pass

        self._run(scenario())
        self.client.flush()

        llm = [e for e in self.captured if e.operation == "voice.turn"]
        self.assertEqual(len(llm), 0)

    def test_tool_call_emits_child_span_and_event(self):
        turn_complete_msg = _msg(
            server_content=_server_content(
                output_transcription=types.SimpleNamespace(text="done"),
                turn_complete=True,
            ),
            usage_metadata=None,
            tool_call=types.SimpleNamespace(function_calls=[
                types.SimpleNamespace(name="get_weather", args={"city": "sf"}),
            ]),
        )
        session = FakeAsyncSession([turn_complete_msg])
        live = install_fake_live(self.client, session)

        async def scenario():
            async with live.connect(model="gemini-2.0-flash-live") as s:
                async for _ in s.receive():
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
