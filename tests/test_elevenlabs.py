import asyncio
import unittest

from tests.helpers import fresh, reset_context


class TestElevenLabs(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_convert_emits_llm_call_span_with_character_count(self):
        from evalkit.instrumentation.llm.elevenlabs import _patch_convert

        class FakeTextToSpeechClient:
            def convert(self, voice_id, *, text, model_id=None, **kwargs):
                yield b"chunk1"
                yield b"chunk2"

        _patch_convert(self.client, FakeTextToSpeechClient, False)
        chunks = list(FakeTextToSpeechClient().convert("voice-123", text="Hello there, testing.", model_id="eleven_multilingual_v2"))
        self.client.flush()

        self.assertEqual(chunks, [b"chunk1", b"chunk2"])
        span = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(span.operation, "audio.speech")
        self.assertEqual(span.provider, "elevenlabs")
        self.assertEqual(span.model, "eleven_multilingual_v2")
        self.assertEqual(span.attributes["gen_ai.audio.input_characters"], len("Hello there, testing."))
        self.assertEqual(span.attributes["gen_ai.request.voice"], "voice-123")
        self.assertEqual(span.completion, "[audio generated]")

    def test_convert_error_emits_error_span(self):
        from evalkit.instrumentation.llm.elevenlabs import _patch_convert

        class FakeTextToSpeechClient:
            def convert(self, voice_id, *, text, model_id=None, **kwargs):
                raise RuntimeError("rate limited")

        _patch_convert(self.client, FakeTextToSpeechClient, False)
        with self.assertRaises(RuntimeError):
            FakeTextToSpeechClient().convert("voice-123", text="hi")
        self.client.flush()
        span = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(span.status, "ERROR")

    def test_async_convert_uses_await(self):
        from evalkit.instrumentation.llm.elevenlabs import _patch_convert

        class FakeAsyncTextToSpeechClient:
            async def convert(self, voice_id, *, text, model_id=None, **kwargs):
                async def _gen():
                    yield b"chunk"
                return _gen()

        _patch_convert(self.client, FakeAsyncTextToSpeechClient, True)

        async def scenario():
            gen = await FakeAsyncTextToSpeechClient().convert("voice-456", text="async hi", model_id="eleven_flash_v2_5")
            return [c async for c in gen]

        chunks = asyncio.run(scenario())
        self.client.flush()

        self.assertEqual(chunks, [b"chunk"])
        span = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(span.model, "eleven_flash_v2_5")
        self.assertEqual(span.attributes["gen_ai.request.voice"], "voice-456")


if __name__ == "__main__":
    unittest.main(verbosity=2)
