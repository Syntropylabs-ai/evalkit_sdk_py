import asyncio
import unittest

from openai.types.audio.transcription import Transcription, UsageDuration, UsageTokens, UsageTokensInputTokenDetails
from openai.types.audio.transcription_verbose import TranscriptionVerbose

from tests.helpers import fresh, reset_context


class TestOpenAIAudio(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_speech_create_emits_llm_call_span_with_character_count(self):
        from evalkit.instrumentation.llm.openai_audio import _patch_speech

        class FakeSpeech:
            def create(self, **kwargs):
                return b"fake-audio-bytes"

        _patch_speech(self.client, FakeSpeech, False)
        FakeSpeech().create(model="tts-1", voice="alloy", input="Hello world, this is a test.")
        self.client.flush()

        span = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(span.operation, "audio.speech")
        self.assertEqual(span.provider, "openai")
        self.assertEqual(span.model, "tts-1")
        self.assertEqual(span.attributes["gen_ai.audio.input_characters"], len("Hello world, this is a test."))
        self.assertEqual(span.attributes["gen_ai.request.voice"], "alloy")
        self.assertEqual(span.completion, "[audio generated]")

    def test_speech_create_error_emits_error_span(self):
        from evalkit.instrumentation.llm.openai_audio import _patch_speech

        class FakeSpeech:
            def create(self, **kwargs):
                raise RuntimeError("quota exceeded")

        _patch_speech(self.client, FakeSpeech, False)
        with self.assertRaises(RuntimeError):
            FakeSpeech().create(model="tts-1", voice="alloy", input="hi")
        self.client.flush()
        span = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(span.status, "ERROR")

    def test_transcription_duration_billed_whisper1(self):
        from evalkit.instrumentation.llm.openai_audio import _patch_transcribe

        class FakeTranscriptions:
            def create(self, **kwargs):
                return TranscriptionVerbose(duration=12.5, language="en", text="hello there")

        _patch_transcribe(self.client, FakeTranscriptions, "audio.transcription", False)
        f = open(__file__, "rb")
        try:
            FakeTranscriptions().create(model="whisper-1", file=f)
        finally:
            f.close()
        self.client.flush()

        span = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(span.operation, "audio.transcription")
        self.assertEqual(span.completion, "hello there")
        self.assertEqual(span.attributes["gen_ai.audio.duration_seconds"], 12.5)
        self.assertIsNone(span.tokens_in)

    def test_transcription_token_billed_gpt4o_transcribe(self):
        from evalkit.instrumentation.llm.openai_audio import _patch_transcribe

        class FakeTranscriptions:
            def create(self, **kwargs):
                return Transcription(
                    text="hello there",
                    usage=UsageTokens(
                        type="tokens", input_tokens=15, output_tokens=4, total_tokens=19,
                        input_token_details=UsageTokensInputTokenDetails(audio_tokens=12, text_tokens=3),
                    ),
                )

        _patch_transcribe(self.client, FakeTranscriptions, "audio.transcription", False)
        f = open(__file__, "rb")
        try:
            FakeTranscriptions().create(model="gpt-4o-transcribe", file=f)
        finally:
            f.close()
        self.client.flush()

        span = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(span.tokens_in, 15)
        self.assertEqual(span.tokens_out, 4)
        self.assertEqual(span.attributes["gen_ai.usage.audio_input_tokens"], 12)

    def test_transcription_duration_usage_variant(self):
        from evalkit.instrumentation.llm.openai_audio import _patch_transcribe

        class FakeTranscriptions:
            def create(self, **kwargs):
                return Transcription(text="hi", usage=UsageDuration(type="duration", seconds=3.2))

        _patch_transcribe(self.client, FakeTranscriptions, "audio.transcription", False)
        f = open(__file__, "rb")
        try:
            FakeTranscriptions().create(model="whisper-1", file=f)
        finally:
            f.close()
        self.client.flush()

        span = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(span.attributes["gen_ai.audio.duration_seconds"], 3.2)

    def test_async_transcription_uses_await_not_sync_call(self):
        from evalkit.instrumentation.llm.openai_audio import _patch_transcribe

        class FakeAsyncTranscriptions:
            async def create(self, **kwargs):
                return Transcription(text="hello async", usage=UsageDuration(type="duration", seconds=1.5))

        _patch_transcribe(self.client, FakeAsyncTranscriptions, "audio.transcription", True)
        f = open(__file__, "rb")
        try:
            asyncio.run(FakeAsyncTranscriptions().create(model="whisper-1", file=f))
        finally:
            f.close()
        self.client.flush()

        span = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(span.completion, "hello async")
        self.assertEqual(span.attributes["gen_ai.audio.duration_seconds"], 1.5)

    def test_async_speech_uses_await_not_sync_call(self):
        from evalkit.instrumentation.llm.openai_audio import _patch_speech

        class FakeAsyncSpeech:
            async def create(self, **kwargs):
                return b"fake-audio-bytes"

        _patch_speech(self.client, FakeAsyncSpeech, True)
        asyncio.run(FakeAsyncSpeech().create(model="tts-1", voice="alloy", input="hi there"))
        self.client.flush()

        span = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(span.completion, "[audio generated]")
        self.assertEqual(span.attributes["gen_ai.audio.input_characters"], len("hi there"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
