import types
import unittest

from tests.helpers import fresh, reset_context


def _transcript(text=None, audio_duration=None, error=None):
    return types.SimpleNamespace(text=text, audio_duration=audio_duration, error=error)


class TestAssemblyAI(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_transcribe_emits_llm_call_span_with_duration(self):
        from evalkit.instrumentation.llm.assemblyai import _patch_transcribe

        class FakeTranscriber:
            def transcribe(self, data, config=None, **kwargs):
                return _transcript(text="hello world", audio_duration=281)

        _patch_transcribe(self.client, FakeTranscriber)
        result = FakeTranscriber().transcribe("https://example.com/some-long-file-name-audio.mp3")
        self.client.flush()

        self.assertEqual(result.text, "hello world")
        span = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(span.operation, "audio.transcription")
        self.assertEqual(span.provider, "assemblyai")
        self.assertEqual(span.completion, "hello world")
        self.assertEqual(span.attributes["gen_ai.audio.duration_seconds"], 281)

    def test_transcribe_never_uses_audio_url_length_for_cost(self):
        # Regression guard for the exact bug found in OpenLit's own AssemblyAI
        # instrumentation: passing the audio_url as a character-billed "prompt"
        # instead of using audio_duration.
        from evalkit.instrumentation.llm.assemblyai import _patch_transcribe

        long_url = "https://cdn.example.com/" + ("a" * 500) + ".mp3"

        class FakeTranscriber:
            def transcribe(self, data, config=None, **kwargs):
                return _transcript(text="short", audio_duration=5)

        _patch_transcribe(self.client, FakeTranscriber)
        FakeTranscriber().transcribe(long_url)
        self.client.flush()

        span = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(span.attributes["gen_ai.audio.duration_seconds"], 5)
        self.assertNotIn("gen_ai.audio.input_characters", span.attributes)

    def test_transcribe_failure_without_exception_emits_error_span(self):
        # AssemblyAI doesn't raise on a failed job -- it returns a Transcript
        # with .error set instead.
        from evalkit.instrumentation.llm.assemblyai import _patch_transcribe

        class FakeTranscriber:
            def transcribe(self, data, config=None, **kwargs):
                return _transcript(text=None, audio_duration=None, error="invalid audio file")

        _patch_transcribe(self.client, FakeTranscriber)
        result = FakeTranscriber().transcribe("https://example.com/bad.mp3")
        self.client.flush()

        self.assertEqual(result.error, "invalid audio file")
        span = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(span.status, "ERROR")
        self.assertIn("invalid audio file", span.status_message)

    def test_transcribe_raising_exception_emits_error_span(self):
        from evalkit.instrumentation.llm.assemblyai import _patch_transcribe

        class FakeTranscriber:
            def transcribe(self, data, config=None, **kwargs):
                raise RuntimeError("network error")

        _patch_transcribe(self.client, FakeTranscriber)
        with self.assertRaises(RuntimeError):
            FakeTranscriber().transcribe("https://example.com/x.mp3")
        self.client.flush()
        span = [e for e in self.captured if e.span_type == "llm_call"][0]
        self.assertEqual(span.status, "ERROR")


if __name__ == "__main__":
    unittest.main(verbosity=2)
