"""The Pipecat observer, driven through a real pipeline with Pipecat's own test utilities. No speech
service runs: the frames the services and the transport would push are pushed by the test, with the
names Pipecat gives its services."""
import asyncio
import json
import time
import unittest
from datetime import datetime

import pytest

pytest.importorskip("pipecat")
pytest.importorskip("pipecat.tests.utils")

from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    InterruptionFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData  # noqa: E402
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.worker import PipelineParams, PipelineWorker  # noqa: E402
from pipecat.processors.filters.identity_filter import IdentityFilter  # noqa: E402
from pipecat.tests.utils import SleepFrame, run_test  # noqa: E402

import evalkit  # noqa: E402
from tests.helpers import fresh, reset_context  # noqa: E402

V = "evalkit.voice.v1."
TOKEN = "sim_1:run_2:scn_3:0"


def parse(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def tts(text):
    return TTSTextFrame(text, text_type="sentence")


def ttfb(processor, seconds):
    return MetricsFrame(data=[TTFBMetricsData(processor=processor, value=seconds)])


def transcript(text):
    return TranscriptionFrame(text=text, user_id="caller", timestamp=datetime.now().isoformat())


def conversation(marks):
    """A greeting, a first turn the caller cuts off, and a second turn. A generator, so each frame,
    and the wall-clock time a VAD frame stamps itself with, is made as it is sent."""
    yield BotStartedSpeakingFrame()
    yield tts("Hi, this is the booking line.")
    yield SleepFrame(sleep=0.05)
    yield BotStoppedSpeakingFrame()
    yield SleepFrame(sleep=0.05)

    marks["turn0_vad_start"] = time.time()
    yield VADUserStartedSpeakingFrame(start_secs=0.2)
    yield UserStartedSpeakingFrame()
    yield SleepFrame(sleep=0.1)
    yield transcript("I need to change my booking")
    yield VADUserStoppedSpeakingFrame(stop_secs=0.3)
    yield SleepFrame(sleep=0.05)
    yield UserStoppedSpeakingFrame()
    yield ttfb("DeepgramSTTService#0", 0.12)
    yield ttfb("OpenAILLMService#0", 0.35)
    yield ttfb("CartesiaTTSService#0", 0.08)
    yield BotStartedSpeakingFrame()
    yield tts("Sure,")
    yield tts("which booking?")
    yield ttfb("CartesiaTTSService#0", 0.4)
    yield SleepFrame(sleep=0.3)

    # The caller talks over the bot; Pipecat interrupts the reply and the bot falls silent.
    marks["turn1_vad_start"] = time.time()
    yield VADUserStartedSpeakingFrame(start_secs=0.2)
    yield UserStartedSpeakingFrame()
    yield InterruptionFrame()
    yield SleepFrame(sleep=0.05)
    marks["turn0_bot_stop"] = time.time()
    yield BotStoppedSpeakingFrame()
    yield SleepFrame(sleep=0.1)
    yield transcript("The one on Friday")
    yield VADUserStoppedSpeakingFrame(stop_secs=0.3)
    yield SleepFrame(sleep=0.05)
    yield UserStoppedSpeakingFrame()
    yield MetricsFrame(data=[ProcessingMetricsData(processor="WhisperSTTService#0", value=0.5)])
    yield ttfb("DeepgramSTTService#0", 0.09)
    yield ttfb("OpenAILLMService#0", 0.28)
    yield ttfb("CartesiaTTSService#0", 0.07)
    yield BotStartedSpeakingFrame()
    yield tts("Done, it is moved.")
    yield SleepFrame(sleep=0.1)
    marks["turn1_bot_stop"] = time.time()
    yield BotStoppedSpeakingFrame()
    yield SleepFrame(sleep=0.05)


def turns_of(captured, operation="voice.turn"):
    return sorted((e for e in captured if e.operation == operation),
                  key=lambda e: e.attributes.get(V + "turn.index", 0))


class TestPipecatObserver(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def tearDown(self):
        reset_context()

    def test_two_turns_with_an_interruption(self):
        marks = {}
        observer = evalkit.pipecat_observer(session_id=TOKEN)
        asyncio.run(run_test(IdentityFilter(), frames_to_send=conversation(marks), observers=[observer]))
        self.client.flush()

        turns = turns_of(self.captured)
        (root,) = turns_of(self.captured, "voice.session")
        self.assertEqual([t.attributes[V + "turn.index"] for t in turns], [0, 1])
        first, second = turns

        for turn in turns:
            self.assertLess(parse(turn.start_time), parse(turn.end_time), "never zero-width")
            self.assertEqual(turn.span_type, "agent")
            self.assertEqual(turn.provider, "pipecat")
            self.assertEqual(turn.parent_span_id, root.span_id)
            self.assertEqual(turn.session_id, TOKEN)
            self.assertEqual(turn.attributes[V + "session"], TOKEN)
            self.assertEqual(turn.attributes["session.id"], TOKEN)
        self.assertEqual(root.session_id, TOKEN)

        # The turn starts where the bot's VAD says the speech began (start_secs before it decided),
        # and ends when the bot falls silent.
        self.assertAlmostEqual(parse(first.start_time), marks["turn0_vad_start"] - 0.2, delta=0.05)
        self.assertAlmostEqual(parse(first.end_time), marks["turn0_bot_stop"], delta=0.05)
        self.assertAlmostEqual(parse(second.start_time), marks["turn1_vad_start"] - 0.2, delta=0.05)
        self.assertAlmostEqual(parse(second.end_time), marks["turn1_bot_stop"], delta=0.05)
        self.assertLess(parse(second.start_time), parse(first.end_time), "the caller talked over the bot")

        a, b = first.attributes, second.attributes
        self.assertTrue(a[V + "interrupted"])
        self.assertFalse(b[V + "interrupted"])
        self.assertEqual(a[V + "stt.duration_ms"], 120)
        self.assertEqual(a[V + "llm.ttft_ms"], 350)
        self.assertEqual(a[V + "tts.ttfb_ms"], 80)
        self.assertGreaterEqual(a[V + "eou.delay_ms"], 340)  # stop_secs plus the turn decision
        self.assertLess(a[V + "eou.delay_ms"], 700)
        self.assertGreater(a[V + "user.speech_end_ms"], 0)
        self.assertGreater(b[V + "user.speech_end_ms"], a[V + "user.speech_end_ms"])
        self.assertEqual(a[V + "transcript.user"], "I need to change my booking")
        self.assertEqual(a[V + "transcript.bot"], "Sure, which booking?")
        self.assertEqual(b[V + "transcript.user"], "The one on Friday")
        self.assertEqual(b[V + "transcript.bot"], "Done, it is moved.")
        # A streaming STT's TTFB is preferred to a processing time reported for the same turn.
        self.assertEqual(b[V + "stt.duration_ms"], 90)
        self.assertEqual(b[V + "llm.ttft_ms"], 280)
        self.assertEqual(b[V + "tts.ttfb_ms"], 70)
        self.assertNotIn("Hi, this is the booking line.", a[V + "transcript.bot"] + b[V + "transcript.bot"])

    def test_instrument_pipecat_on_a_worker(self):
        async def drive():
            worker = PipelineWorker(Pipeline([IdentityFilter()]), cancel_on_idle_timeout=False, enable_rtvi=False,
                                    params=PipelineParams(enable_metrics=True))
            observer = evalkit.instrument_pipecat(worker, session_id=TOKEN)
            started = asyncio.Event()

            @worker.event_handler("on_pipeline_started")
            async def _started(worker, frame):
                started.set()

            async def push():
                await asyncio.wait_for(started.wait(), timeout=2)
                for frame in (UserStartedSpeakingFrame(), transcript("hello"), UserStoppedSpeakingFrame(),
                              BotStartedSpeakingFrame(), tts("Hi there."), BotStoppedSpeakingFrame()):
                    await worker.queue_frame(frame)
                    await asyncio.sleep(0.02)
                await worker.queue_frame(EndFrame())

            from pipecat.workers.runner import WorkerRunner

            runner = WorkerRunner()
            await runner.add_workers(worker)
            await asyncio.gather(runner.run(), push())
            return observer

        observer = asyncio.run(drive())
        self.assertIsNotNone(observer)
        self.assertEqual(observer.session, TOKEN)
        self.client.flush()
        (turn,) = turns_of(self.captured)
        self.assertEqual(turn.attributes[V + "transcript.bot"], "Hi there.")
        self.assertEqual(turn.session_id, TOKEN)


class FakeWebSocket:
    """A FastAPI WebSocket as far as the token readers and parse_telephony_websocket use it."""

    def __init__(self, url="wss://bot.example/ws", messages=()):
        self.url = url
        from urllib.parse import parse_qsl, urlsplit

        self.query_params = dict(parse_qsl(urlsplit(url).query))
        self._messages = list(messages)

    def iter_text(self):
        async def gen():
            for m in self._messages:
                yield m

        return gen()


def twilio_messages(params):
    return [
        json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}),
        json.dumps({"event": "start", "sequenceNumber": "1", "streamSid": "MZ1", "start": {
            "streamSid": "MZ1", "accountSid": "AC1", "callSid": "CA1", "tracks": ["inbound"],
            "customParameters": params,
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}}}),
    ]


class TestPipecatSessionToken(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.client = fresh([])

    def tearDown(self):
        evalkit.clear_voice_session()
        reset_context()

    def test_from_the_websocket_query(self):
        ws = FakeWebSocket(f"wss://bot.example/ws?evalkit_session={TOKEN}")
        self.assertEqual(evalkit.pipecat_observer(websocket=ws).session, TOKEN)

    def test_from_the_twilio_start_message_pipecat_parsed(self):
        from pipecat.runner.utils import parse_telephony_websocket

        ws = FakeWebSocket(messages=twilio_messages({"evalkit_session": TOKEN}))
        transport_type, call_data = asyncio.run(parse_telephony_websocket(ws))
        self.assertEqual(transport_type, "twilio")
        self.assertEqual(evalkit.pipecat_observer(call_data=call_data).session, TOKEN)
        # The parse is cached on the websocket, so the websocket alone is enough too.
        self.assertEqual(evalkit.pipecat_observer(websocket=ws).session, TOKEN)

    def test_from_the_start_message_the_serializer_reads(self):
        from pipecat.serializers.twilio import TwilioFrameSerializer

        serializer = TwilioFrameSerializer("MZ1", params=TwilioFrameSerializer.InputParams(auto_hang_up=False))
        observer = evalkit.pipecat_observer(serializer=serializer)
        self.assertIsNone(observer.session)
        start = twilio_messages({"evalkit_session": TOKEN})[1]
        asyncio.run(serializer.deserialize(start))
        self.assertEqual(observer.session, TOKEN)

    def test_from_a_pcm16_start_frame(self):
        from pipecat.frames.frames import InputAudioRawFrame
        from pipecat.serializers.base_serializer import FrameSerializer

        class PCM16Serializer(FrameSerializer):
            async def serialize(self, frame):
                return None

            async def deserialize(self, data):
                if isinstance(data, bytes):
                    return InputAudioRawFrame(audio=data, sample_rate=16000, num_channels=1)
                return None

        serializer = PCM16Serializer()
        observer = evalkit.pipecat_observer(serializer=serializer)
        frame = asyncio.run(serializer.deserialize(b"\x00\x01" * 160))
        self.assertIsInstance(frame, InputAudioRawFrame)
        asyncio.run(serializer.deserialize(json.dumps({"type": "start", "session": TOKEN, "sample_rate": 16000})))
        self.assertEqual(observer.session, TOKEN)

    def test_from_set_voice_session(self):
        evalkit.set_voice_session(TOKEN)
        self.assertEqual(evalkit.pipecat_observer().session, TOKEN)


if __name__ == "__main__":
    unittest.main()
