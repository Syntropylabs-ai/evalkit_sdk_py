"""The voice turn model both framework integrations share, and the LiveKit hooks driven by a stand-in
session. The framework tests (test_pipecat.py, test_livekit_agents.py) drive the real frameworks."""
import time
import types
import unittest
from datetime import datetime

import evalkit
from evalkit.instrumentation.voice.livekit_agents import LiveKitAgentsHooks
from evalkit.instrumentation.voice.turns import VoiceTurnRecorder
from tests.helpers import fresh, reset_context

V = "evalkit.voice.v1."
TOKEN = "sim_1:run_2:scn_3:0"
MS = 1_000_000


def parse(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def spans(captured, operation):
    return sorted((e for e in captured if e.operation == operation),
                  key=lambda e: (e.attributes or {}).get(V + "turn.index", 0))


class TwoTurnCall:
    """A call with a greeting, a first turn the user cuts off, and a second turn, in bot-side times."""

    def __init__(self, t0):
        self.t0 = t0

    def at(self, ms):
        return self.t0 + ms * MS

    def play(self, rec):
        at = self.at
        rec.bot_started(at(200))
        rec.bot_text("Hi, this is the booking line.")
        rec.bot_stopped(at(1500))
        rec.user_started(at(2000))
        rec.user_text("I need to change")
        rec.user_text("my booking")
        rec.timing(V + "stt.duration_ms", 140)
        rec.user_stopped(at(3600), speech_end_ns=at(3200))
        rec.timing(V + "llm.ttft_ms", 320)
        rec.timing(V + "tts.ttfb_ms", 90)
        rec.timing(V + "tts.ttfb_ms", 400)  # the second sentence's synthesis, not the turn's
        rec.bot_started(at(4300))
        rec.bot_text("Sure,")
        rec.bot_text("which booking?")
        rec.user_started(at(5000))  # the user talks over the bot
        rec.interrupted(at(5050))
        rec.timing(V + "llm.ttft_ms", 999)  # the cut-off reply's late figure
        rec.bot_stopped(at(5200))
        rec.user_text("The one on Friday")
        rec.user_stopped(at(6400), speech_end_ns=at(6000))
        rec.timing(V + "llm.ttft_ms", 280)
        rec.bot_started(at(7000))
        rec.bot_text("Done, it is moved.")
        rec.bot_stopped(at(8800))
        rec.close(at(9500))


class TestVoiceTurnRecorder(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self.client._opts.capture_content = True

    def tearDown(self):
        self.client._opts.capture_content = True
        evalkit.clear_voice_session()
        reset_context()

    def record(self, **kwargs):
        t0 = time.time_ns()
        rec = VoiceTurnRecorder(self.client, provider="test", started_ns=t0, **kwargs)
        TwoTurnCall(t0).play(rec)
        self.client.flush()
        return t0

    def test_two_turns_with_an_interruption(self):
        t0 = self.record(session=TOKEN)
        turns = spans(self.captured, "voice.turn")
        (root,) = spans(self.captured, "voice.session")
        self.assertEqual([t.attributes[V + "turn.index"] for t in turns], [0, 1])
        first, second = turns

        self.assertEqual(first.span_type, "agent")
        self.assertEqual(first.parent_span_id, root.span_id)
        self.assertEqual(first.trace_id, root.trace_id)
        self.assertTrue(first.attributes[V + "interrupted"])
        self.assertFalse(second.attributes[V + "interrupted"])

        # Real start and end: the user's first word to the bot's last.
        self.assertAlmostEqual(parse(first.start_time), (t0 + 2000 * MS) / 1e9, places=3)
        self.assertAlmostEqual(parse(first.end_time), (t0 + 5200 * MS) / 1e9, places=3)
        self.assertAlmostEqual(parse(second.start_time), (t0 + 5000 * MS) / 1e9, places=3)
        self.assertAlmostEqual(parse(second.end_time), (t0 + 8800 * MS) / 1e9, places=3)
        self.assertEqual(first.latency_ms, 3200)

        a = first.attributes
        self.assertEqual(a[V + "user.speech_end_ms"], 3600)
        self.assertEqual(a[V + "eou.delay_ms"], 400)
        self.assertEqual(a[V + "stt.duration_ms"], 140)
        self.assertEqual(a[V + "llm.ttft_ms"], 320)
        self.assertEqual(a[V + "tts.ttfb_ms"], 90)
        self.assertEqual(a[V + "transcript.user"], "I need to change my booking")
        self.assertEqual(a[V + "transcript.bot"], "Sure, which booking?")
        self.assertEqual(first.prompt, "I need to change my booking")
        self.assertEqual(first.completion, "Sure, which booking?")

        b = second.attributes
        self.assertEqual(b[V + "user.speech_end_ms"], 6400)
        self.assertEqual(b[V + "eou.delay_ms"], 400)
        self.assertEqual(b[V + "llm.ttft_ms"], 280)
        self.assertNotIn(V + "tts.ttfb_ms", b)
        self.assertEqual(b[V + "transcript.bot"], "Done, it is moved.")

        for span in (first, second, root):
            self.assertEqual(span.session_id, TOKEN)
            self.assertEqual(span.attributes[V + "session"], TOKEN)
            self.assertEqual(span.attributes["session.id"], TOKEN)
            self.assertEqual(span.provider, "test")
        self.assertAlmostEqual(parse(root.start_time), t0 / 1e9, places=3)
        self.assertAlmostEqual(parse(root.end_time), (t0 + 9500 * MS) / 1e9, places=3)
        self.assertEqual(root.status, "OK")

    def test_the_greeting_is_no_turn(self):
        self.record()
        texts = [t.attributes[V + "transcript.bot"] for t in spans(self.captured, "voice.turn")]
        self.assertNotIn("Hi, this is the booking line.", " ".join(texts))

    def test_token_from_set_voice_session(self):
        evalkit.set_voice_session(TOKEN)
        self.record()
        for span in self.captured:
            self.assertEqual(span.session_id, TOKEN)
        self.assertEqual(spans(self.captured, "voice.turn")[0].attributes[V + "session"], TOKEN)

    def test_without_a_token_the_ambient_session_applies(self):
        with evalkit.session("conv_9"):
            self.record()
        turn = spans(self.captured, "voice.turn")[0]
        self.assertEqual(turn.session_id, "conv_9")
        self.assertNotIn(V + "session", turn.attributes)

    def test_capture_content_off_drops_the_transcripts_and_keeps_the_timings(self):
        self.client._opts.capture_content = False
        self.record(session=TOKEN)
        first = spans(self.captured, "voice.turn")[0]
        self.assertIsNone(first.prompt)
        self.assertIsNone(first.completion)
        self.assertNotIn(V + "transcript.user", first.attributes)
        self.assertNotIn(V + "transcript.bot", first.attributes)
        self.assertEqual(first.attributes[V + "llm.ttft_ms"], 320)
        self.assertTrue(first.attributes[V + "interrupted"])
        self.assertFalse(first.attributes["evalkit.content_captured"])

    def test_mask_sees_the_transcripts(self):
        def mask(span):
            if span.attributes and V + "transcript.user" in span.attributes:
                span.attributes[V + "transcript.user"] = "<redacted>"
            return span

        self.client._opts.mask = mask
        try:
            self.record()
        finally:
            self.client._opts.mask = None
        self.assertEqual(spans(self.captured, "voice.turn")[0].attributes[V + "transcript.user"], "<redacted>")

    def test_a_turn_is_never_zero_width(self):
        t0 = time.time_ns()
        rec = VoiceTurnRecorder(self.client, provider="test", started_ns=t0)
        rec.user_started(t0 + 10 * MS)
        rec.close(t0 + 10 * MS)
        self.client.flush()
        (turn,) = spans(self.captured, "voice.turn")
        self.assertLess(parse(turn.start_time), parse(turn.end_time))

    def test_a_reply_still_playing_at_hang_up_ends_with_the_call(self):
        t0 = time.time_ns()
        rec = VoiceTurnRecorder(self.client, provider="test", started_ns=t0)
        rec.user_started(t0 + 100 * MS)
        rec.user_stopped(t0 + 900 * MS)
        rec.bot_started(t0 + 1500 * MS)
        rec.close(t0 + 2000 * MS)
        self.client.flush()
        (turn,) = spans(self.captured, "voice.turn")
        self.assertAlmostEqual(parse(turn.end_time), (t0 + 2000 * MS) / 1e9, places=3)
        self.assertFalse(turn.attributes[V + "interrupted"])

    def test_nests_under_an_evalkit_span_the_developer_opened(self):
        trace_id, end, ctx = evalkit.start_trace("call_handler")
        try:
            rec = VoiceTurnRecorder(self.client, provider="test")
            rec.close()
        finally:
            end()
        self.client.flush()
        (root,) = spans(self.captured, "voice.session")
        self.assertEqual(root.trace_id, trace_id)
        self.assertEqual(root.parent_span_id, ctx.span_id)


class FakeSession:
    """Stands in for an AgentSession's event emitter."""

    def __init__(self):
        self.handlers = {}

    def on(self, event, callback):
        self.handlers.setdefault(event, []).append(callback)
        return callback

    def off(self, event, callback):
        self.handlers.get(event, []).remove(callback)

    def emit(self, event, ev):
        for cb in list(self.handlers.get(event, [])):
            cb(ev)


def ev(**kw):
    return types.SimpleNamespace(**kw)


def message(role, text, created_at, metrics=None, interrupted=False):
    item = types.SimpleNamespace(type="message", role=role, content=[text], text_content=text,
                                 interrupted=interrupted, created_at=created_at)
    if metrics is not None:
        item.metrics = metrics
    return ev(item=item, created_at=created_at)


class TestLiveKitHooksWithAStandInSession(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def tearDown(self):
        reset_context()

    def test_turn_timings_from_the_messages(self):
        s = FakeSession()
        hooks = LiveKitAgentsHooks(self.client, s, session_id=TOKEN, legacy_metrics=False)
        self.assertNotIn("metrics_collected", s.handlers)
        t0 = time.time()
        s.emit("agent_state_changed", ev(old_state="initializing", new_state="listening", created_at=t0))
        s.emit("conversation_item_added", message("user", "I need to change my booking", t0 + 2.6, {
            "started_speaking_at": t0 + 1.0, "stopped_speaking_at": t0 + 2.0,
            "transcription_delay": 0.15, "end_of_turn_delay": 0.4}))
        s.emit("conversation_item_added", message("assistant", "Sure, which booking?", t0 + 4.0, {
            "started_speaking_at": t0 + 3.0, "stopped_speaking_at": t0 + 4.0,
            "llm_node_ttft": 0.3, "tts_node_ttfb": 0.09}, interrupted=True))
        s.emit("conversation_item_added", message("user", "The one on Friday", t0 + 5.6, {
            "started_speaking_at": t0 + 3.8, "stopped_speaking_at": t0 + 5.0,
            "transcription_delay": 0.1, "end_of_turn_delay": 0.5}))
        s.emit("conversation_item_added", message("assistant", "Done, it is moved.", t0 + 7.5, {
            "started_speaking_at": t0 + 6.2, "stopped_speaking_at": t0 + 7.5,
            "llm_node_ttft": 0.25, "tts_node_ttfb": 0.08}))
        s.emit("close", ev(error=None, reason="user_initiated", created_at=t0 + 8.0))
        self.client.flush()

        first, second = spans(self.captured, "voice.turn")
        self.assertAlmostEqual(parse(first.start_time), t0 + 1.0, places=3)
        self.assertAlmostEqual(parse(first.end_time), t0 + 4.0, places=3)
        self.assertAlmostEqual(parse(second.start_time), t0 + 3.8, places=3)
        self.assertAlmostEqual(parse(second.end_time), t0 + 7.5, places=3)
        a, b = first.attributes, second.attributes
        self.assertEqual((a[V + "turn.index"], a[V + "interrupted"]), (0, True))
        self.assertEqual((b[V + "turn.index"], b[V + "interrupted"]), (1, False))
        self.assertEqual(a[V + "user.speech_end_ms"], 2400)
        self.assertEqual(a[V + "eou.delay_ms"], 400)
        self.assertEqual(a[V + "stt.duration_ms"], 150)
        self.assertEqual(a[V + "llm.ttft_ms"], 300)
        self.assertEqual(a[V + "tts.ttfb_ms"], 90)
        self.assertEqual(b[V + "user.speech_end_ms"], 5500)
        self.assertEqual(first.provider, "livekit-agents")
        self.assertEqual(first.session_id, TOKEN)
        self.assertEqual(s.handlers.get("close"), [], "the hooks let go of the session when it closes")
        self.assertEqual(hooks.session, TOKEN)

    def test_older_versions_from_state_changes_and_metrics_collected(self):
        s = FakeSession()
        LiveKitAgentsHooks(self.client, s, session_id=TOKEN, legacy_metrics=True)
        t0 = time.time()
        metrics = lambda **kw: ev(metrics=types.SimpleNamespace(**kw))  # noqa: E731
        s.emit("agent_state_changed", ev(old_state="initializing", new_state="listening", created_at=t0))
        s.emit("user_state_changed", ev(old_state="listening", new_state="speaking", created_at=t0 + 1.0))
        s.emit("user_state_changed", ev(old_state="speaking", new_state="listening", created_at=t0 + 2.5))
        s.emit("metrics_collected", metrics(type="eou_metrics", end_of_utterance_delay=0.4, transcription_delay=0.15,
                                            on_user_turn_completed_delay=0.0, speech_id="speech_1",
                                            last_speaking_time=t0 + 2.0, timestamp=t0 + 2.4))
        s.emit("conversation_item_added", message("user", "I need to change my booking", t0 + 2.6))
        s.emit("agent_state_changed", ev(old_state="thinking", new_state="speaking", created_at=t0 + 3.0))
        s.emit("metrics_collected", metrics(type="llm_metrics", ttft=0.3, speech_id="speech_1"))
        s.emit("metrics_collected", metrics(type="tts_metrics", ttfb=0.09, speech_id="speech_1"))
        s.emit("conversation_item_added", message("assistant", "Sure, which booking?", t0 + 4.0, interrupted=True))
        s.emit("metrics_collected", metrics(type="llm_metrics", ttft=0.9, speech_id="speech_1"))
        s.emit("close", ev(error=None, created_at=t0 + 5.0))
        self.client.flush()

        (turn,) = spans(self.captured, "voice.turn")
        a = turn.attributes
        self.assertAlmostEqual(parse(turn.start_time), t0 + 1.0, places=3)
        self.assertAlmostEqual(parse(turn.end_time), t0 + 4.0, places=3)
        self.assertEqual(a[V + "user.speech_end_ms"], 2400)
        self.assertEqual(a[V + "eou.delay_ms"], 400)
        self.assertEqual(a[V + "stt.duration_ms"], 150)
        self.assertEqual(a[V + "llm.ttft_ms"], 300)
        self.assertEqual(a[V + "tts.ttfb_ms"], 90)
        self.assertTrue(a[V + "interrupted"])

    def test_a_late_figure_from_another_reply_is_kept_out(self):
        s = FakeSession()
        LiveKitAgentsHooks(self.client, s, legacy_metrics=True)
        t0 = time.time()
        metrics = lambda **kw: ev(metrics=types.SimpleNamespace(**kw))  # noqa: E731
        s.emit("metrics_collected", metrics(type="eou_metrics", end_of_utterance_delay=0.4, transcription_delay=0.1,
                                            speech_id="speech_2", last_speaking_time=t0 + 2.0, timestamp=t0 + 2.4))
        s.emit("conversation_item_added", message("user", "hello", t0 + 2.5))
        s.emit("metrics_collected", metrics(type="llm_metrics", ttft=0.7, speech_id="speech_1"))
        s.emit("metrics_collected", metrics(type="llm_metrics", ttft=0.2, speech_id="speech_2"))
        s.emit("close", ev(error=None, created_at=t0 + 3.0))
        self.client.flush()
        (turn,) = spans(self.captured, "voice.turn")
        self.assertEqual(turn.attributes[V + "llm.ttft_ms"], 200)

    def test_participant_attribute_gives_the_token(self):
        s = FakeSession()
        participant = types.SimpleNamespace(attributes={"evalkit.session": TOKEN})
        s._room_io = types.SimpleNamespace(linked_participant=participant)
        LiveKitAgentsHooks(self.client, s, legacy_metrics=False)
        t0 = time.time()
        s.emit("conversation_item_added", message("user", "hello", t0 + 1.0))
        s.emit("close", ev(error=None, created_at=t0 + 2.0))
        self.client.flush()
        (turn,) = spans(self.captured, "voice.turn")
        self.assertEqual(turn.session_id, TOKEN)

    def test_a_session_closed_by_an_error(self):
        s = FakeSession()
        LiveKitAgentsHooks(self.client, s, legacy_metrics=False)

        class STTError(Exception):
            pass

        s.emit("close", ev(error=STTError("boom"), created_at=time.time()))
        self.client.flush()
        (root,) = spans(self.captured, "voice.session")
        self.assertEqual(root.status, "ERROR")


class TestLiveKitContentOnBridgedSpans(unittest.TestCase):
    def test_lk_pii_attributes_are_content(self):
        from evalkit.core.redact import is_content_attribute

        for key in ("lk.pii.user_transcript", "lk.pii.chat_ctx", "lk.pii.response.text", "lk.pii.function_tool.arguments"):
            self.assertTrue(is_content_attribute(key), key)
        for key in ("lk.speech_id", "lk.response.ttft", "lk.interrupted", "lk.eou.endpointing_delay"):
            self.assertFalse(is_content_attribute(key), key)


if __name__ == "__main__":
    unittest.main()
