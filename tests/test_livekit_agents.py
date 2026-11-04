"""The LiveKit Agents hooks on a real AgentSession. No room, model or speech service runs: the session
emits the events and messages its activity would, as LiveKit's own types."""
import asyncio
import logging
import time
import types
import unittest
from datetime import datetime

import pytest

pytest.importorskip("livekit.agents")

from livekit.agents import AgentSession  # noqa: E402
from livekit.agents.llm import ChatMessage  # noqa: E402
from livekit.agents.voice.events import (  # noqa: E402
    AgentStateChangedEvent,
    CloseEvent,
    CloseReason,
    ConversationItemAddedEvent,
    UserStateChangedEvent,
)

import evalkit  # noqa: E402
from tests.helpers import fresh, reset_context  # noqa: E402

V = "evalkit.voice.v1."
TOKEN = "sim_1:run_2:scn_3:0"
METRICS_ON_MESSAGES = "metrics" in ChatMessage.model_fields


def parse(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def item(role, text, at, metrics=None, interrupted=False):
    kwargs = {"role": role, "content": [text], "interrupted": interrupted, "created_at": at}
    if metrics is not None and METRICS_ON_MESSAGES:
        kwargs["metrics"] = metrics
    return ConversationItemAddedEvent(item=ChatMessage(**kwargs), created_at=at)


@unittest.skipUnless(METRICS_ON_MESSAGES, "this LiveKit Agents version reports turn timings by metrics_collected")
class TestLiveKitAgentsHooks(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def tearDown(self):
        reset_context()

    def play(self, session, t0):
        session.emit("agent_state_changed", AgentStateChangedEvent(old_state="initializing", new_state="listening", created_at=t0))
        # The agent greets first: no turn.
        session.emit("agent_state_changed", AgentStateChangedEvent(old_state="listening", new_state="speaking", created_at=t0 + 0.1))
        session.emit("conversation_item_added", item("assistant", "Hi, this is the booking line.", t0 + 0.8, {
            "started_speaking_at": t0 + 0.1, "stopped_speaking_at": t0 + 0.8}))
        session.emit("user_state_changed", UserStateChangedEvent(old_state="listening", new_state="speaking", created_at=t0 + 1.0))
        session.emit("conversation_item_added", item("user", "I need to change my booking", t0 + 2.6, {
            "started_speaking_at": t0 + 1.0, "stopped_speaking_at": t0 + 2.0,
            "transcription_delay": 0.15, "end_of_turn_delay": 0.4, "on_user_turn_completed_delay": 0.01}))
        # The caller talks over the reply; LiveKit adds it as interrupted, with what was played.
        session.emit("conversation_item_added", item("assistant", "Sure, which", t0 + 4.0, {
            "started_speaking_at": t0 + 3.0, "stopped_speaking_at": t0 + 4.0,
            "llm_node_ttft": 0.3, "tts_node_ttfb": 0.09, "e2e_latency": 1.0}, interrupted=True))
        session.emit("conversation_item_added", item("user", "The one on Friday", t0 + 5.6, {
            "started_speaking_at": t0 + 3.8, "stopped_speaking_at": t0 + 5.0,
            "transcription_delay": 0.1, "end_of_turn_delay": 0.5}))
        session.emit("conversation_item_added", item("assistant", "Done, it is moved.", t0 + 7.5, {
            "started_speaking_at": t0 + 6.2, "stopped_speaking_at": t0 + 7.5,
            "llm_node_ttft": 0.25, "tts_node_ttfb": 0.08}))
        session.emit("close", CloseEvent(reason=CloseReason.USER_INITIATED, created_at=t0 + 8.0))

    def test_two_turns_with_an_interruption(self):
        async def run():
            session = AgentSession()
            hooks = evalkit.instrument_livekit_agents(session, session_id=TOKEN)
            self.play(session, t0)
            return hooks

        t0 = time.time()
        with self.assertNoLogs("livekit.agents", level=logging.WARNING):
            hooks = asyncio.run(run())
        self.assertIsNotNone(hooks)
        self.client.flush()

        turns = sorted((e for e in self.captured if e.operation == "voice.turn"),
                       key=lambda e: e.attributes[V + "turn.index"])
        (root,) = [e for e in self.captured if e.operation == "voice.session"]
        first, second = turns
        self.assertEqual([t.attributes[V + "turn.index"] for t in turns], [0, 1])
        for turn in turns:
            self.assertLess(parse(turn.start_time), parse(turn.end_time), "never zero-width")
            self.assertEqual(turn.provider, "livekit-agents")
            self.assertEqual(turn.span_type, "agent")
            self.assertEqual(turn.parent_span_id, root.span_id)
            self.assertEqual(turn.session_id, TOKEN)
            self.assertEqual(turn.attributes[V + "session"], TOKEN)
        self.assertAlmostEqual(parse(first.start_time), t0 + 1.0, places=3)
        self.assertAlmostEqual(parse(first.end_time), t0 + 4.0, places=3)
        self.assertAlmostEqual(parse(second.start_time), t0 + 3.8, places=3)
        self.assertAlmostEqual(parse(second.end_time), t0 + 7.5, places=3)
        self.assertAlmostEqual(parse(root.start_time), t0, places=3)
        self.assertAlmostEqual(parse(root.end_time), t0 + 8.0, places=3)

        a, b = first.attributes, second.attributes
        self.assertTrue(a[V + "interrupted"])
        self.assertFalse(b[V + "interrupted"])
        self.assertEqual(a[V + "user.speech_end_ms"], 2400)
        self.assertEqual(a[V + "eou.delay_ms"], 400)
        self.assertEqual(a[V + "stt.duration_ms"], 150)
        self.assertEqual(a[V + "llm.ttft_ms"], 300)
        self.assertEqual(a[V + "tts.ttfb_ms"], 90)
        self.assertEqual(a[V + "transcript.user"], "I need to change my booking")
        self.assertEqual(a[V + "transcript.bot"], "Sure, which")
        self.assertEqual(b[V + "user.speech_end_ms"], 5500)
        self.assertEqual(b[V + "eou.delay_ms"], 500)
        self.assertEqual(b[V + "llm.ttft_ms"], 250)
        self.assertEqual(b[V + "transcript.bot"], "Done, it is moved.")

    def test_token_from_the_participant(self):
        async def run():
            session = AgentSession()
            participant = types.SimpleNamespace(identity="caller", attributes={"evalkit.session": TOKEN})
            evalkit.instrument_livekit_agents(session, participant=participant)
            self.play(session, time.time())

        asyncio.run(run())
        self.client.flush()
        self.assertTrue(self.captured)
        for span in self.captured:
            self.assertEqual(span.session_id, TOKEN)

    def test_does_not_listen_to_the_deprecated_metrics_event(self):
        async def run():
            session = AgentSession()
            hooks = evalkit.instrument_livekit_agents(session)
            return [event for event, _ in hooks._listeners]

        self.assertNotIn("metrics_collected", asyncio.run(run()))


if __name__ == "__main__":
    unittest.main()
