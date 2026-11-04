import json
import types
import unittest

import evalkit
from evalkit.instrumentation.voice.session_token import MAX_TOKEN_CHARS, current_voice_session
from tests.helpers import fresh, reset_context

TOKEN = "sim_1:run_2:scn_3:0"


def twilio_start(params):
    return {
        "event": "start",
        "sequenceNumber": "1",
        "start": {
            "streamSid": "MZ1",
            "accountSid": "AC1",
            "callSid": "CA1",
            "tracks": ["inbound"],
            "customParameters": params,
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
        },
        "streamSid": "MZ1",
    }


class TestTokenReaders(unittest.TestCase):
    def test_url_query_in_every_shape_a_websocket_server_gives(self):
        self.assertEqual(evalkit.voice_session_from_url(f"wss://bot.example/ws?evalkit_session={TOKEN}&x=1"), TOKEN)
        self.assertEqual(evalkit.voice_session_from_url(f"/ws?evalkit_session={TOKEN}"), TOKEN)
        self.assertEqual(evalkit.voice_session_from_url(f"evalkit_session={TOKEN}"), TOKEN)
        self.assertEqual(evalkit.voice_session_from_url({"evalkit_session": TOKEN}), TOKEN)
        # Percent-encoded, as a client puts the colons on the wire.
        self.assertEqual(evalkit.voice_session_from_url("wss://h/ws?evalkit_session=sim_1%3Arun_2%3Ascn_3%3A0"), TOKEN)

    def test_url_query_param_can_be_renamed_by_the_target(self):
        self.assertEqual(evalkit.voice_session_from_url(f"wss://h/ws?test_call={TOKEN}", param="test_call"), TOKEN)
        self.assertIsNone(evalkit.voice_session_from_url(f"wss://h/ws?test_call={TOKEN}"))

    def test_url_without_the_token(self):
        for value in (None, "", "wss://h/ws", "wss://h/ws?evalkit_session=", {"other": "x"}, 42):
            self.assertIsNone(evalkit.voice_session_from_url(value), value)

    def test_twilio_start_parsed_text_bytes_and_inner_object(self):
        message = twilio_start({"evalkit_session": TOKEN, "from_number": "+1555"})
        self.assertEqual(evalkit.voice_session_from_twilio_start(message), TOKEN)
        self.assertEqual(evalkit.voice_session_from_twilio_start(json.dumps(message)), TOKEN)
        self.assertEqual(evalkit.voice_session_from_twilio_start(json.dumps(message).encode()), TOKEN)
        self.assertEqual(evalkit.voice_session_from_twilio_start(message["start"]), TOKEN)

    def test_twilio_messages_without_the_token(self):
        self.assertIsNone(evalkit.voice_session_from_twilio_start(twilio_start({})))
        self.assertIsNone(evalkit.voice_session_from_twilio_start({"event": "connected", "protocol": "Call"}))
        self.assertIsNone(evalkit.voice_session_from_twilio_start("not json"))
        self.assertIsNone(evalkit.voice_session_from_twilio_start(b"\x00\x01binary audio"))
        self.assertIsNone(evalkit.voice_session_from_twilio_start(None))

    def test_pcm16_start(self):
        start = {"type": "start", "session": TOKEN, "sample_rate": 16000}
        self.assertEqual(evalkit.voice_session_from_pcm16_start(start), TOKEN)
        self.assertEqual(evalkit.voice_session_from_pcm16_start(json.dumps(start)), TOKEN)
        self.assertIsNone(evalkit.voice_session_from_pcm16_start({"type": "clear"}))
        self.assertIsNone(evalkit.voice_session_from_pcm16_start({"type": "start", "sample_rate": 8000}))
        self.assertIsNone(evalkit.voice_session_from_pcm16_start({"session": TOKEN}))

    def test_start_message_of_either_format(self):
        self.assertEqual(evalkit.voice_session_from_start_message(twilio_start({"evalkit_session": TOKEN})), TOKEN)
        self.assertEqual(evalkit.voice_session_from_start_message({"type": "start", "session": TOKEN}), TOKEN)
        self.assertIsNone(evalkit.voice_session_from_start_message({"event": "media"}))

    def test_livekit_participant_attribute(self):
        participant = types.SimpleNamespace(identity="caller", attributes={"evalkit.session": TOKEN})
        self.assertEqual(evalkit.voice_session_from_livekit_participant(participant), TOKEN)
        self.assertEqual(evalkit.voice_session_from_livekit_participant({"evalkit.session": TOKEN}), TOKEN)
        self.assertIsNone(evalkit.voice_session_from_livekit_participant(types.SimpleNamespace(attributes={})))
        self.assertIsNone(evalkit.voice_session_from_livekit_participant(None))

    def test_values_that_are_not_tokens(self):
        too_long = "x" * (MAX_TOKEN_CHARS + 1)
        for bad in ("", "   ", "a\nb", "a\x00b", too_long, 7, ["x"]):
            self.assertIsNone(evalkit.voice_session_from_pcm16_start({"type": "start", "session": bad}), repr(bad)[:20])
        self.assertEqual(evalkit.voice_session_from_pcm16_start({"type": "start", "session": f"  {TOKEN} "}), TOKEN)


class TestSetVoiceSession(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def tearDown(self):
        evalkit.clear_voice_session()
        reset_context()

    def test_every_span_in_the_call_joins_the_session(self):
        handle = evalkit.set_voice_session(TOKEN)
        self.assertIsNotNone(handle)
        self.assertEqual(current_voice_session(), TOKEN)
        _tid, end, _ctx = evalkit.start_trace("handle_call")
        end_child, _c = evalkit.start_span("llm_step")
        end_child()
        end()
        evalkit.clear_voice_session(handle)
        self.assertIsNone(current_voice_session())
        _tid, end, _ctx = evalkit.start_trace("after_the_call")
        end()
        self.client.flush()
        by_op = {e.operation: e for e in self.captured}
        self.assertEqual(by_op["handle_call"].session_id, TOKEN)
        self.assertEqual(by_op["llm_step"].session_id, TOKEN)
        self.assertIsNone(by_op["after_the_call"].session_id)

    def test_clearing_keeps_the_user(self):
        evalkit.set_user(user_id="u_1")
        evalkit.set_voice_session(TOKEN)
        evalkit.clear_voice_session()
        self.assertEqual(evalkit.current_user().user_id, "u_1")
        self.assertIsNone(evalkit.current_user().session_id)

    def test_an_unusable_token_changes_nothing(self):
        with evalkit.session("conv_1"):
            self.assertIsNone(evalkit.set_voice_session(""))
            self.assertIsNone(evalkit.set_voice_session(None))
            self.assertEqual(evalkit.current_user().session_id, "conv_1")
            self.assertIsNone(current_voice_session())


if __name__ == "__main__":
    unittest.main()
