import base64
import json

from .support import require

require("numpy")

import numpy as np
import pytest

from evalkit.voicetest.transport import BotAudio, BotClear, BotMark, BotOther, Target
from evalkit.voicetest.transport.pcm16 import Pcm16Transport
from evalkit.voicetest.transport.twilio import (
    caller_media_message,
    parse_bot_message,
    start_message,
    stop_message,
)


def test_target_validation():
    with pytest.raises(ValueError):
        Target(url="http://x", format="pcm16")
    with pytest.raises(ValueError):
        Target(url="ws://x", format="sip")
    with pytest.raises(ValueError):
        Target(url="ws://x", format="pcm16", sample_rate=44100)
    with pytest.raises(ValueError):
        Target(url="ws://x", format="pcm16", headers={f"h{i}": "v" for i in range(11)})
    assert Target(url="wss://x", format="twilio", sample_rate=16000).sample_rate == 8000


def test_session_goes_into_the_query():
    t = Target(url="ws://h:1/path?a=1&evalkit_session=old", format="pcm16")
    url = t.url_with_session("sim:run:sc:3")
    assert url.startswith("ws://h:1/path?")
    assert "a=1" in url and "evalkit_session=sim%3Arun%3Asc%3A3" in url and "old" not in url
    t2 = Target(url="ws://h/p", format="pcm16", session_query_param="sid")
    assert t2.url_with_session("x") == "ws://h/p?sid=x"


def test_describe_never_carries_secrets():
    t = Target(
        url="wss://user:pw@h/p?token=s3cret&evalkit_session=tok",
        format="pcm16",
        headers={"Authorization": "Bearer s3cret"},
    )
    d = json.dumps(t.describe())
    assert "s3cret" not in d and "pw" not in d
    assert t.describe()["header_names"] == ["Authorization"]
    assert "evalkit_session=tok" in t.describe()["url"]


def test_twilio_caller_messages():
    start = start_message(1, "MZ1", "AC1", "CA1", "tok")
    assert start["start"]["customParameters"] == {"evalkit_session": "tok"}
    assert start["start"]["mediaFormat"] == {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}
    assert start["start"]["tracks"] == ["inbound"] and start["sequenceNumber"] == "1"
    media = caller_media_message(2, 1, 0, b"\xff" * 160, "MZ1")
    assert media["media"]["track"] == "inbound"
    assert (media["sequenceNumber"], media["media"]["chunk"], media["media"]["timestamp"]) == ("2", "1", "0")
    assert len(base64.b64decode(media["media"]["payload"])) == 160
    stop = stop_message(9, "MZ1", "AC1", "CA1")
    assert stop["stop"] == {"accountSid": "AC1", "callSid": "CA1"} and stop["sequenceNumber"] == "9"


def test_twilio_parses_bot_messages():
    payload = base64.b64encode(b"\xff" * 80).decode()
    ev = parse_bot_message(json.dumps({"event": "media", "streamSid": "MZ", "media": {"payload": payload}}))
    assert isinstance(ev, BotAudio) and ev.nbytes == 80 and len(ev.pcm) == 80 and not ev.pcm.any()
    assert isinstance(parse_bot_message('{"event":"clear","streamSid":"MZ"}'), BotClear)
    mark = parse_bot_message('{"event":"mark","streamSid":"MZ","mark":{"name":"m1"}}')
    assert isinstance(mark, BotMark) and mark.name == "m1"
    for bad in ('{"event":"hello"}', "not json", '{"event":"media","media":{"payload":"%%"}}', b"\x00"):
        assert isinstance(parse_bot_message(bad), BotOther)


def test_pcm16_parses_bot_frames():
    t = Pcm16Transport(Target(url="ws://x", format="pcm16"), "tok")
    x = np.array([1, -2, 300, -32768], dtype="<i2").tobytes()
    a = t.parse(x[:3])  # a sample split across frames is carried over
    b = t.parse(x[3:])
    assert list(a.pcm) == [1] and list(b.pcm) == [-2, 300, -32768]
    assert (a.nbytes, b.nbytes) == (3, 5)
    assert isinstance(t.parse('{"type":"clear"}'), BotClear)
    assert isinstance(t.parse('{"type":"other"}'), BotOther)
    assert isinstance(t.parse("[]"), BotOther)
