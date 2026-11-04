from .support import require

require("numpy")

import numpy as np

from evalkit.voicetest.transport import PlaybackBuffer

RATE = 8000
F = 160  # 20 ms


def tone(n, start=1):
    return (np.arange(start, start + n) % 30000 + 1).astype(np.int16)


def drain(buf, until, step=F):
    out, marks = [], []
    pos = buf.head
    while pos < until:
        pos = min(until, pos + step)
        pcm, reached = buf.render(pos)
        out.append(pcm)
        marks += reached
    return np.concatenate(out) if out else np.zeros(0, np.int16), marks


def test_burst_drains_at_exactly_1x():
    buf = PlaybackBuffer(RATE)
    audio = tone(RATE)  # 1 s arrives at once
    s = buf.push(audio, arrival=0)
    assert (s.start, s.end, s.queued) == (0, RATE, RATE)
    depths = []
    for k in range(1, 51):
        buf.render(k * F)
        depths.append(buf.queued_samples())
    assert depths[:3] == [RATE - F, RATE - 2 * F, RATE - 3 * F]
    assert depths[49] == 0
    buf2 = PlaybackBuffer(RATE)
    buf2.push(audio, arrival=0)
    heard, _ = drain(buf2, RATE + 4 * F)
    assert np.array_equal(heard[:RATE], audio)
    assert np.all(heard[RATE:] == 0)


def test_many_chunks_in_a_burst_play_back_to_back():
    buf = PlaybackBuffer(RATE)
    chunks = [tone(400, start=i * 400) for i in range(10)]
    for i, c in enumerate(chunks):
        s = buf.push(c, arrival=3 + i)  # all within the first ms
        assert s.start == 3 + i * 400
    heard, _ = drain(buf, 4400)
    assert np.all(heard[:3] == 0)
    assert np.array_equal(heard[3:4003], np.concatenate(chunks))


def test_chunk_after_an_underrun_starts_at_its_arrival():
    buf = PlaybackBuffer(RATE)
    buf.push(tone(F), arrival=0)
    drain(buf, 2 * F)
    s = buf.push(tone(F), arrival=1000)
    assert s.start == 1000 and s.queued == F
    heard, _ = drain(buf, 1400)
    assert np.all(heard[: 1000 - 2 * F] == 0)
    assert np.all(heard[1000 - 2 * F : 1000 - 2 * F + F] != 0)


def test_arrival_in_the_rendered_past_is_clamped():
    buf = PlaybackBuffer(RATE)
    drain(buf, 5 * F)
    s = buf.push(tone(10), arrival=100)
    assert s.start == 5 * F


def test_clear_drops_only_what_is_still_queued():
    buf = PlaybackBuffer(RATE)
    audio = tone(RATE)
    buf.push(audio, arrival=0)
    played, _ = drain(buf, 2400)  # 300 ms heard
    dropped, released = buf.clear(arrival=2400 + 37)  # arrives mid-frame, not yet rendered
    assert dropped == RATE - 2437 and released == []
    rest, _ = drain(buf, RATE + F)
    heard = np.concatenate([played, rest])
    assert np.array_equal(heard[:2437], audio[:2437])
    assert np.all(heard[2437:] == 0)
    assert buf.queued_samples() == 0
    s = buf.push(tone(F), arrival=9000)  # new audio after a clear starts at its arrival
    assert s.start == 9000


def test_clear_with_nothing_queued_drops_nothing():
    buf = PlaybackBuffer(RATE)
    assert buf.clear(arrival=500) == (0, [])


def test_marks_are_reached_when_playback_gets_there():
    buf = PlaybackBuffer(RATE)
    buf.push(tone(1000), arrival=10)
    assert buf.push_mark("a", arrival=11) == 1010
    buf.push(tone(500), arrival=12)
    assert buf.push_mark("b", arrival=13) == 1510
    _, marks = drain(buf, 1000)
    assert marks == []
    _, marks = drain(buf, 2000)
    assert [(m.name, m.position, m.cleared) for m in marks] == [("a", 1010, False), ("b", 1510, False)]


def test_mark_on_an_empty_queue_is_reached_at_once():
    buf = PlaybackBuffer(RATE)
    drain(buf, 2 * F)
    assert buf.push_mark("now", arrival=400) == 400
    _, marks = drain(buf, 3 * F)
    assert [(m.name, m.position) for m in marks] == [("now", 400)]


def test_clear_releases_marks_behind_dropped_audio():
    buf = PlaybackBuffer(RATE)
    buf.push(tone(4000), arrival=0)
    buf.push_mark("end", arrival=1)
    drain(buf, 800)
    dropped, released = buf.clear(arrival=900)
    assert dropped == 3100
    assert [(m.name, m.position, m.cleared) for m in released] == [("end", 900, True)]
    _, marks = drain(buf, 5000)
    assert marks == []
