"""The run block ``evalkit voice-test run`` prints, from ``GET /v1/voice-tests/:runId`` as the
orchestrator answers it: each evaluator's figures written out as the evaluators display them."""

from __future__ import annotations

from types import SimpleNamespace

from evalkit.voicetest.summary import figures_text, run_lines

# One call of a real run against a bot that answered after 2.5 s and talked over the caller once.
RUN = {
    "runId": "r1", "status": "completed", "passed": False, "consoleUrl": "https://console.example/r1",
    "calls": [{
        "callIndex": 0, "scenarioId": "dup-charge", "status": "reported", "error": None,
        "evaluators": [
            {"ruleId": "a", "key": "voice_response_delay", "name": "Voice Response Delay", "score": 0,
             "passed": False, "figures": {"limitMs": 1200, "p50Ms": 2504, "p95Ms": 2507, "turns": 3,
                                          "turnsOver": 3, "unanswered": 0, "unscoredTurns": []}},
            {"ruleId": "b", "key": "voice_talk_over", "name": "Voice Talk-Over", "score": 0.5, "passed": False,
             "figures": {"limitMs": 300, "maxCount": None, "talkOverMs": 700, "talkOverCount": 1,
                         "maxTurnMs": 700, "turns": 2, "turnsOver": 1, "unscoredTurns": []}},
            {"ruleId": "c", "key": "voice_interruption_stop", "name": "Voice Interruption Stop", "score": None,
             "passed": None, "figures": {"unscoredTurns": []}},
            {"ruleId": "d", "key": "voice_silence", "name": "Voice Silence", "score": 1, "passed": True,
             "figures": {"limitMs": 3000, "longestSilenceMs": 0, "turns": 3, "turnsOver": 0, "unscoredTurns": []}},
        ],
    }],
}


def test_the_run_block_shows_each_evaluators_figures_and_which_were_not_scored():
    report = SimpleNamespace(run_id="r1", calls=[], summary=RUN, problems=[], out_dir=None,
                             console_url=RUN["consoleUrl"], verdict="failed", exit_code=1)
    lines = run_lines(report)
    assert "    Voice Response Delay  0.00  failed  p95 2,507 ms, limit 1,200 ms, 3 of 3 turns over" in lines
    assert "    Voice Talk-Over  0.50  failed  1 talk-over, 700 ms in all, limit 300 ms, 1 of 2 turns over" in lines
    assert "    Voice Interruption Stop  not scored" in lines
    assert "    Voice Silence  1.00  passed  no silence over 3,000 ms, limit 3,000 ms, 0 of 3 turns over" in lines


def test_figures_text_names_unanswered_and_unhealthy_turns_and_reads_interruptions():
    assert figures_text({"p95Ms": 1840, "limitMs": 1200, "turns": 9, "turnsOver": 3, "unanswered": 1,
                         "unscoredTurns": [4]}) == (
        "p95 1,840 ms, limit 1,200 ms, 3 of 9 turns over (1 unanswered); turns 4 left out (caller unhealthy)")
    assert figures_text({"slowestMs": 620, "p95Ms": 610, "limitMs": 500, "interruptions": 2,
                         "interruptionsOver": 1}) == "slowest stop 620 ms, limit 500 ms, 1 of 2 interruptions over"
    assert figures_text({"longestSilenceMs": 4200, "limitMs": 3000, "turns": 1, "turnsOver": 1}) == (
        "longest silence 4,200 ms, limit 3,000 ms, 1 of 1 turn over")
    assert figures_text(None) == "" and figures_text({"unscoredTurns": []}) == ""


def test_figures_text_counts_interruptions_the_bot_finished_before():
    assert figures_text({"slowestMs": 320, "limitMs": 500, "interruptions": 1, "interruptionsOver": 0,
                         "notReached": 1}) == "slowest stop 320 ms, limit 500 ms, 0 of 1 interruption over (1 not reached)"


def test_a_dynamic_call_shows_its_turns_a_conversation_excerpt_and_task_success():
    from evalkit.voicetest.summary import call_lines, caller_text

    transcript = [{"turn": 0, "role": "caller", "text": "Hi, I was charged twice for my order."},
                  {"turn": 0, "role": "bot", "text": "Sorry to hear that. Which order is it?"},
                  {"turn": 1, "role": "caller", "text": "It's 4471.", "think_ms": 1400},
                  {"turn": 1, "role": "bot", "text": "Found it."},
                  {"turn": 2, "role": "caller", "text": "When does the money arrive?", "think_ms": 1200},
                  {"turn": 2, "role": "bot", "text": "In three to five working days. " * 6}]
    card = {"call": {"response_delay_p50_ms": 640, "response_delay_p95_ms": 700, "transcript": transcript,
                     "mode": "dynamic"},
            "turns": [{"response_delay_ms": 640}] * 3,
            "caller_health": {"healthy": True, "late_frames": 0, "think_ms_p50": 1400, "think_ms_max": 1400}}
    outcome = SimpleNamespace(scenario_name="duplicate charge", scenario_id="s", call_index=0, repeat=0,
                              duration_ms=21000, error=None, scorecard=card, problems=[], recording=None,
                              turns_taken=2)
    lines = call_lines(outcome)
    start = lines.index("  conversation     2 turns after the opening")
    assert lines[start + 1:start + 6] == [
        "    caller: Hi, I was charged twice for my order.",
        "    bot:    Sorry to hear that. Which order is it?",
        "    ...",
        "    caller: When does the money arrive?",
        "    bot:    In three to five working days. In three to five working days. In three to five working...",
    ]
    assert "  caller           healthy (no late frames; deciding p50 1,400 ms, max 1,400 ms)" in lines

    assert caller_text({"provider": "google", "model": "gemini-2.5-flash", "keySource": "platform"}, 87) == (
        "caller model: google/gemini-2.5-flash (the SyntropyLabs model); 87 SyntropyLabs model minutes left "
        "this month")
    assert caller_text({"provider": "google", "keySource": "request"}, -1) == (
        "caller model: google (your key, sent with the run); SyntropyLabs model minutes: unlimited")
    assert caller_text(None, None) is None

    run = dict(RUN, platformMinutes=2.25, calls=[dict(RUN["calls"][0], evaluators=RUN["calls"][0]["evaluators"] + [
        {"ruleId": "e", "key": "voice_task_success", "name": "Voice Task Success", "score": 0.5, "passed": True,
         "reason": "Met: the refund. Missed: when the money arrives."}])])
    report = SimpleNamespace(calls=[outcome], summary=run, problems=[], out_dir=None, console_url=None,
                             run_id="r1", verdict="failed", exit_code=1,
                             caller={"provider": "google", "model": "gemini-2.5-flash", "keySource": "organization"},
                             platform_minutes_left=None)
    out = run_lines(report)
    assert out[1:3] == ["caller model: google/gemini-2.5-flash (your organization's key)",
                        "this run used 2.2 SyntropyLabs model minutes"]
    assert out[out.index("task success:") + 1] == (
        "  call 0: 0.50 passed Met: the refund. Missed: when the money arrives.")
