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
