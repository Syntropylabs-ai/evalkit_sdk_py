"""Inline scenarios as written in a file or passed from Python: the aliases they are normalised from,
the API's limits checked before anything is sent, the starter file `evalkit voice-test init` writes,
and the run arguments that choose between a scenario set and inline scenarios. Standard library only,
except the YAML cases, which need PyYAML."""

from __future__ import annotations

import json
import sys

import pytest

from evalkit.voicetest.runner import VoiceTestSetupError, _scenario_source, _voice_spec
from evalkit.voicetest.scenarios import (
    STARTER_YAML,
    ScenarioError,
    describe,
    load_scenarios,
    normalize_scenarios,
    starter_scenarios,
    with_max_turns,
    write_starter,
)


def _one(**scenario):
    scenario.setdefault("name", "s")
    scenario.setdefault("lines", ["hello"])
    return normalize_scenarios([scenario])[0]


def test_lines_may_be_strings_and_aliases_become_the_wire_shape():
    out = normalize_scenarios({"scenarios": [{
        "name": "  Refund  ",
        "success_criteria": "  refunds the charge ",
        "lines": [
            " Hi, I was charged twice. ",
            {"text": "Order 4471.", "wait": {"mode": "bot_done", "silence_ms": 900, "max_ms": 20000,
                                             "start_within_ms": 2000}},
            {"text": "Two seconds later.", "wait": {"ms": 2000}},
            {"text": "Sorry, how long?", "interrupt_after_bot_ms": 1500},
        ],
    }]})
    assert out == [{
        "name": "Refund",
        "lines": [
            {"text": "Hi, I was charged twice."},
            {"text": "Order 4471.", "wait": {"rule": "bot_done", "silenceMs": 900, "maxMs": 20000,
                                             "startWithinMs": 2000}},
            {"text": "Two seconds later.", "wait": {"rule": "fixed", "ms": 2000}},
            {"text": "Sorry, how long?", "interruptAfterBotMs": 1500},
        ],
        "successCriteria": ["refunds the charge"],
    }]


def test_a_generated_scenario_plays_its_turns_and_its_other_fields_are_not_sent():
    generated = {"scenario_id": "scn_ab12", "name": "late refund", "category": "happy_path",
                 "starting_prompt": "Hi", "turns": ["Hi", "It was order 9."], "persona": "NOVICE",
                 "success_criteria": ["offers credit"], "hidden_info": {}, "tool_fixtures": []}
    assert normalize_scenarios([generated]) == [
        {"name": "late refund", "lines": [{"text": "Hi"}, {"text": "It was order 9."}],
         "successCriteria": ["offers credit"]}]
    # Written for a live simulated user (no turns): played as a caller with a goal.
    live = dict(generated, turns=[], user_goal="Get money back for late headphones", starting_prompt="Hi there",
                hidden_info={"order": {"value": "9", "reveal_trigger": "asked for the order"}, "name": "Ana"},
                persona={"name": "NOVICE", "description": "new to this"})
    assert normalize_scenarios([live]) == [
        {"name": "late refund", "goal": "Get money back for late headphones", "persona": "NOVICE",
         "facts": {"order": {"value": "9", "revealWhen": "asked for the order"}, "name": "Ana"},
         "opening": "Hi there", "successCriteria": ["offers credit"]}]
    with pytest.raises(ScenarioError, match="has a user_goal instead; this one has neither"):
        normalize_scenarios([dict(generated, turns=[])])


@pytest.mark.parametrize("scenarios, message", [
    ([{"name": f"s{i}", "lines": ["x"]} for i in range(51)], "51 scenarios; a run takes at most 50"),
    ([{"name": "s", "lines": ["x"] * 41}], r'scenarios\[0\] \("s"\): 41 lines; a scenario takes at most 40'),
    ([{"name": "s", "lines": ["ok", "y" * 501]}], r'scenarios\[0\] \("s"\): lines\[1\]: the text is 501 characters'),
    ([{"name": "n" * 201, "lines": ["x"]}], "the name takes at most 200 characters"),
    ([{"name": "s", "lines": ["x"], "successCriteria": ["c"] * 21}], "21 success criteria"),
    ([{"name": "s", "lines": ["x"], "successCriteria": ["c" * 501]}], "at most 500 characters"),
    ([{"name": "s", "lines": []}], "a non-empty lines list"),
    ([{"lines": ["x"]}], "a scenario needs a name"),
    ([{"name": "s", "lines": ["   "]}], "the text is empty"),
    ([{"name": "s", "lines": [True]}], r'lines\[0\] is true, not text; in YAML, quote'),
    ([{"name": "s", "lines": [{"Note": "the colon"}]}], "has no text"),
    ([{"name": "s", "lines": [{"text": "x", "pause": 3}]}], r"unknown line fields \['pause'\]"),
    ([{"name": "s", "line": ["x"]}], r"unknown scenario fields \['line'\]"),
    ([{"name": "s", "lines": [{"text": "x", "wait": {"rule": "soon"}}]}], "rule must be bot_done or fixed"),
    ([{"name": "s", "lines": [{"text": "x", "wait": {"rule": "fixed"}}]}], "a fixed wait needs ms"),
    ([{"name": "s", "lines": [{"text": "x", "wait": {"ms": 60001}}]}], "ms must be from 0 to 60,000 ms"),
    ([{"name": "s", "lines": [{"text": "x", "wait": {"maxMs": 500}}]}], "maxMs must be from 1,000 to 60,000"),
    ([{"name": "s", "lines": [{"text": "x", "wait": {"silenceMs": 10001}}]}], "silenceMs must be from 0 to 10,000"),
    ([{"name": "s", "lines": [{"text": "x", "wait": {"maxMs": 5000, "startWithinMs": 6000}}]}],
     "startWithinMs must be from 0 to 5,000"),
    ([{"name": "s", "lines": [{"text": "x", "wait": {"rule": "fixed", "ms": 1, "maxMs": 2000}}]}],
     "a fixed wait takes only ms"),
    ([{"name": "s", "lines": [{"text": "x", "interruptAfterBotMs": 30001}]}],
     "interruptAfterBotMs must be from 0 to 30,000"),
    ([{"name": "s", "lines": [{"text": "x", "interruptAfterBotMs": 900, "wait": {"rule": "fixed", "ms": 0}}]}],
     "a line that interrupts waits for the bot to speak"),
    ([{"name": "s", "lines": [{"text": "x", "interruptAfterBotMs": 900, "wait": {"startWithinMs": 100}}]}],
     "without startWithinMs"),
    ([{"name": "s", "lines": [{"text": "x", "interruptAfterBotMs": 15000}]}],
     r"interruptAfterBotMs \(15,000\) must be less than the wait's maxMs \(15,000\)"),
    ({"scenarios": [{"name": "s", "lines": ["x"]}], "url": "ws://bot"}, "unknown top-level fields"),
    ([], "no scenarios"),
])
def test_the_api_limits_are_checked_with_the_scenario_and_line_named(scenarios, message):
    with pytest.raises(ScenarioError, match=message):
        normalize_scenarios(scenarios, where="calls.yaml")


def test_a_goal_scenario_in_the_wire_shape_from_its_aliases():
    out = normalize_scenarios([{
        "name": " Refund ", "goal": "  Get the second charge refunded ", "persona": "impatient",
        "facts": {"order_number": "4471", "email": {"value": "a@b.c", "reveal_when": "asked to confirm"}},
        "opening": " Hi, I was charged twice. ", "success_criteria": "refunds it", "max_turns": 6,
        "interrupt": {"turn": 2, "after_bot_ms": 1500},
    }])
    assert out == [{"name": "Refund", "goal": "Get the second charge refunded", "persona": "IMPATIENT",
                    "facts": {"order_number": "4471", "email": {"value": "a@b.c", "revealWhen": "asked to confirm"}},
                    "opening": "Hi, I was charged twice.", "successCriteria": ["refunds it"], "maxTurns": 6,
                    "interrupt": {"turn": 2, "afterBotMs": 1500}}]
    # free text is a persona too
    (free,) = normalize_scenarios([{"name": "s", "goal": "g", "persona": "A retired teacher who reads every word"}])
    assert free["persona"] == "A retired teacher who reads every word"


@pytest.mark.parametrize("scenario, message", [
    ({"goal": "g", "lines": ["x"]}, "a goal .* or lines .*, never both"),
    ({}, "needs lines .* or a goal"),
    ({"goal": ""}, "the goal must be non-empty text"),
    ({"goal": "g" * 1001}, "the goal is 1,001 characters; it takes at most 1,000"),
    ({"goal": "g", "persona": "p" * 501}, "persona is 501 characters"),
    ({"goal": "g", "facts": {f"k{i}": "v" for i in range(13)}}, "13 facts; a scenario takes at most 12"),
    ({"goal": "g", "facts": {"order": 4471}}, r'facts.order: the fact is 4471, not text; in YAML, quote it'),
    ({"goal": "g", "facts": {"order": {"value": "1", "when": "x"}}}, r"unknown fact fields \['when'\]"),
    ({"goal": "g", "facts": ["order"]}, "facts is a mapping"),
    ({"goal": "g", "opening": "o" * 501}, "the opening is 501 characters"),
    ({"goal": "g", "maxTurns": 21}, "maxTurns must be from 1 to 20, not 21"),
    ({"goal": "g", "maxTurns": 0}, "maxTurns must be from 1 to 20"),
    ({"goal": "g", "maxTurns": True}, "maxTurns must be a whole number"),
    ({"goal": "g", "interrupt": {"turn": 9, "afterBotMs": 100}}, r"interrupt.turn must be a caller line from 1 to maxTurns \(8\)"),
    ({"goal": "g", "maxTurns": 3, "interrupt": {"turn": 4, "afterBotMs": 100}}, r"from 1 to maxTurns \(3\)"),
    ({"goal": "g", "interrupt": {"turn": 1, "afterBotMs": 30001}}, "interrupt.afterBotMs must be from 0 to 30,000"),
    ({"goal": "g", "interrupt": {"turn": 1}}, "interrupt needs afterBotMs"),
    ({"goal": "g", "interrupt": {"afterBotMs": 5}}, "interrupt needs the turn"),
    ({"goal": "g", "interrupt": {"turn": 1, "afterBotMs": 5, "at": 2}}, r"unknown interrupt fields \['at'\]"),
    ({"goal": "g", "wait": {"rule": "fixed", "ms": 0}}, r"unknown scenario fields \['wait'\] \(a scenario with a goal"),
    ({"goal": "g", "maxTurns": 2, "max_turns": 3}, "maxTurns is given twice"),
    ({"lines": ["x"], "persona": "NOVICE"}, "persona belong to a scenario with a goal"),
])
def test_a_goal_scenarios_limits_are_checked_with_the_scenario_named(scenario, message):
    with pytest.raises(ScenarioError, match=message) as info:
        normalize_scenarios([dict(scenario, name="refund")], where="calls.yaml")
    assert str(info.value).startswith('calls.yaml: scenarios[0] ("refund")')


def test_a_goal_scenarios_limits_themselves_are_allowed():
    out = normalize_scenarios([{"name": "s", "goal": "g" * 1000, "persona": "p" * 500, "opening": "o" * 500,
                                "facts": {f"k{i}": "v" * 500 for i in range(12)}, "maxTurns": 20,
                                "interrupt": {"turn": 20, "afterBotMs": 30000}}])
    assert out[0]["maxTurns"] == 20 and len(out[0]["facts"]) == 12


def test_max_turns_fills_in_the_goal_scenarios_that_set_none():
    scenarios = normalize_scenarios([{"name": "a", "goal": "g"}, {"name": "b", "goal": "g", "maxTurns": 2},
                                     {"name": "c", "lines": ["x"]}])
    assert [s.get("maxTurns") for s in with_max_turns(scenarios, 4)] == [4, 2, None]
    with pytest.raises(ScenarioError, match="after the last of 3 turns"):
        with_max_turns(normalize_scenarios([{"name": "a", "goal": "g", "interrupt": {"turn": 5, "afterBotMs": 1}}]), 3)
    assert describe(scenarios) == "3 scenarios: 2 with a goal, 1 with 1 line"
    assert describe(normalize_scenarios([{"name": "c", "lines": ["x", "y"]}])) == "1 scenario, 2 lines"


def test_the_limits_themselves_are_allowed():
    out = normalize_scenarios([{"name": "n" * 200, "lines": ["y" * 500] * 40, "successCriteria": ["c" * 500] * 20}
                               for _ in range(50)])
    assert len(out) == 50 and len(out[0]["lines"]) == 40
    assert _one(lines=[{"text": "x", "interruptAfterBotMs": 14999}])["lines"][0]["interruptAfterBotMs"] == 14999


def test_a_json_file_is_read_and_its_errors_name_the_file(tmp_path):
    path = tmp_path / "calls.json"
    path.write_text(json.dumps([{"name": "s", "lines": ["hello", {"text": "x" * 600}]}]))
    with pytest.raises(ScenarioError) as info:
        load_scenarios(path)
    assert str(info.value).startswith(f"{path}: scenarios[0] (\"s\"): lines[1]: the text is 600 characters")
    path.write_text("{not json")
    with pytest.raises(ScenarioError, match="not valid JSON"):
        load_scenarios(path)
    with pytest.raises(ScenarioError, match="No such file"):
        load_scenarios(tmp_path / "missing.json")


def test_a_yaml_file_without_pyyaml_says_how_to_get_it(tmp_path, monkeypatch):
    path = tmp_path / "calls.yaml"
    path.write_text("scenarios: []\n")
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(ScenarioError, match=r"reading YAML needs PyYAML: pip install 'syntropylabs-evalkit\[voice\]'"):
        load_scenarios(path)


def test_the_starter_yaml_is_the_starter_scenarios(tmp_path):
    pytest.importorskip("yaml")
    path = write_starter(tmp_path / "voice-tests.yaml")
    assert open(path, encoding="utf-8").read() == STARTER_YAML
    loaded = load_scenarios(path)
    assert loaded == starter_scenarios()
    assert [s["name"] for s in loaded] == ["Refund for a duplicate charge", "Refund outside the return window",
                                           "Opening hours, exact lines"]
    assert loaded[0] == {
        "name": "Refund for a duplicate charge",
        "goal": "Get the second charge on order 4471 refunded and find out when the money arrives",
        "persona": "IMPATIENT", "facts": {"order_number": "4471"},
        "successCriteria": ["The agent refunds the second charge", "The agent says when the refund will arrive"],
        "interrupt": {"turn": 1, "afterBotMs": 1500}}
    assert loaded[2]["lines"] == [{"text": "Hi, what are your opening hours on Saturday?"},
                                  {"text": "Thanks, that's all."}]
    with pytest.raises(FileExistsError):
        write_starter(path)
    assert write_starter(path, force=True) == str(path)


def test_the_readme_shows_the_starter_byte_for_byte():
    import os

    from .support import SDK_ROOT

    with open(os.path.join(SDK_ROOT, "README.md"), encoding="utf-8") as fh:
        assert "```yaml\n" + STARTER_YAML + "```" in fh.read()


def test_the_json_starter_needs_no_yaml(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)
    path = write_starter(tmp_path / "voice-tests.json")
    assert load_scenarios(path) == starter_scenarios()


# The run's arguments -----------------------------------------------------------------------------


def test_a_scenario_set_with_ids_as_scenarios_is_the_older_subset_call():
    assert _scenario_source("set-1", ["a", "b"], None, 1) == ("set-1", ["a", "b"], None)
    assert _scenario_source("set-1", None, ["a"], 2) == ("set-1", ["a"], None)
    with pytest.raises(VoiceTestSetupError, match="pass scenario_set too"):
        _scenario_source(None, ["a"], None, 1)


def test_inline_scenarios_exclude_a_set_and_a_subset_and_count_calls():
    inline = [{"name": "s", "lines": ["hi"]}]
    assert _scenario_source(None, inline, None, 1) == (None, (), [{"name": "s", "lines": [{"text": "hi"}]}])
    with pytest.raises(VoiceTestSetupError, match="either scenario_set or inline scenarios"):
        _scenario_source("set-1", inline, None, 1)
    with pytest.raises(VoiceTestSetupError, match="scenario_ids picks from a scenario set"):
        _scenario_source(None, inline, ["a"], 1)
    with pytest.raises(VoiceTestSetupError, match="26 scenarios x 2 repeats is 52 calls; one run places at most 50"):
        _scenario_source(None, [{"name": f"s{i}", "lines": ["hi"]} for i in range(26)], None, 2)
    with pytest.raises(VoiceTestSetupError, match="pass scenario_set .* or scenarios"):
        _scenario_source(None, None, None, 1)
    with pytest.raises(VoiceTestSetupError, match=r"lines\[0\]: the text is empty"):
        _scenario_source(None, [{"name": "s", "lines": [""]}], None, 1)


def test_a_loaded_scenarios_file_mapping_is_inline_scenarios():
    loaded = {"scenarios": [{"name": "s", "lines": ["hi"]}]}
    assert _scenario_source(None, loaded, None, 1) == (None, (), [{"name": "s", "lines": [{"text": "hi"}]}])
    with pytest.raises(VoiceTestSetupError, match="either scenario_set or inline scenarios"):
        _scenario_source("set-1", loaded, None, 1)
    with pytest.raises(VoiceTestSetupError, match="unknown top-level fields"):
        _scenario_source(None, {"scenarios": loaded["scenarios"], "url": "wss://x"}, None, 1)


def test_inline_scenarios_may_be_a_file_path(tmp_path):
    path = tmp_path / "calls.json"
    path.write_text(json.dumps({"scenarios": [{"name": "s", "lines": ["hi"]}]}))
    assert _scenario_source(None, str(path), None, 1)[2] == [{"name": "s", "lines": [{"text": "hi"}]}]
    assert _scenario_source(None, path, None, 1)[2] == [{"name": "s", "lines": [{"text": "hi"}]}]


def test_the_voice_key_is_sent_as_api_key_and_checked():
    assert _voice_spec({"provider": "OpenAI", "model": "tts-1", "voice": "alloy", "api_key": " sk-x "}) == {
        "provider": "openai", "model": "tts-1", "voice": "alloy", "apiKey": "sk-x"}
    assert _voice_spec({"provider": "google", "model": "chirp3-hd", "voice": "Kore", "apiKey": "AIza1"})["apiKey"] == "AIza1"
    assert "apiKey" not in _voice_spec({"provider": "openai", "model": "tts-1", "voice": "alloy", "api_key": ""})
    assert _voice_spec(None) is None
    # The provider alone: the API picks the model and voice
    assert _voice_spec({"provider": "openai", "api_key": "sk-x"}) == {"provider": "openai", "apiKey": "sk-x"}
    assert _voice_spec({"provider": "openai", "model": "tts-1"}) == {"provider": "openai", "model": "tts-1"}
    with pytest.raises(VoiceTestSetupError, match="voice needs a provider"):
        _voice_spec({"model": "tts-1", "voice": "alloy"})
    with pytest.raises(VoiceTestSetupError, match=r"not \['key'\]"):
        _voice_spec({"provider": "openai", "model": "tts-1", "voice": "alloy", "key": "sk-x"})


# The command line's voice flags ------------------------------------------------------------------


def _args(**kw):
    import argparse

    base = {"tts_provider": None, "tts_model": None, "tts_voice": None, "tts_key_env": None, "voice": None}
    base.update(kw)
    return argparse.Namespace(**base)


def test_the_tts_key_comes_from_the_providers_usual_variable():
    import io

    from evalkit.voicetest.cli import _tts_voice, tts_key_env

    assert tts_key_env("openai") == ("OPENAI_API_KEY",)
    assert tts_key_env("Gemini") == ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    assert tts_key_env("my-tts.v2") == ("MY_TTS_V2_API_KEY",)
    err = io.StringIO()
    assert _tts_voice(_args(tts_provider="google"), {"GEMINI_API_KEY": "AIza-k"}, err) == {
        "provider": "google", "api_key": "AIza-k"}
    assert "synthesized with the key in GEMINI_API_KEY" in err.getvalue() and "AIza-k" not in err.getvalue()
    err = io.StringIO()
    assert _tts_voice(_args(tts_provider="my-tts", tts_model="m1", tts_voice="v1"), {"MY_TTS_API_KEY": "k9"}, err) == {
        "provider": "my-tts", "model": "m1", "voice": "v1", "api_key": "k9"}
    err = io.StringIO()
    assert _tts_voice(_args(tts_provider="cartesia", tts_key_env="CI_CARTESIA"), {"CARTESIA_API_KEY": "x"}, err) == {
        "provider": "cartesia"}
    assert err.getvalue().strip() == ("evalkit voice-test: CI_CARTESIA is not set, so no key is sent; the "
                                      "organization's stored cartesia key will be used")


def _caller_args(**kw):
    import argparse

    base = {"caller_provider": None, "caller_model": None, "caller_key_env": None}
    base.update(kw)
    return argparse.Namespace(**base)


def test_the_callers_key_comes_from_the_environment_and_google_is_the_default():
    import io

    from evalkit.voicetest.cli import UsageError, _caller_model

    err = io.StringIO()
    assert _caller_model(_caller_args(), {"GOOGLE_API_KEY": "AIza-x"}, err) is None  # no flag, nothing sent
    assert _caller_model(_caller_args(caller_model="gemini-2.5-pro"), {"GEMINI_API_KEY": "AIza-g"}, err) == {
        "provider": "google", "model": "gemini-2.5-pro", "api_key": "AIza-g"}
    assert "decided by google gemini-2.5-pro, with the key in GEMINI_API_KEY" in err.getvalue()
    assert "AIza-g" not in err.getvalue()
    err = io.StringIO()
    assert _caller_model(_caller_args(caller_key_env="CI_GEMINI"), {"CI_GEMINI": "AIza-ci", "GOOGLE_API_KEY": "no"},
                         err) == {"provider": "google", "api_key": "AIza-ci"}
    err = io.StringIO()
    assert _caller_model(_caller_args(caller_provider="openai"), {}, err) == {"provider": "openai"}
    assert err.getvalue().strip() == ("evalkit voice-test: OPENAI_API_KEY is not set, so no caller key is sent; the "
                                      "organization's stored openai key will be used")
    with pytest.raises(UsageError, match="takes the name of an environment variable"):
        _caller_model(_caller_args(caller_key_env="AIza-secret-value"), {}, io.StringIO())
    with pytest.raises(UsageError, match="--caller-provider takes a provider name") as info:
        _caller_model(_caller_args(caller_provider="AIzaSyA-secret"), {}, io.StringIO())
    assert "AIzaSyA-secret" not in str(info.value)


def test_the_caller_spec_and_generated_callers_as_the_api_takes_them():
    from evalkit.voicetest.runner import _caller_spec, _generate_spec

    assert _caller_spec({"api_key": " AIza-k "}) == {"provider": "google", "apiKey": "AIza-k"}
    assert _caller_spec({"provider": "Google", "model": "gemini-2.5-flash"}) == {
        "provider": "google", "model": "gemini-2.5-flash"}
    assert _caller_spec(None) is None
    with pytest.raises(VoiceTestSetupError, match=r"not \['key'\]"):
        _caller_spec({"key": "AIza-k"})
    assert _generate_spec("A refund desk.", None, None) == {"about": "A refund desk.", "count": 5}
    assert _generate_spec(" A refund desk. ", 3, 4) == {"about": "A refund desk.", "count": 3, "maxTurns": 4}
    with pytest.raises(VoiceTestSetupError, match="callers must be a whole number from 1 to 20"):
        _generate_spec("A refund desk.", 21, None)
    with pytest.raises(VoiceTestSetupError, match="pass about too"):
        _generate_spec(None, 3, None)
    with pytest.raises(VoiceTestSetupError, match="about must say what your bot does"):
        _generate_spec("  ", None, None)
