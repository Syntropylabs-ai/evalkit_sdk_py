"""A voice bot that never calls evalkit.init() (no key where it runs) takes its calls untraced."""

import logging

import evalkit


def test_voice_hooks_return_none_and_warn_once_without_init(caplog):
    saved, warned = evalkit._instance, evalkit._voice_uninit_warned
    evalkit._instance = None
    evalkit._voice_uninit_warned = False
    try:
        with caplog.at_level(logging.WARNING, logger="evalkit"):
            assert evalkit.instrument_pipecat(object()) is None
            assert evalkit.instrument_livekit_agents(object()) is None
            assert evalkit.instrument_pipecat(object()) is None
        warnings = [r for r in caplog.records if "evalkit.init() was not called" in r.getMessage()]
        assert len(warnings) == 1
    finally:
        evalkit._instance, evalkit._voice_uninit_warned = saved, warned
