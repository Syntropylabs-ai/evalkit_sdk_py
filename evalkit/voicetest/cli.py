"""``evalkit voice-test``: `run` tests a bot through SyntropyLabs, against callers written from what the bot
does (``--about``), the scenarios of a YAML or JSON file, or a scenario set; `init` writes a starter
scenarios file; `check`, `call` and `refbot` are local tools that need no backend (a quick reachability and
format check, one scripted call, and the reference bot).

`main(argv, prog)` is the entry point (``evalkit voice-test ...`` and ``python -m evalkit.voicetest``);
`add_subcommands(subparsers)` mounts the commands under another parser. Importing this module loads
nothing beyond argparse: each command imports what it needs when it runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, TextIO, Tuple

log = logging.getLogger("evalkit.voicetest.cli")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_SETUP = 2
EXIT_INTERRUPTED = 130


class UsageError(ValueError):
    pass


def _headers(values: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise UsageError("--header expects K=V")
        k, v = item.split("=", 1)
        if not k.strip():
            raise UsageError("--header needs a name before the =")
        out[k.strip()] = v
    return out


def _voice(value: Optional[str]) -> Optional[Dict[str, str]]:
    if not value:
        return None
    parts = value.split(":")
    if len(parts) != 3 or not all(p.strip() for p in parts):
        raise UsageError("--voice expects PROVIDER:MODEL:VOICE, for example openai:tts-1:alloy")
    return {"provider": parts[0].strip(), "model": parts[1].strip(), "voice": parts[2].strip()}


# The environment variables a text-to-speech provider's key is read from, as the console's run commands
# name them: the variable the provider's own SDK reads. A provider not listed reads <PROVIDER>_API_KEY.
TTS_KEY_ENV: Mapping[str, Tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "elevenlabs": ("ELEVENLABS_API_KEY",),
    "cartesia": ("CARTESIA_API_KEY",),
}
# A variable's name, never its value: a key on the command line would be kept in the shell's history.
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{0,99}$")


def tts_key_env(provider: str) -> Tuple[str, ...]:
    """Where `provider`'s key is read from: its usual variables, else `<PROVIDER>_API_KEY`."""
    known = TTS_KEY_ENV.get(provider.lower())
    if known:
        return known
    word = re.sub(r"[^A-Z0-9]+", "_", provider.upper()).strip("_")
    return (f"{word or 'TTS'}_API_KEY",)


def _tts_voice(args, env: Mapping[str, str], err: TextIO) -> Optional[Dict[str, str]]:
    """The caller's voice from --tts-provider (with --tts-model and --tts-voice when given), and the
    provider's key read from the environment (--tts-key-env, else the provider's usual variable). The
    API picks the model and voice the flags leave out. With no variable set, no key is sent, and the
    organization's stored key for the provider is used."""
    given = [flag for flag, value in (("--tts-model", args.tts_model), ("--tts-voice", args.tts_voice),
                                      ("--tts-key-env", args.tts_key_env)) if value]
    if not args.tts_provider:
        if given:
            raise UsageError(f"{', '.join(given)} needs --tts-provider")
        return None
    if args.voice:
        raise UsageError("pass either --voice or --tts-provider, not both")
    provider = args.tts_provider.strip().lower()
    # Refused without echoing it: a key pasted here by mistake must not reach the terminal or a CI log.
    if not re.match(r"^[a-z][a-z0-9_.-]{0,39}$", provider) or provider.startswith(("sk-", "sk_", "aiza")):
        raise UsageError("--tts-provider takes a provider name, such as openai, google, elevenlabs or cartesia; "
                         "the key itself is read from the environment")
    if args.tts_key_env:
        if not _ENV_NAME.match(args.tts_key_env):
            raise UsageError("--tts-key-env takes the name of an environment variable in capitals, such as "
                             "MY_TTS_KEY; the key itself is never taken on the command line")
        names: Tuple[str, ...] = (args.tts_key_env,)
    else:
        names = tts_key_env(provider)
    spec = {"provider": provider}
    for flag, value in (("model", args.tts_model), ("voice", args.tts_voice)):
        if value and value.strip():
            spec[flag] = value.strip()
    shown = " ".join(spec.values())
    for name in names:
        value = (env.get(name) or "").strip()
        if value:
            spec["api_key"] = value
            print(f"evalkit voice-test: the caller's voice is {shown}, synthesized with the key in {name}",
                  file=err)
            return spec
    print(f"evalkit voice-test: {' and '.join(names)} {'is' if len(names) == 1 else 'are'} not set, so no key is "
          f"sent; the organization's stored {provider} key will be used", file=err)
    return spec


def _provider_name(value: str, flag: str) -> str:
    provider = value.strip().lower()
    # Refused without echoing it: a key pasted here by mistake must not reach the terminal or a CI log.
    if not re.match(r"^[a-z][a-z0-9_.-]{0,39}$", provider) or provider.startswith(("sk-", "sk_", "aiza")):
        raise UsageError(f"{flag} takes a provider name, such as google or openai; the key itself is read from "
                         "the environment")
    return provider


def _caller_model(args, env: Mapping[str, str], err: TextIO) -> Optional[Dict[str, str]]:
    """The model that decides a dynamic call's lines, from --caller-provider (google when only
    --caller-model or --caller-key-env is given), --caller-model, and the key read from the environment
    (--caller-key-env, else the provider's usual variable). With none of the flags no caller is sent, and
    the API takes the organization's own audio-capable model, else the SyntropyLabs model."""
    if not (args.caller_provider or args.caller_model or args.caller_key_env):
        return None
    from .runner import DEFAULT_CALLER_PROVIDER

    provider = _provider_name(args.caller_provider or DEFAULT_CALLER_PROVIDER, "--caller-provider")
    if args.caller_key_env:
        if not _ENV_NAME.match(args.caller_key_env):
            raise UsageError("--caller-key-env takes the name of an environment variable in capitals, such as "
                             "GOOGLE_API_KEY; the key itself is never taken on the command line")
        names: Tuple[str, ...] = (args.caller_key_env,)
    else:
        names = tts_key_env(provider)
    spec = {"provider": provider}
    if args.caller_model and args.caller_model.strip():
        spec["model"] = args.caller_model.strip()
    shown = " ".join(spec.values())
    for name in names:
        value = (env.get(name) or "").strip()
        if value:
            spec["api_key"] = value
            print(f"evalkit voice-test: the caller's lines are decided by {shown}, with the key in {name}", file=err)
            return spec
    print(f"evalkit voice-test: {' and '.join(names)} {'is' if len(names) == 1 else 'are'} not set, so no caller "
          f"key is sent; the organization's stored {provider} key will be used", file=err)
    return spec


def _about(args) -> Optional[str]:
    if args.about is not None:
        if not args.about.strip():
            raise UsageError("--about needs a few sentences on what your bot does")
        return args.about
    if args.about_file is None:
        return None
    try:
        with open(args.about_file, "r", encoding="utf-8") as fh:
            text = fh.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise UsageError(f"--about-file {args.about_file}: {getattr(exc, 'strerror', None) or exc}") from None
    if not text.strip():
        raise UsageError(f"--about-file {args.about_file} is empty; it should say what your bot does")
    return text


def _first_speaker_flag(p: argparse.ArgumentParser, default: Optional[str]) -> None:
    from .script import FIRST_SPEAKERS  # standard library only, like this module

    p.add_argument("--first-speaker", choices=FIRST_SPEAKERS, default=default,
                   help="who opens the call: auto (the default) waits up to 3 s for a greeting to start, "
                        "then for it to end; bot waits for the greeting to end; caller speaks at once. "
                        "Either wait gives up 15 s after the call opens, even mid-greeting")


def _target_flags(p: argparse.ArgumentParser, format_required: bool = True) -> None:
    p.add_argument("--url", required=True, help="ws:// or wss:// URL of the bot")
    p.add_argument("--format", required=format_required, choices=("twilio", "pcm16"),
                   default=None if format_required else "pcm16")
    p.add_argument("--sample-rate", type=int, default=16000, choices=(8000, 16000, 24000),
                   help="pcm16 only (twilio is always 8000)")
    p.add_argument("--header", action="append", default=[], metavar="K=V",
                   help="handshake header for the bot (repeatable, up to 10); sent to the bot only")
    p.add_argument("--session-query-param", default="evalkit_session",
                   help="the URL query parameter that carries the session token")
    p.add_argument("--no-start", action="store_true", help="pcm16: do not send the start message")


def _add_run(sub) -> None:
    p = sub.add_parser("run", help="test the bot with callers and report to SyntropyLabs",
                       description="Creates a voice test run, calls the bot once per planned call, "
                                   "reports each call and waits for the evaluators. The callers come from "
                                   "exactly one of --about (we write them from what your bot does), "
                                   "--scenarios FILE or --scenario-set ID. Exit 0 passed, 1 failed, 2 setup or "
                                   "network failure, 130 interrupted.")
    _target_flags(p)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--about", default=None, metavar="TEXT",
                        help="what your bot does, in a few sentences: SyntropyLabs writes callers with goals from "
                             "it, who answer what your bot actually says")
    source.add_argument("--about-file", default=None, metavar="FILE", help="--about, read from a text file")
    source.add_argument("--scenarios", default=None, metavar="FILE",
                        help="a YAML or JSON file of scenarios, with goals or exact lines (evalkit voice-test init "
                             "writes one)")
    source.add_argument("--scenario-set", default=None, metavar="ID",
                        help="the id of a scenario set saved in the console")
    p.add_argument("--callers", type=int, default=None, metavar="N",
                   help="with --about: how many callers to write, 1 to 20 (default 5)")
    p.add_argument("--max-turns", type=int, default=None, metavar="N",
                   help="caller lines after the opening of each caller with a goal, 1 to 20, where the scenario "
                        "sets none (default 8)")
    p.add_argument("--caller-provider", default=None, metavar="PROVIDER",
                   help="the model provider that hears your bot and decides the caller's lines (default google), "
                        "with your own key read from the environment")
    p.add_argument("--caller-model", default=None, metavar="MODEL",
                   help="the caller's model (default: the API's, gemini-2.5-flash on google)")
    p.add_argument("--caller-key-env", default=None, metavar="VAR",
                   help="the environment variable holding the caller's key (default GOOGLE_API_KEY, else "
                        "GEMINI_API_KEY, for google; <PROVIDER>_API_KEY otherwise)")
    p.add_argument("--scenario", action="append", default=[], metavar="ID",
                   help="only this scenario of the set (repeatable; with --scenario-set)")
    p.add_argument("--repeats", type=int, default=1, help="calls per scenario (default 1)")
    p.add_argument("--agent", default=None, help="score the run under this registered agent id")
    p.add_argument("--name", default=None, help="the run's name (default: 'voice test <time>')")
    p.add_argument("--voice", default=None, metavar="PROVIDER:MODEL:VOICE",
                   help="the caller's voice, synthesized with the organization's connection for the provider "
                        "(default: the organization's Text-to-Speech model)")
    p.add_argument("--tts-provider", default=None, metavar="PROVIDER",
                   help="the caller's voice from your own provider key, read from the environment "
                        "(OPENAI_API_KEY, GOOGLE_API_KEY or GEMINI_API_KEY, ELEVENLABS_API_KEY, CARTESIA_API_KEY, "
                        "else <PROVIDER>_API_KEY); unset, the organization's stored key is used")
    p.add_argument("--tts-model", default=None,
                   help="the text-to-speech model (default: the project's Text-to-Speech model on the provider, "
                        "else the API's default for it)")
    p.add_argument("--tts-voice", default=None, help="the voice (default: as for --tts-model)")
    p.add_argument("--tts-key-env", default=None, metavar="VAR",
                   help="the environment variable holding the key, when not the provider's usual one")
    p.add_argument("--timing-only", action="store_true", help="upload no audio; send timings only")
    # Left out unless given, so the API's own default applies.
    _first_speaker_flag(p, None)
    p.add_argument("--out", default=None, metavar="DIR",
                   help="keep the recordings and scorecards under DIR/<runId>/")
    p.add_argument("--json", action="store_true", help="print one JSON object instead of the summary")
    p.add_argument("--key", default=None,
                   help="subscription key (default: EVALKIT_API_KEY, EVALKIT_SUBSCRIPTION_KEY, then the "
                        "stored coding config)")
    p.add_argument("--api-url", default=None,
                   help="API base URL (default: EVALKIT_API_URL, then https://api.syntropylabs.ai)")
    p.add_argument("--wait-timeout", type=float, default=600.0, metavar="SECONDS",
                   help="how long to wait for the evaluators after the last call (default 600)")
    p.add_argument("--cache-dir", default=None, help="where downloaded caller audio is kept")
    p.set_defaults(func=_run_run)


def _add_init(sub) -> None:
    from .scenarios import DEFAULT_FILE

    p = sub.add_parser("init", help="write a starter scenarios file for evalkit voice-test run --scenarios",
                       description="Writes three callers for a refund desk, to edit into your bot's: two with a "
                                   "goal (one cuts in on the bot) and one with exact lines. YAML unless PATH ends "
                                   "in .json.")
    p.add_argument("path", nargs="?", default=DEFAULT_FILE, metavar="PATH",
                   help=f"where to write it (default {DEFAULT_FILE})")
    p.add_argument("--force", action="store_true", help="overwrite PATH if it exists")
    p.set_defaults(func=_run_init)


def _add_check(sub) -> None:
    p = sub.add_parser("check", help="connect, say one line, and report whether the bot answered",
                       description="One short call with no backend: whether the bot is reachable, "
                                   "answers a line, and speaks the format. Exit 0 ok, 1 problems "
                                   "found, 2 bad arguments.")
    _target_flags(p, format_required=False)
    p.add_argument("--timeout", type=float, default=8.0, metavar="SECONDS",
                   help="how long to wait for the answer after the line (default 8)")
    _first_speaker_flag(p, "auto")
    p.add_argument("--out", default=None, metavar="DIR", help="keep the recording in DIR")
    p.add_argument("--json", action="store_true", help="print the findings as JSON")
    p.set_defaults(func=_run_check)


def _add_call(sub) -> None:
    p = sub.add_parser("call", help="run a scripted call against a voice bot")
    _target_flags(p)
    p.add_argument("--session", default=None, help="session token (default: a fresh local one)")
    p.add_argument("--script", required=True, help="script JSON file")
    p.add_argument("--out", required=True, help="output directory for this call")
    p.add_argument("--analyse", action="store_true", help="also write scorecard.json")
    p.add_argument("--cache-dir", default=None, help="where downloaded line audio is kept")
    p.set_defaults(func=_run_call)


def _add_refbot(sub) -> None:
    p = sub.add_parser("refbot", help="run the reference voice bot")
    p.add_argument("--format", required=True, choices=("twilio", "pcm16"))
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--behaviour", default=None, help="behaviour JSON file (default behaviour if absent)")
    p.add_argument("--truth-out", default=None, help="append ground truth to this JSONL file")
    p.add_argument("--sample-rate", type=int, default=None, choices=(8000, 16000, 24000),
                   help="pcm16 rate when the caller sends no start message")
    p.set_defaults(func=_run_refbot)


def add_subcommands(subparsers) -> None:
    _add_run(subparsers)
    _add_init(subparsers)
    _add_check(subparsers)
    _add_call(subparsers)
    _add_refbot(subparsers)


def _print(out: TextIO, lines: List[str]) -> None:
    for line in lines:
        print(line, file=out)
    out.flush()


def _setup_failure(message: str) -> int:
    print(f"evalkit voice-test: {message}", file=sys.stderr)
    return EXIT_SETUP


def _missing_extra() -> Optional[str]:
    missing = []
    for name in ("numpy", "websockets", "webrtcvad"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        return f"this command needs {', '.join(missing)}: pip install 'syntropylabs-evalkit[voice]'"
    return None


def _run_run(args, out: TextIO) -> int:
    from .runner import VoiceTestSetupError, run_async
    from .scenarios import ScenarioError, load_scenarios
    from .summary import call_lines, run_lines

    from .runner import MAX_CALLERS

    try:
        headers = _headers(args.header)
        if not (args.about is not None or args.about_file or args.scenarios or args.scenario_set):
            raise UsageError("say who calls your bot: --about \"what your bot does\" (we write the callers), "
                             "--scenarios FILE (evalkit voice-test init writes one) or --scenario-set ID")
        if args.scenario and not args.scenario_set:
            raise UsageError("--scenario picks from a scenario set; with --scenarios, list only the ones to play")
        about = _about(args)
        if args.callers is not None:
            if about is None:
                raise UsageError("--callers goes with --about or --about-file")
            if not 1 <= args.callers <= MAX_CALLERS:
                raise UsageError(f"--callers takes 1 to {MAX_CALLERS}")
        if args.max_turns is not None:
            if args.scenario_set:
                raise UsageError("--max-turns applies to --about and --scenarios; a scenario set's scenarios carry "
                                 "their own")
            if not 1 <= args.max_turns <= 20:
                raise UsageError("--max-turns takes 1 to 20")
        voice = _tts_voice(args, os.environ, sys.stderr) or _voice(args.voice)
        caller = _caller_model(args, os.environ, sys.stderr)
        scenarios = load_scenarios(args.scenarios) if args.scenarios else None
    except (UsageError, ScenarioError) as exc:
        return _setup_failure(str(exc))
    try:
        from evalkit_cli.config import ConfigStore
        from evalkit_cli.evaluation import resolve_api_url, resolve_key

        store = ConfigStore()
        key, api_url = resolve_key(args.key, store), resolve_api_url(args.api_url, store)
    except Exception as exc:  # an unreadable stored config is not worth failing over
        log.debug("the stored config could not be read: %s", exc)
        key = args.key or os.getenv("EVALKIT_API_KEY") or os.getenv("EVALKIT_SUBSCRIPTION_KEY")
        api_url = args.api_url or os.getenv("EVALKIT_API_URL")

    def on_call(outcome) -> None:
        _print(out, call_lines(outcome, args.timing_only))

    def on_message(text: str) -> None:
        _print(out, [text])

    try:
        report = asyncio.run(run_async(
            args.url,
            format=args.format,
            sample_rate=args.sample_rate,
            headers=headers,
            session_query_param=args.session_query_param,
            send_start=not args.no_start,
            scenario_set=args.scenario_set,
            scenarios=scenarios,
            scenario_ids=args.scenario,
            about=about,
            callers=args.callers,
            caller=caller,
            max_turns=args.max_turns,
            repeats=args.repeats,
            agent=args.agent,
            name=args.name,
            voice=voice,
            timing_only=args.timing_only,
            first_speaker=args.first_speaker,
            out_dir=args.out,
            subscription_key=key or None,
            api_url=api_url or None,
            wait_timeout_s=args.wait_timeout,
            cache_dir=args.cache_dir,
            on_call=None if args.json else on_call,
            on_message=(lambda text: log.info("%s", text)) if args.json else on_message,
        ))
    except VoiceTestSetupError as exc:
        return _setup_failure(str(exc))
    if args.json:
        print(json.dumps(report.to_json(), indent=2, default=str), file=out)
    else:
        _print(out, [""] + run_lines(report))
    return report.exit_code


def _run_init(args, out: TextIO) -> int:
    from .scenarios import describe, starter_scenarios, write_starter

    try:
        path = write_starter(args.path, force=args.force)
    except FileExistsError:
        return _setup_failure(f"{args.path} exists; pass --force to overwrite it")
    except OSError as exc:
        return _setup_failure(f"could not write {args.path}: {exc.strerror or exc}")
    lines = [f"wrote {path}: {describe(starter_scenarios())}",
             "edit them into your bot's callers, then run them:",
             f"  evalkit voice-test run --url ws://localhost:8765/ws --format pcm16 --scenarios {path}",
             "or have the callers written from what your bot does:",
             "  evalkit voice-test run --url ws://localhost:8765/ws --format pcm16 --about \"a refund desk for an "
             "online shop\""]
    if not path.lower().endswith(".json"):
        try:
            import yaml  # noqa: F401
        except ImportError:
            lines.append("reading YAML needs PyYAML: pip install 'syntropylabs-evalkit[voice]' "
                         "(or write JSON: evalkit voice-test init voice-tests.json)")
    _print(out, lines)
    return EXIT_OK


def _run_check(args, out: TextIO) -> int:
    from .check import check_async
    from .summary import check_lines
    from .transport import Target

    missing = _missing_extra()
    if missing:
        return _setup_failure(missing)
    try:
        target = Target(url=args.url, format=args.format, sample_rate=args.sample_rate,
                        headers=_headers(args.header), session_query_param=args.session_query_param,
                        send_start=not args.no_start)
    except ValueError as exc:
        return _setup_failure(str(exc))
    result = asyncio.run(check_async(target, answer_timeout_ms=int(args.timeout * 1000), out_dir=args.out,
                                     first_speaker=args.first_speaker))
    if args.json:
        print(json.dumps(result, indent=2, default=str), file=out)
    else:
        _print(out, check_lines(result))
    return EXIT_OK if result.get("ok") else EXIT_FAILED


def _run_call(args, out: TextIO) -> int:
    missing = _missing_extra()
    if missing:
        return _setup_failure(missing)
    from .audio_cache import AudioCache
    from .caller import CallError, run_call
    from .plan import PLAN_SOURCE_KINDS
    from .script import ScriptError, load_script
    from .sources import PlanAudioSource, default_sources
    from .transport import Target

    try:
        script = load_script(args.script, source_kinds=PLAN_SOURCE_KINDS)
        target = Target(
            url=args.url,
            format=args.format,
            sample_rate=args.sample_rate,
            headers=_headers(args.header),
            session_query_param=args.session_query_param,
            send_start=not args.no_start,
        )
    except (ScriptError, ValueError, OSError) as exc:
        return _setup_failure(str(exc))
    sources = default_sources([PlanAudioSource(AudioCache(args.cache_dir))])
    try:
        result = asyncio.run(run_call(target, script, args.out, session=args.session, sources=sources))
    except CallError as exc:
        log.error("%s", exc)
        return EXIT_FAILED
    except (ValueError, OSError, RuntimeError) as exc:  # a line's audio that cannot be had
        return _setup_failure(str(exc))
    summary: Dict[str, Any] = {
        "out": result.out_dir,
        "session": result.session,
        "duration_ms": result.duration_ms,
        "ended_by": result.ended_by,
        "pacing": result.pacing,
        "errors": result.errors,
    }
    if args.analyse:
        from .analysis.scorecard import write_scorecard

        summary["scorecard"] = write_scorecard(args.out)
    print(json.dumps(summary), file=out)
    return EXIT_OK if result.ended_by in ("script", "max_call", "bot_closed") else EXIT_FAILED


def _run_refbot(args, out: TextIO) -> int:
    missing = _missing_extra()
    if missing:
        return _setup_failure(missing)
    from .refbot import Behaviour, BehaviourError, RefBotServer, load_behaviour

    try:
        behaviour = load_behaviour(args.behaviour) if args.behaviour else Behaviour.from_json({})
    except (BehaviourError, ValueError, OSError) as exc:
        return _setup_failure(str(exc))
    server = RefBotServer(
        behaviour,
        args.format,
        host=args.host,
        port=args.port,
        truth_path=args.truth_out,
        sample_rate=args.sample_rate,
    )

    async def serve() -> None:
        try:
            await server.serve_forever()
        finally:
            await server.close()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
    return EXIT_OK


def build_parser(prog: str = "evalkit voice-test") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="SyntropyLabs voice bot testing over WebSocket")
    parser.add_argument("-v", "--verbose", action="store_true", help="log what the runner does")
    sub = parser.add_subparsers(dest="command")
    sub.required = True
    add_subcommands(sub)
    return parser


def _configure_logging(verbose: bool) -> None:
    logger = logging.getLogger("evalkit.voicetest")
    if not any(getattr(h, "_evalkit_voicetest", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
        handler._evalkit_voicetest = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)


def main(argv: Optional[List[str]] = None, prog: str = "evalkit voice-test",
         out: Optional[TextIO] = None) -> int:
    parser = build_parser(prog)
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        # Named without their values: a value given to an option that does not exist could be a key.
        flags = sorted({a.split("=", 1)[0] for a in unknown if a.startswith("-")})
        parser.error(f"unrecognized arguments: {' '.join(flags) or 'extra values'} (values not shown)")
    _configure_logging(args.verbose)
    try:
        return int(args.func(args, out or sys.stdout) or 0)
    except KeyboardInterrupt:
        print("evalkit voice-test: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
