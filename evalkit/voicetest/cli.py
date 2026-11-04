"""``evalkit voice-test``: `run` tests a bot against a scenario set through SyntropyLabs; `check`, `call`
and `refbot` are local tools that need no backend (a quick reachability and format check, one scripted
call, and the reference bot).

`main(argv, prog)` is the entry point (``evalkit voice-test ...`` and ``python -m evalkit.voicetest``);
`add_subcommands(subparsers)` mounts the commands under another parser. Importing this module loads
nothing beyond argparse: each command imports what it needs when it runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional, TextIO

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


def _first_speaker_flag(p: argparse.ArgumentParser, default: Optional[str]) -> None:
    from .script import FIRST_SPEAKERS  # standard library only, like this module

    p.add_argument("--first-speaker", choices=FIRST_SPEAKERS, default=default,
                   help="who opens the call: auto (the default) waits up to 3 s for a greeting to start, "
                        "bot waits up to 15 s for it, caller speaks at once")


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
    p = sub.add_parser("run", help="test the bot against a scenario set and report to SyntropyLabs",
                       description="Creates a voice test run, calls the bot once per planned call, "
                                   "reports each call and waits for the evaluators. Exit 0 passed, "
                                   "1 failed, 2 setup or network failure, 130 interrupted.")
    _target_flags(p)
    p.add_argument("--scenario-set", required=True, help="the scenario set id")
    p.add_argument("--scenario", action="append", default=[], metavar="ID",
                   help="only this scenario of the set (repeatable)")
    p.add_argument("--repeats", type=int, default=1, help="calls per scenario (default 1)")
    p.add_argument("--agent", default=None, help="score the run under this registered agent id")
    p.add_argument("--name", default=None, help="the run's name (default: 'voice test <time>')")
    p.add_argument("--voice", default=None, metavar="PROVIDER:MODEL:VOICE",
                   help="the caller's voice (default: the organisation's)")
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
    from .summary import call_lines, run_lines

    try:
        headers = _headers(args.header)
        voice = _voice(args.voice)
    except UsageError as exc:
        return _setup_failure(str(exc))
    try:
        from evalkit_cli.config import ConfigStore
        from evalkit_cli.evaluation import resolve_api_url, resolve_key

        store = ConfigStore()
        key, api_url = resolve_key(args.key, store), resolve_api_url(args.api_url, store)
    except Exception as exc:  # an unreadable stored config is not worth failing over
        log.debug("the stored config could not be read: %s", exc)
        import os

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
            scenarios=args.scenario,
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
    args = build_parser(prog).parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return int(args.func(args, out or sys.stdout) or 0)
    except KeyboardInterrupt:
        print("evalkit voice-test: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
