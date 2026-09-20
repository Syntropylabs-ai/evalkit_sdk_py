from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from . import doctor as doctor_run
from .config import ConfigStore
from .files import MalformedFile
from .vendors import VENDORS, ClaudeCodeInstaller, InstallOptions, VendorState, installer_for

HOOK_VENDORS = ("claude-code", "codex", "cursor", "windsurf")
BACKFILL_VENDORS = ("codex",)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalkit", description="EvalKit command line")
    sub = parser.add_subparsers(dest="command", required=True)
    coding = sub.add_parser("coding", help="connect coding agents to EvalKit")
    actions = coding.add_subparsers(dest="action", required=True)

    install = actions.add_parser("install")
    install.add_argument("--vendor", required=True, choices=VENDORS)
    install.add_argument("--key")
    install.add_argument("--endpoint")
    install.add_argument("--scope", choices=("user", "project"), default="user")
    install.add_argument("--capture", choices=("minimal", "full"), default=None)
    install.add_argument("--tenant")
    install.add_argument(
        "--traces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable the vendor's own span export (on by default; --no-traces keeps log events only)",
    )
    install.add_argument("--dry-run", action="store_true")
    install.add_argument("--managed", action="store_true")
    install.add_argument(
        "--subagents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="claude-code only: merge the PostToolUse and SubagentStop hooks that report tool output and subagent attribution (--no-subagents writes no hooks at all)",
    )

    actions.add_parser("status")
    doctor = actions.add_parser("doctor")
    doctor.add_argument("--vendor", choices=VENDORS)
    doctor.add_argument("--tenant")

    uninstall = actions.add_parser("uninstall")
    uninstall.add_argument("--vendor", required=True, choices=VENDORS)
    uninstall.add_argument("--purge", action="store_true")
    uninstall.add_argument("--dry-run", action="store_true")

    hook = actions.add_parser("hook")
    hook.add_argument("--vendor", required=True, choices=HOOK_VENDORS)
    hook.add_argument("--event", required=True)

    backfill = actions.add_parser("backfill", help="emit the assistant replies a coding agent wrote to disk but never exported")
    backfill.add_argument("--vendor", required=True, choices=BACKFILL_VENDORS)
    backfill.add_argument("--session", help="thread id to read (default: the most recent session)")
    backfill.add_argument("--last", type=int, help="read the N most recent sessions")
    backfill.add_argument("--since", help="skip sessions last updated before this ISO timestamp")
    backfill.add_argument("--all", action="store_true", help="read every session on disk")
    backfill.add_argument("--dry-run", action="store_true", help="print what would be emitted and emit nothing")
    backfill.add_argument("--force", action="store_true", help="ignore the stored offsets and re-read each session from the start")

    evaluation = sub.add_parser("eval", help="run an evaluation and gate a CI step on its verdict")
    eval_actions = evaluation.add_subparsers(dest="action", required=True)
    run = eval_actions.add_parser(
        "run",
        help="run Eval() from eval.json or module.py[:attr]; exit 0 passed, 1 failed, 2 undecided, 3 error",
    )
    run.add_argument("target", help="eval.json, or module.py[:attr] / package.module[:attr] (attr defaults to 'eval')")
    _eval_flags(run)
    gate = eval_actions.add_parser("gate", help="read an existing run's gate and exit with the same codes")
    gate.add_argument("run_id")
    _eval_flags(gate)
    return parser


def _eval_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--key", help="subscription key (default: EVALKIT_API_KEY, then the stored coding config)")
    parser.add_argument("--api-url", help="control-plane base URL (default: EVALKIT_API_URL, then https://api.syntropylabs.ai)")
    parser.add_argument("--json", action="store_true", help="print the run summary and gate as one JSON object")
    parser.add_argument(
        "--allow-undecided",
        action="store_true",
        help="exit 0 instead of 2 when the gate could not be decided (no gate, or a check without data)",
    )


def cmd_install(args: argparse.Namespace, store: ConfigStore, out) -> int:
    cfg = store.load()
    if args.key:
        cfg.key = args.key
    if args.endpoint:
        cfg.endpoint = args.endpoint.rstrip("/")
    if args.capture:
        cfg.capture = args.capture
    if args.tenant:
        cfg.tenant = args.tenant
    if not cfg.key:
        print("a subscription key is required: pass --key or set EVALKIT_API_KEY", file=sys.stderr)
        return 2
    opts = InstallOptions(scope=args.scope, capture=cfg.capture, traces=args.traces, dry_run=args.dry_run, subagents=args.subagents)
    installer = installer_for(args.vendor, store.home)
    if args.managed and isinstance(installer, ClaudeCodeInstaller):
        print(json.dumps(installer.managed_snippet(cfg, opts), indent=2), file=out)
        return 0
    saved = store.load_state()
    existing = _vendor_state(saved.vendors.get(args.vendor))
    try:
        state, changes = installer.install(cfg, opts, existing)
    except MalformedFile as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    prefix = "would write" if args.dry_run else "wrote"
    for change in changes:
        print(f"{prefix} {change.path}: {change.summary}", file=out)
    if not args.dry_run:
        store.save(cfg)
        saved.vendors[args.vendor] = {"scope": state.scope, "files": state.files, "capture": state.capture, "traces": state.traces}
        store.save_state(saved)
        print(f"{args.vendor} connected to {cfg.endpoint} (key {cfg.masked_key()})", file=out)
    return 0


def cmd_uninstall(args: argparse.Namespace, store: ConfigStore, out) -> int:
    saved = store.load_state()
    existing = _vendor_state(saved.vendors.get(args.vendor))
    if existing is None:
        print(f"{args.vendor} is not installed", file=out)
    else:
        installer = installer_for(args.vendor, store.home)
        changes = installer.uninstall(existing, args.dry_run)
        prefix = "would update" if args.dry_run else "updated"
        for change in changes:
            print(f"{prefix} {change.path}: {change.summary}", file=out)
        if not args.dry_run:
            del saved.vendors[args.vendor]
            store.save_state(saved)
    if args.purge and not args.dry_run and not saved.vendors:
        store.purge()
        print("removed shared evalkit config", file=out)
    return 0


def _vendor_state(entry) -> Optional[VendorState]:
    if not entry:
        return None
    return VendorState(
        scope=entry["scope"],
        files=entry["files"],
        capture=entry.get("capture", "minimal"),
        traces=bool(entry.get("traces", False)),
    )


def _hook_summary(status) -> str:
    events = status.get("hooks") or []
    if "subagents" in status:
        listed = f" ({', '.join(events)})" if events and status["subagents"] else ""
        return f" · agent hooks: {'on' if status['subagents'] else 'off'}{listed}"
    if events:
        return f" · hooks: {len(events)} events"
    return ""


def cmd_status(store: ConfigStore, out) -> int:
    cfg = store.load()
    saved = store.load_state()
    print(f"endpoint: {cfg.endpoint}", file=out)
    print(f"key: {cfg.masked_key()}", file=out)
    print(f"content capture: {cfg.capture}", file=out)
    print(f"last doctor: {_doctor_line(saved.doctor)}", file=out)
    for vendor in VENDORS:
        installer = installer_for(vendor, store.home)
        status = installer.status(_vendor_state(saved.vendors.get(vendor)))
        if not status["installed"]:
            print(f"{vendor}: not installed", file=out)
            continue
        files = ", ".join(f"{path}{'' if present else ' (missing)'}" for path, present in status["files"].items())
        detail = f"capture {status['capture']} · traces {'on' if status['traces'] else 'off'}{_hook_summary(status)}"
        print(f"{vendor}: installed ({status['scope']}) · {detail} → {files}", file=out)
    return 0


def _doctor_line(entry) -> str:
    if not entry:
        return "never run"
    return f"{'ok' if entry.get('ok') else 'failed'} at {entry.get('at')} ({entry.get('detail')})"


def cmd_doctor(args: argparse.Namespace, store: ConfigStore, out) -> int:
    cfg = store.load()
    if not cfg.key:
        print("no key configured; run `evalkit coding install` first", file=sys.stderr)
        return 2
    saved = store.load_state()
    ok, detail = doctor_run.run(cfg, args.tenant or cfg.tenant or None)
    print(f"round trip: {'ok' if ok else 'failed'} ({detail}) → exporter {cfg.endpoint}", file=out)
    saved.doctor = {"at": _timestamp(), "ok": ok, "detail": detail}
    store.save_state(saved)
    vendors = [args.vendor] if args.vendor else [v for v in VENDORS if v in saved.vendors]
    for vendor in vendors:
        installer = installer_for(vendor, store.home)
        status = installer.status(_vendor_state(saved.vendors.get(vendor)))
        if not status["installed"]:
            print(f"{vendor}: not installed", file=out)
            continue
        print(f"{vendor}: installed ({status['scope']}) · capture {status['capture']} · traces {'on' if status['traces'] else 'off'}{_hook_summary(status)}", file=out)
    if not vendors:
        print("nothing installed yet", file=out)
    return 0 if ok else 1


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: Optional[List[str]] = None, home: Optional[Path] = None, out=None) -> int:
    out = out or sys.stdout
    args = _parser().parse_args(argv)
    store = ConfigStore(home)
    if args.command == "eval":
        from .evaluation import cmd_eval_gate, cmd_eval_run

        return cmd_eval_run(args, store, out) if args.action == "run" else cmd_eval_gate(args, store, out)
    if args.command != "coding":
        return 2
    if args.action == "install":
        return cmd_install(args, store, out)
    if args.action == "uninstall":
        return cmd_uninstall(args, store, out)
    if args.action == "status":
        return cmd_status(store, out)
    if args.action == "doctor":
        return cmd_doctor(args, store, out)
    if args.action == "hook":
        from .hook import run_hook

        return run_hook(args.vendor, args.event)
    if args.action == "backfill":
        from .codex_capture import cmd_backfill

        return cmd_backfill(args, store, out)
    return 2


if __name__ == "__main__":
    sys.exit(main())
