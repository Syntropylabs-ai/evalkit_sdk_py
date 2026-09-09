from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from . import turns
from .config import SharedConfig
from .opencode_plugin import PLUGIN_FILE_NAME, PLUGIN_SOURCE
from .files import (
    delete_nested,
    get_nested,
    has_toml_table_outside,
    read_json,
    replace_marker_block,
    set_nested,
    write_atomic,
    write_json,
)

VENDORS = ("claude-code", "codex", "gemini-cli", "cursor", "windsurf", "opencode")
TOML_BEGIN = "# evalkit-managed-begin"
TOML_END = "# evalkit-managed-end"
ENV_BEGIN = "# evalkit-begin"
ENV_END = "# evalkit-end"


@dataclass
class InstallOptions:
    scope: str = "user"
    capture: str = "minimal"
    traces: bool = True
    dry_run: bool = False
    subagents: bool = True
    cwd: Path = field(default_factory=Path.cwd)


@dataclass
class Change:
    path: Path
    summary: str


@dataclass
class VendorState:
    scope: str
    files: Dict[str, Dict[str, Any]]
    capture: str = "minimal"
    traces: bool = True


class VendorInstaller(Protocol):
    name: str

    def install(self, cfg: SharedConfig, opts: InstallOptions, existing: Optional[VendorState] = None) -> tuple[VendorState, List[Change]]: ...

    def uninstall(self, state: VendorState, dry_run: bool) -> List[Change]: ...

    def status(self, state: Optional[VendorState]) -> Dict[str, Any]: ...


def _pathkey(path: Path) -> str:
    return str(path)


def hook_binary() -> str:
    found = shutil.which("evalkit")
    return _shell_word(found) if found else f"{_shell_word(sys.executable)} -m evalkit_cli.main"


def _shell_word(text: str) -> str:
    return json.dumps(text) if any(ch.isspace() for ch in text) else text


def _owned_hook(marker: str, hook: Any) -> bool:
    return isinstance(hook, dict) and marker in str(hook.get("command", ""))


def _status_of(state: Optional[VendorState]) -> Dict[str, Any]:
    if not state:
        return {"installed": False}
    events: List[str] = []
    for info in state.files.values():
        events.extend(info.get("hookEvents") or [])
    return {
        "installed": True,
        "scope": state.scope,
        "capture": state.capture,
        "traces": state.traces,
        "hooks": events,
        "files": {raw: Path(raw).exists() for raw in state.files},
    }


def _state(opts: InstallOptions, files: Dict[str, Dict[str, Any]]) -> VendorState:
    return VendorState(scope=opts.scope, files=files, capture=opts.capture, traces=opts.traces)


def _carry_files(existing: Optional[VendorState], written: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    files = dict(existing.files) if existing else {}
    files.update(written)
    return files


class _JsonKeysInstaller:
    name = ""
    tool_dir = ""
    file_name = "settings.json"

    def target(self, opts: InstallOptions, home: Path) -> Path:
        base = opts.cwd if opts.scope == "project" else home
        return base / self.tool_dir / self.file_name

    def keys(self, cfg: SharedConfig, opts: InstallOptions) -> Dict[str, Any]:
        raise NotImplementedError

    def prefix(self) -> List[str]:
        raise NotImplementedError

    def __init__(self, home: Path) -> None:
        self.home = home

    def merge_extra(self, data: Dict[str, Any], opts: InstallOptions, info: Dict[str, Any]) -> List[str]:
        return []

    def restore_extra(self, data: Dict[str, Any], info: Dict[str, Any]) -> List[str]:
        return []

    def install(self, cfg: SharedConfig, opts: InstallOptions, existing: Optional[VendorState] = None) -> tuple[VendorState, List[Change]]:
        path = self.target(opts, self.home)
        data = read_json(path)
        desired = self.keys(cfg, opts)
        remembered = ((existing.files.get(_pathkey(path)) or {}) if existing else {})
        owned = set(remembered.get("keys") or [])
        previous: Dict[str, Any] = dict(remembered.get("previous") or {})
        for key, value in desired.items():
            found, old = get_nested(data, [*self.prefix(), key])
            if found and key not in owned:
                previous[key] = old
            set_nested(data, [*self.prefix(), key], value)
        for key in sorted(owned - set(desired)):
            delete_nested(data, [*self.prefix(), key])
            if key in previous:
                set_nested(data, [*self.prefix(), key], previous.pop(key))
        info: Dict[str, Any] = {"keys": sorted(desired), "previous": previous, "prefix": self.prefix()}
        summary = [f"set {len(desired)} keys under {'.'.join(self.prefix())}", *self.merge_extra(data, opts, info)]
        if not opts.dry_run:
            write_json(path, data)
        return _state(opts, _carry_files(existing, {_pathkey(path): info})), [Change(path, "; ".join(summary))]

    def uninstall(self, state: VendorState, dry_run: bool) -> List[Change]:
        changes: List[Change] = []
        for raw, info in state.files.items():
            path = Path(raw)
            if not path.exists():
                continue
            data = read_json(path)
            prefix = list(info.get("prefix") or self.prefix())
            for key in info.get("keys", []):
                delete_nested(data, [*prefix, key])
            for key, old in (info.get("previous") or {}).items():
                set_nested(data, [*prefix, key], old)
            summary = [f"removed {len(info.get('keys', []))} keys", *self.restore_extra(data, info)]
            if not dry_run:
                write_json(path, data)
            changes.append(Change(path, "; ".join(summary)))
        return changes

    def status(self, state: Optional[VendorState]) -> Dict[str, Any]:
        return _status_of(state)


class ClaudeCodeInstaller(_JsonKeysInstaller):
    name = "claude-code"
    tool_dir = ".claude"
    hook_events = ("PostToolUse", "SubagentStop")
    hook_timeout_s = 5

    def prefix(self) -> List[str]:
        return ["env"]

    def marker(self) -> str:
        return f"coding hook --vendor {self.name}"

    def hook_group(self, event: str) -> Dict[str, Any]:
        command = f"{hook_binary()} coding hook --vendor {self.name} --event {event}"
        return {"hooks": [{"type": "command", "command": command, "timeout": self.hook_timeout_s}]}

    def _without_ours(self, groups: Any, marker: str) -> tuple[List[Any], int]:
        kept: List[Any] = []
        removed = 0
        for group in groups if isinstance(groups, list) else []:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                kept.append(group)
                continue
            inner = [h for h in group["hooks"] if not _owned_hook(marker, h)]
            removed += len(group["hooks"]) - len(inner)
            if inner:
                kept.append({**group, "hooks": inner})
        return kept, removed

    def _strip_hooks(self, data: Dict[str, Any], marker: str) -> int:
        hooks = data.get("hooks")
        if not isinstance(hooks, dict):
            return 0
        removed = 0
        for event in list(hooks):
            if not isinstance(hooks[event], list):
                continue
            kept, gone = self._without_ours(hooks[event], marker)
            if gone == 0:
                continue
            removed += gone
            if kept:
                hooks[event] = kept
            else:
                del hooks[event]
        if not hooks:
            del data["hooks"]
        return removed

    def merge_extra(self, data: Dict[str, Any], opts: InstallOptions, info: Dict[str, Any]) -> List[str]:
        if not opts.subagents:
            removed = self._strip_hooks(data, self.marker())
            return [f"removed {removed} agent hooks"] if removed else []
        self._strip_hooks(data, self.marker())
        hooks = data.get("hooks")
        if not isinstance(hooks, dict):
            hooks = {}
            data["hooks"] = hooks
        for event in self.hook_events:
            current = hooks.get(event)
            if current is not None and not isinstance(current, list):
                raise RuntimeError(f"hooks.{event} in the settings file must be a list of hook groups; fix it before installing")
            hooks[event] = [*(current or []), self.hook_group(event)]
        info["hookEvents"] = list(self.hook_events)
        info["marker"] = self.marker()
        return [f"merged {len(self.hook_events)} agent hooks"]

    def restore_extra(self, data: Dict[str, Any], info: Dict[str, Any]) -> List[str]:
        if "hookEvents" not in info:
            return []
        removed = self._strip_hooks(data, info.get("marker") or self.marker())
        return [f"removed {removed} agent hooks"]

    def status(self, state: Optional[VendorState]) -> Dict[str, Any]:
        result = super().status(state)
        if state:
            result["subagents"] = any("hookEvents" in info for info in state.files.values())
        return result

    def keys(self, cfg: SharedConfig, opts: InstallOptions) -> Dict[str, Any]:
        env: Dict[str, Any] = {
            "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
            "OTEL_METRICS_EXPORTER": "otlp",
            "OTEL_LOGS_EXPORTER": "otlp",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
            "OTEL_EXPORTER_OTLP_ENDPOINT": cfg.endpoint,
            "OTEL_EXPORTER_OTLP_HEADERS": f"Authorization=Bearer {cfg.key}",
            "OTEL_RESOURCE_ATTRIBUTES": "evalkit.vendor=claude-code",
            "OTEL_LOGS_EXPORT_INTERVAL": "5000",
        }
        if opts.traces:
            env["CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"] = "1"
            env["OTEL_TRACES_EXPORTER"] = "otlp"
            env["OTEL_TRACES_EXPORT_INTERVAL"] = "1000"
        if opts.capture == "full":
            env["OTEL_LOG_USER_PROMPTS"] = "1"
            env["OTEL_LOG_ASSISTANT_RESPONSES"] = "1"
            env["OTEL_LOG_TOOL_DETAILS"] = "1"
            env["OTEL_LOG_TOOL_CONTENT"] = "1"
            env["OTEL_LOG_RAW_API_BODIES"] = "1"
            env["CLAUDE_CODE_OTEL_CONTENT_MAX_LENGTH"] = "524288"
        return env

    def managed_snippet(self, cfg: SharedConfig, opts: InstallOptions) -> Dict[str, Any]:
        env = self.keys(cfg, opts)
        env["OTEL_EXPORTER_OTLP_HEADERS"] = "Authorization=Bearer <EVALKIT_KEY>"
        return {"env": env}


class GeminiCliInstaller(_JsonKeysInstaller):
    name = "gemini-cli"
    tool_dir = ".gemini"

    def prefix(self) -> List[str]:
        return ["telemetry"]

    def keys(self, cfg: SharedConfig, opts: InstallOptions) -> Dict[str, Any]:
        return {
            "enabled": True,
            "target": "local",
            "otlpEndpoint": cfg.endpoint,
            "otlpProtocol": "http",
            "logPrompts": opts.capture == "full",
            "traces": bool(opts.traces),
        }

    def install(self, cfg: SharedConfig, opts: InstallOptions, existing: Optional[VendorState] = None) -> tuple[VendorState, List[Change]]:
        state, changes = super().install(cfg, opts, existing)
        env_path = self.target(opts, self.home).with_name(".env")
        text = env_path.read_text() if env_path.exists() else ""
        block = f"OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer {cfg.key}"
        if not opts.dry_run:
            write_atomic(env_path, replace_marker_block(text, ENV_BEGIN, ENV_END, block), mode=0o600)
        state.files = _carry_files(existing, {**state.files, _pathkey(env_path): {"block": [ENV_BEGIN, ENV_END]}})
        changes.append(Change(env_path, "wrote OTEL_EXPORTER_OTLP_HEADERS between evalkit markers"))
        return state, changes

    def uninstall(self, state: VendorState, dry_run: bool) -> List[Change]:
        changes: List[Change] = []
        for raw, info in state.files.items():
            path = Path(raw)
            if "block" in info and path.exists():
                begin, end = info["block"]
                if not dry_run:
                    write_atomic(path, replace_marker_block(path.read_text(), begin, end, None))
                changes.append(Change(path, "removed evalkit marker block"))
        keyed = VendorState(scope=state.scope, files={raw: info for raw, info in state.files.items() if "keys" in info})
        return changes + super().uninstall(keyed, dry_run)


class CodexInstaller:
    name = "codex"

    def __init__(self, home: Path) -> None:
        self.home = home

    def target(self, opts: InstallOptions) -> Path:
        base = opts.cwd if opts.scope == "project" else self.home
        return base / ".codex" / "config.toml"

    def block(self, cfg: SharedConfig, opts: InstallOptions) -> str:
        endpoint = cfg.endpoint.rstrip("/")
        headers = f'headers = {{ "Authorization" = "Bearer {cfg.key}" }}'

        def exporter(signal: str) -> str:
            return f'{{ otlp-http = {{ endpoint = "{endpoint}/v1/{signal}", protocol = "binary", {headers} }} }}'

        prompts = "true" if opts.capture == "full" else "false"
        traces = exporter("traces") if opts.traces else '"none"'
        lines = [
            "[otel]",
            'environment = "prod"',
            f"log_user_prompt = {prompts}",
            f"exporter = {exporter('logs')}",
            f"trace_exporter = {traces}",
            f"metrics_exporter = {exporter('metrics')}",
        ]
        return "\n".join(lines)

    def install(self, cfg: SharedConfig, opts: InstallOptions, existing: Optional[VendorState] = None) -> tuple[VendorState, List[Change]]:
        path = self.target(opts)
        text = path.read_text() if path.exists() else ""
        if has_toml_table_outside(text, "otel", TOML_BEGIN, TOML_END):
            raise RuntimeError(f"{path} already has an [otel] table outside the evalkit markers; remove or move it first")
        if not opts.dry_run:
            write_atomic(path, replace_marker_block(text, TOML_BEGIN, TOML_END, self.block(cfg, opts)), mode=0o600)
        state = _state(opts, _carry_files(existing, {_pathkey(path): {"block": [TOML_BEGIN, TOML_END]}}))
        return state, [Change(path, "wrote [otel] block between evalkit markers")]

    def uninstall(self, state: VendorState, dry_run: bool) -> List[Change]:
        changes: List[Change] = []
        for raw, info in state.files.items():
            path = Path(raw)
            if not path.exists():
                continue
            begin, end = info["block"]
            if not dry_run:
                write_atomic(path, replace_marker_block(path.read_text(), begin, end, None))
            changes.append(Change(path, "removed [otel] block"))
        return changes

    def status(self, state: Optional[VendorState]) -> Dict[str, Any]:
        return _status_of(state)


class _HooksInstaller:
    name = ""
    user_path: tuple[str, ...] = ()
    project_path: tuple[str, ...] = ()
    events: tuple[str, ...] = ()

    versioned = False
    timeout_ms: Optional[int] = None

    def __init__(self, home: Path) -> None:
        self.home = home

    def target(self, opts: InstallOptions) -> Path:
        return (opts.cwd.joinpath(*self.project_path)) if opts.scope == "project" else self.home.joinpath(*self.user_path)

    def marker(self) -> str:
        return f"coding hook --vendor {self.name}"

    def command(self, event: str) -> str:
        return f"{hook_binary()} coding hook --vendor {self.name} --event {event}"

    def entry(self, event: str) -> Dict[str, Any]:
        item: Dict[str, Any] = {"command": self.command(event)}
        if self.timeout_ms is not None:
            item["timeout"] = self.timeout_ms
        return item

    def _strip(self, data: Dict[str, Any], marker: str) -> int:
        hooks = data.get("hooks")
        if not isinstance(hooks, dict):
            return 0
        removed = 0
        for event in list(hooks):
            entries = hooks.get(event)
            if not isinstance(entries, list):
                continue
            kept = [h for h in entries if not _owned_hook(marker, h)]
            if len(kept) == len(entries):
                continue
            removed += len(entries) - len(kept)
            if kept:
                hooks[event] = kept
            else:
                del hooks[event]
        if not hooks:
            del data["hooks"]
        return removed

    def install(self, cfg: SharedConfig, opts: InstallOptions, existing: Optional[VendorState] = None) -> tuple[VendorState, List[Change]]:
        path = self.target(opts)
        data = read_json(path)
        if self.versioned:
            data.setdefault("version", 1)
        for event in self.events:
            current = (data.get("hooks") or {}).get(event) if isinstance(data.get("hooks"), dict) else None
            if current is not None and not isinstance(current, list):
                raise RuntimeError(f"hooks.{event} in {path} must be a list of hook entries; fix it before installing")
        self._strip(data, self.marker())
        hooks = data.get("hooks")
        if not isinstance(hooks, dict):
            hooks = {}
            data["hooks"] = hooks
        for event in self.events:
            current = hooks.get(event)
            hooks[event] = [*(current or []), self.entry(event)]
        if not opts.dry_run:
            write_json(path, data)
        state = _state(opts, _carry_files(existing, {_pathkey(path): {"hookEvents": list(self.events), "marker": self.marker()}}))
        return state, [Change(path, f"merged {len(self.events)} hook events")]

    def uninstall(self, state: VendorState, dry_run: bool) -> List[Change]:
        changes: List[Change] = []
        for raw, info in state.files.items():
            path = Path(raw)
            if not path.exists():
                continue
            data = read_json(path)
            self._strip(data, info.get("marker") or self.marker())
            if not dry_run:
                write_json(path, data)
            changes.append(Change(path, "removed evalkit hook entries"))
        return changes

    def status(self, state: Optional[VendorState]) -> Dict[str, Any]:
        return _status_of(state)


class CursorInstaller(_HooksInstaller):
    name = "cursor"
    user_path = (".cursor", "hooks.json")
    project_path = (".cursor", "hooks.json")
    versioned = True
    timeout_ms = 5000
    events = turns.events("cursor")


class WindsurfInstaller(_HooksInstaller):
    name = "windsurf"
    user_path = (".codeium", "windsurf", "hooks.json")
    project_path = (".windsurf", "hooks.json")
    events = turns.events("windsurf")


class OpenCodeInstaller:
    name = "opencode"
    array_key = "plugin"

    def __init__(self, home: Path) -> None:
        self.home = home

    def targets(self, opts: InstallOptions) -> tuple[Path, Path, str]:
        if opts.scope == "project":
            return opts.cwd / "opencode.json", opts.cwd / ".opencode" / "plugins" / PLUGIN_FILE_NAME, f"./.opencode/plugins/{PLUGIN_FILE_NAME}"
        base = self.home / ".config" / "opencode"
        return base / "opencode.json", base / "plugins" / PLUGIN_FILE_NAME, f"./plugins/{PLUGIN_FILE_NAME}"

    def install(self, cfg: SharedConfig, opts: InstallOptions, existing: Optional[VendorState] = None) -> tuple[VendorState, List[Change]]:
        config_path, plugin_path, reference = self.targets(opts)
        data = read_json(config_path)
        entries = data.get(self.array_key)
        if entries is not None and not isinstance(entries, list):
            raise RuntimeError(f'{config_path} has a non-list "{self.array_key}" value; fix it before installing')
        data[self.array_key] = [*[e for e in (entries or []) if e != reference], reference]
        if not opts.dry_run:
            write_json(config_path, data)
            write_atomic(plugin_path, PLUGIN_SOURCE, mode=0o644)
        files = {
            _pathkey(config_path): {"arrayKey": self.array_key, "value": reference},
            _pathkey(plugin_path): {"file": True},
        }
        changes = [Change(config_path, f'registered {reference} in "{self.array_key}"'), Change(plugin_path, "wrote the evalkit OpenCode plugin")]
        return _state(opts, _carry_files(existing, files)), changes

    def uninstall(self, state: VendorState, dry_run: bool) -> List[Change]:
        changes: List[Change] = []
        for raw, info in state.files.items():
            path = Path(raw)
            if not path.exists():
                continue
            if info.get("file"):
                if not dry_run:
                    path.unlink()
                    if not any(path.parent.iterdir()):
                        path.parent.rmdir()
                changes.append(Change(path, "removed the evalkit OpenCode plugin"))
                continue
            key, value = info.get("arrayKey"), info.get("value")
            data = read_json(path)
            entries = data.get(key)
            if isinstance(entries, list):
                kept = [e for e in entries if e != value]
                if kept:
                    data[key] = kept
                else:
                    data.pop(key, None)
            if not dry_run:
                write_json(path, data)
            changes.append(Change(path, f'removed {value} from "{key}"'))
        return changes

    def status(self, state: Optional[VendorState]) -> Dict[str, Any]:
        return _status_of(state)


def installer_for(vendor: str, home: Path):
    registry = {
        "claude-code": ClaudeCodeInstaller,
        "codex": CodexInstaller,
        "gemini-cli": GeminiCliInstaller,
        "cursor": CursorInstaller,
        "windsurf": WindsurfInstaller,
        "opencode": OpenCodeInstaller,
    }
    if vendor not in registry:
        raise ValueError(f"unknown vendor {vendor!r}; choose from {', '.join(VENDORS)}")
    return registry[vendor](home)
