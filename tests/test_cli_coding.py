import io
import json
import os
import stat
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from evalkit_cli import turns
from evalkit_cli.hook import build_records
from evalkit_cli.main import main

KEY = "tk_live_0123456789abcdef0123456789abcdef"
CURSOR_EVENTS = list(turns.events("cursor"))
WINDSURF_EVENTS = list(turns.events("windsurf"))
NOW_NS = 1_700_000_000_000_000_000


def record_of(vendor, event, payload, capture_content):
    records = build_records(vendor, event, payload, capture_content, NOW_NS)
    return records[0][1] if records else None


def run(argv, home, cwd=None):
    out = io.StringIO()
    old = Path.cwd()
    if cwd:
        os.chdir(cwd)
    try:
        code = main(argv, home=Path(home), out=out)
    finally:
        os.chdir(old)
    return code, out.getvalue()


def _capture_server(status=200):
    received = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            ln = int(self.headers.get("Content-Length", 0))
            received.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": json.loads(self.rfile.read(ln) or b"{}")})
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, received


def _attrs(items):
    return {a["key"]: list(a["value"].values())[0] for a in items or []}


def _receiver():
    store = {"posts": [], "gets": [], "spans": {}, "enrichments": []}

    class H(BaseHTTPRequestHandler):
        def _json(self, status, body):
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            ln = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(ln) or b"{}")
            store["posts"].append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
            for resource in body.get("resourceSpans", []):
                for scope in resource.get("scopeSpans", []):
                    for span in scope.get("spans", []):
                        row = {
                            "traceId": span["traceId"],
                            "spanId": span["spanId"],
                            "parentSpanId": span.get("parentSpanId"),
                            "name": span["name"],
                            "attrs": _attrs(span.get("attributes")),
                        }
                        store["spans"].setdefault(row["traceId"], []).append(row)
            for resource in body.get("resourceLogs", []):
                for scope in resource.get("scopeLogs", []):
                    for record in scope.get("logRecords", []):
                        store["enrichments"].append({"traceId": record.get("traceId"), "spanId": record.get("spanId"), "attrs": _attrs(record.get("attributes"))})
            self._json(200, {})

        def do_GET(self):
            path = self.path.split("?")[0].strip("/").split("/")
            tenant, trace = path[-2], path[-1]
            store["gets"].append({"tenant": tenant, "trace": trace, "auth": self.headers.get("Authorization")})
            rows = []
            for span in store["spans"].get(trace, []):
                row = {"traceId": span["traceId"], "spanId": span["spanId"], "prompt": span["attrs"].get("prompt", ""), "completion": ""}
                for enrichment in store["enrichments"]:
                    if enrichment["spanId"] == span["spanId"]:
                        row["completion"] = enrichment["attrs"].get("completion", row["completion"])
                        row["prompt"] = row["prompt"] or enrichment["attrs"].get("prompt", "")
                rows.append(row)
            self._json(200, {"data": rows, "meta": {"count": len(rows)}})

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, store


class TestClaudeCode(unittest.TestCase):
    def test_install_merges_env_and_uninstall_restores(self):
        with TemporaryDirectory() as home:
            settings = Path(home) / ".claude" / "settings.json"
            settings.parent.mkdir()
            settings.write_text(json.dumps({"permissions": {"allow": ["Bash"]}, "env": {"OTEL_LOGS_EXPORTER": "console", "FOO": "bar"}}))
            code, out = run(["coding", "install", "--vendor", "claude-code", "--key", KEY, "--endpoint", "https://x.test/otlp/"], home)
            self.assertEqual(code, 0, out)
            self.assertNotIn(KEY, out)
            data = json.loads(settings.read_text())
            self.assertEqual(data["permissions"], {"allow": ["Bash"]})
            self.assertEqual(data["env"]["FOO"], "bar")
            self.assertEqual(data["env"]["OTEL_LOGS_EXPORTER"], "otlp")
            self.assertEqual(data["env"]["OTEL_EXPORTER_OTLP_ENDPOINT"], "https://x.test/otlp")
            self.assertEqual(data["env"]["OTEL_EXPORTER_OTLP_HEADERS"], f"Authorization=Bearer {KEY}")
            self.assertNotIn("OTEL_LOG_USER_PROMPTS", data["env"])
            self.assertEqual(data["env"]["CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"], "1")
            self.assertEqual(data["env"]["OTEL_TRACES_EXPORTER"], "otlp")
            self.assertEqual(data["env"]["OTEL_TRACES_EXPORT_INTERVAL"], "1000")
            cfg = Path(home) / ".config" / "evalkit" / "config.env"
            self.assertEqual(stat.S_IMODE(cfg.stat().st_mode), 0o600)
            self.assertIn(f"EVALKIT_API_KEY={KEY}", cfg.read_text())
            state = json.loads((Path(home) / ".config" / "evalkit" / "state.json").read_text())
            self.assertEqual(state["vendors"]["claude-code"]["files"][str(settings)]["previous"], {"OTEL_LOGS_EXPORTER": "console"})

            before = settings.read_text()
            code, _ = run(["coding", "install", "--vendor", "claude-code"], home)
            self.assertEqual(code, 0)
            self.assertEqual(settings.read_text(), before)

            code, out = run(["coding", "uninstall", "--vendor", "claude-code"], home)
            self.assertEqual(code, 0, out)
            data = json.loads(settings.read_text())
            self.assertEqual(data["env"], {"OTEL_LOGS_EXPORTER": "console", "FOO": "bar"})
            self.assertEqual(data["permissions"], {"allow": ["Bash"]})

    def test_no_traces_opts_out_of_span_mode(self):
        with TemporaryDirectory() as home:
            code, out = run(["coding", "install", "--vendor", "claude-code", "--key", KEY, "--no-traces"], home)
            self.assertEqual(code, 0, out)
            env = json.loads((Path(home) / ".claude" / "settings.json").read_text())["env"]
            self.assertNotIn("OTEL_TRACES_EXPORTER", env)
            self.assertNotIn("CLAUDE_CODE_ENHANCED_TELEMETRY_BETA", env)
            self.assertNotIn("OTEL_TRACES_EXPORT_INTERVAL", env)
            code, out = run(["coding", "status"], home)
            self.assertIn("traces off", out)

    def test_full_capture_traces_and_dry_run(self):
        with TemporaryDirectory() as home:
            code, out = run(["coding", "install", "--vendor", "claude-code", "--key", KEY, "--capture", "full", "--traces", "--dry-run"], home)
            self.assertEqual(code, 0)
            self.assertIn("would write", out)
            self.assertFalse((Path(home) / ".claude" / "settings.json").exists())
            self.assertFalse((Path(home) / ".config" / "evalkit").exists())
            code, _ = run(["coding", "install", "--vendor", "claude-code", "--key", KEY, "--capture", "full", "--traces"], home)
            env = json.loads((Path(home) / ".claude" / "settings.json").read_text())["env"]
            self.assertEqual(env["OTEL_LOG_USER_PROMPTS"], "1")
            self.assertEqual(env["OTEL_LOG_TOOL_CONTENT"], "1")
            self.assertEqual(env["OTEL_LOG_ASSISTANT_RESPONSES"], "1")
            self.assertEqual(env["OTEL_LOG_RAW_API_BODIES"], "1")
            self.assertEqual(env["CLAUDE_CODE_OTEL_CONTENT_MAX_LENGTH"], "524288")
            self.assertEqual(env["OTEL_TRACES_EXPORTER"], "otlp")
            self.assertEqual(env["CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"], "1")

    def test_managed_snippet_hides_key(self):
        with TemporaryDirectory() as home:
            code, out = run(["coding", "install", "--vendor", "claude-code", "--key", KEY, "--managed"], home)
            self.assertEqual(code, 0)
            self.assertNotIn(KEY, out)
            self.assertIn("<EVALKIT_KEY>", out)
            self.assertFalse((Path(home) / ".claude").exists())

    def test_missing_key_fails(self):
        with TemporaryDirectory() as home:
            code, _ = run(["coding", "install", "--vendor", "claude-code"], home)
            self.assertEqual(code, 2)


class TestCodex(unittest.TestCase):
    def test_block_is_written_between_markers_and_removed(self):
        with TemporaryDirectory() as home:
            cfg = Path(home) / ".codex" / "config.toml"
            cfg.parent.mkdir()
            cfg.write_text('model = "gpt-5"\n\n[sandbox]\nmode = "read-only"\n')
            code, out = run(["coding", "install", "--vendor", "codex", "--key", KEY, "--endpoint", "https://x.test/otlp"], home)
            self.assertEqual(code, 0, out)
            text = cfg.read_text()
            self.assertTrue(text.startswith('model = "gpt-5"'))
            self.assertIn("[sandbox]", text)
            self.assertIn("# evalkit-managed-begin\n[otel]\n", text)
            self.assertIn('endpoint = "https://x.test/otlp/v1/logs", protocol = "binary"', text)
            self.assertIn(f'"Authorization" = "Bearer {KEY}"', text)
            self.assertIn('endpoint = "https://x.test/otlp/v1/traces", protocol = "binary"', text)
            self.assertIn("log_user_prompt = false", text)
            before = text
            run(["coding", "install", "--vendor", "codex"], home)
            self.assertEqual(cfg.read_text(), before)
            run(["coding", "install", "--vendor", "codex", "--no-traces", "--capture", "full"], home)
            text = cfg.read_text()
            self.assertEqual(text.count("[otel]"), 1)
            self.assertIn('trace_exporter = "none"', text)
            self.assertIn("log_user_prompt = true", text)
            code, _ = run(["coding", "uninstall", "--vendor", "codex"], home)
            self.assertEqual(code, 0)
            self.assertEqual(cfg.read_text(), 'model = "gpt-5"\n\n[sandbox]\nmode = "read-only"\n')

    def test_refuses_foreign_otel_table(self):
        with TemporaryDirectory() as home:
            cfg = Path(home) / ".codex" / "config.toml"
            cfg.parent.mkdir()
            cfg.write_text('[otel]\nexporter = "none"\n')
            code, _ = run(["coding", "install", "--vendor", "codex", "--key", KEY], home)
            self.assertEqual(code, 1)
            self.assertEqual(cfg.read_text(), '[otel]\nexporter = "none"\n')


class TestGeminiCli(unittest.TestCase):
    def test_settings_and_env_markers(self):
        with TemporaryDirectory() as home:
            settings = Path(home) / ".gemini" / "settings.json"
            settings.parent.mkdir()
            settings.write_text(json.dumps({"theme": "dark", "telemetry": {"enabled": False, "logPrompts": True}}))
            env = settings.with_name(".env")
            env.write_text("GEMINI_API_KEY=abc\n")
            code, out = run(["coding", "install", "--vendor", "gemini-cli", "--key", KEY, "--endpoint", "https://x.test/otlp"], home)
            self.assertEqual(code, 0, out)
            data = json.loads(settings.read_text())
            self.assertEqual(data["theme"], "dark")
            self.assertEqual(data["telemetry"], {"enabled": True, "target": "local", "otlpEndpoint": "https://x.test/otlp", "otlpProtocol": "http", "logPrompts": False, "traces": True})
            self.assertEqual(env.read_text(), f"GEMINI_API_KEY=abc\n\n# evalkit-begin\nOTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer {KEY}\n# evalkit-end\n")
            self.assertEqual(stat.S_IMODE(env.stat().st_mode), 0o600)
            code, _ = run(["coding", "uninstall", "--vendor", "gemini-cli"], home)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(settings.read_text()), {"theme": "dark", "telemetry": {"enabled": False, "logPrompts": True}})
            self.assertEqual(env.read_text(), "GEMINI_API_KEY=abc\n")


class TestHooksVendors(unittest.TestCase):
    def test_cursor_merge_is_idempotent_and_preserves_foreign_hooks(self):
        with TemporaryDirectory() as home:
            hooks = Path(home) / ".cursor" / "hooks.json"
            hooks.parent.mkdir()
            hooks.write_text(json.dumps({"version": 1, "hooks": {"stop": [{"command": "notify-send done"}]}}))
            code, out = run(["coding", "install", "--vendor", "cursor", "--key", KEY], home)
            self.assertEqual(code, 0, out)
            data = json.loads(hooks.read_text())
            self.assertEqual(data["version"], 1)
            self.assertEqual(sorted(data["hooks"]), sorted(CURSOR_EVENTS))
            self.assertEqual(data["hooks"]["stop"][0], {"command": "notify-send done"})
            self.assertIn("coding hook --vendor cursor --event stop", data["hooks"]["stop"][1]["command"])
            self.assertEqual(data["hooks"]["stop"][1]["timeout"], 5000)
            run(["coding", "install", "--vendor", "cursor"], home)
            data = json.loads(hooks.read_text())
            self.assertEqual(len(data["hooks"]["stop"]), 2)
            code, _ = run(["coding", "uninstall", "--vendor", "cursor"], home)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(hooks.read_text()), {"version": 1, "hooks": {"stop": [{"command": "notify-send done"}]}})

    def test_malformed_hooks_file_is_refused(self):
        with TemporaryDirectory() as home:
            hooks = Path(home) / ".cursor" / "hooks.json"
            hooks.parent.mkdir()
            hooks.write_text("{ not json")
            code, _ = run(["coding", "install", "--vendor", "cursor", "--key", KEY], home)
            self.assertEqual(code, 1)
            self.assertEqual(hooks.read_text(), "{ not json")

    def test_windsurf_project_scope_writes_into_cwd(self):
        with TemporaryDirectory() as home, TemporaryDirectory() as project:
            code, _ = run(["coding", "install", "--vendor", "windsurf", "--key", KEY, "--scope", "project"], home, cwd=project)
            self.assertEqual(code, 0)
            data = json.loads((Path(project) / ".windsurf" / "hooks.json").read_text())
            self.assertEqual(sorted(data["hooks"]), sorted(WINDSURF_EVENTS))
            self.assertNotIn("version", data)
            self.assertNotIn("timeout", data["hooks"]["pre_user_prompt"][0])


class TestStatusDoctorHook(unittest.TestCase):
    def test_status_masks_key(self):
        with TemporaryDirectory() as home:
            run(["coding", "install", "--vendor", "claude-code", "--key", KEY], home)
            code, out = run(["coding", "status"], home)
            self.assertEqual(code, 0)
            self.assertNotIn(KEY, out)
            self.assertIn("tk_live_…cdef", out)
            self.assertIn("claude-code: installed (user)", out)
            self.assertIn("codex: not installed", out)

    def test_doctor_posts_a_synthetic_turn_and_reads_the_merged_tree_back(self):
        srv, store = _receiver()
        with TemporaryDirectory() as home:
            try:
                endpoint = f"http://127.0.0.1:{srv.server_address[1]}/v1/otlp"
                run(["coding", "install", "--vendor", "codex", "--key", KEY, "--endpoint", endpoint, "--tenant", "tenant-1"], home)
                code, out = run(["coding", "doctor"], home)
            finally:
                srv.shutdown()
        self.assertEqual(code, 0, out)
        self.assertIn("round trip: ok", out)
        self.assertIn("3 spans merged, prompt present", out)
        self.assertIn("codex: installed (user) · capture minimal · traces on", out)
        self.assertEqual([call["path"] for call in store["posts"]], ["/v1/otlp/v1/traces", "/v1/otlp/v1/logs"])
        self.assertEqual(store["posts"][0]["auth"], f"Bearer {KEY}")
        self.assertEqual(store["gets"][0]["tenant"], "tenant-1")
        trace = store["gets"][0]["trace"]
        spans = store["spans"][trace]
        self.assertEqual([span["attrs"]["evalkit.span_type"] for span in spans], ["agent", "llm_call", "tool_call"])
        self.assertEqual([span["parentSpanId"] for span in spans[1:]], [spans[0]["spanId"], spans[0]["spanId"]])
        self.assertEqual(spans[0]["attrs"]["gen_ai.operation.name"], "invoke_agent")
        self.assertEqual(store["enrichments"][0]["spanId"], spans[0]["spanId"])
        self.assertTrue(store["enrichments"][0]["attrs"]["evalkit.enrichment"])
        self.assertIn("completion", store["enrichments"][0]["attrs"])

    def test_doctor_fails_and_is_remembered_when_the_receiver_rejects(self):
        srv, _ = _capture_server(500)
        with TemporaryDirectory() as home:
            try:
                endpoint = f"http://127.0.0.1:{srv.server_address[1]}/v1/otlp"
                run(["coding", "install", "--vendor", "codex", "--key", KEY, "--endpoint", endpoint], home)
                code, out = run(["coding", "doctor"], home)
            finally:
                srv.shutdown()
        self.assertEqual(code, 1, out)
        self.assertIn("posting spans failed", out)
        with TemporaryDirectory() as home:
            code, out = run(["coding", "doctor"], home)
            self.assertEqual(code, 2)

    def test_hook_posts_a_tool_span_drops_content_and_never_fails(self):
        srv, received = _capture_server()
        with TemporaryDirectory() as home:
            try:
                endpoint = f"http://127.0.0.1:{srv.server_address[1]}"
                run(["coding", "install", "--vendor", "cursor", "--key", KEY, "--endpoint", endpoint], home)
                payload = json.dumps(
                    {"conversation_id": "c1", "generation_id": "g1", "tool_name": "Bash", "tool_use_id": "tu1", "duration": 12, "tool_input": {"command": "ls"}, "tool_output": "secret text"}
                )
                proc = subprocess.run(
                    [sys.executable, "-m", "evalkit_cli.main", "coding", "hook", "--vendor", "cursor", "--event", "postToolUse"],
                    input=payload, capture_output=True, text=True, env={**os.environ, "HOME": home}, timeout=20,
                )
            finally:
                srv.shutdown()
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(proc.stdout, "")
            self.assertEqual(received[0]["path"], "/v1/traces")
            span = received[0]["body"]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
            attrs = {a["key"]: a["value"] for a in span["attributes"]}
            self.assertEqual(span["name"], "Bash")
            self.assertEqual(attrs["evalkit.span_type"], {"stringValue": "tool_call"})
            self.assertEqual(attrs["gen_ai.tool.name"], {"stringValue": "Bash"})
            self.assertEqual(attrs["gen_ai.tool.call.id"], {"stringValue": "tu1"})
            self.assertEqual(attrs["duration_ms"], {"intValue": "12"})
            self.assertNotIn("tool.result", attrs)
            self.assertNotIn("tool.arguments", attrs)
            resource = {a["key"]: a["value"] for a in received[0]["body"]["resourceSpans"][0]["resource"]["attributes"]}
            self.assertEqual(resource["agent.vendor"], {"stringValue": "cursor"})

    def test_hook_with_full_capture_and_unreachable_endpoint(self):
        with TemporaryDirectory() as home:
            run(["coding", "install", "--vendor", "cursor", "--key", KEY, "--endpoint", "http://127.0.0.1:9", "--capture", "full"], home)
            proc = subprocess.run(
                [sys.executable, "-m", "evalkit_cli.main", "coding", "hook", "--vendor", "cursor", "--event", "stop"],
                input="not json at all", capture_output=True, text=True, env={**os.environ, "HOME": home}, timeout=20,
            )
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(proc.stdout, "")

    def test_uninstall_purge_removes_shared_config(self):
        with TemporaryDirectory() as home:
            run(["coding", "install", "--vendor", "codex", "--key", KEY], home)
            code, out = run(["coding", "uninstall", "--vendor", "codex", "--purge"], home)
            self.assertEqual(code, 0, out)
            self.assertFalse((Path(home) / ".config" / "evalkit").exists())


class TestClaudeCodeSubagents(unittest.TestCase):
    ORIGINAL = {
        "env": {"FOO": "bar"},
        "hooks": {
            "SubagentStop": [{"matcher": "Explore", "hooks": [{"type": "command", "command": "notify-send done"}]}],
            "PreToolUse": [{"hooks": [{"type": "command", "command": "echo pre"}]}],
        },
    }

    def _settings(self, home):
        settings = Path(home) / ".claude" / "settings.json"
        settings.parent.mkdir()
        settings.write_text(json.dumps(self.ORIGINAL))
        return settings

    def test_hooks_merge_is_idempotent_and_uninstall_restores(self):
        with TemporaryDirectory() as home:
            settings = self._settings(home)
            code, out = run(["coding", "install", "--vendor", "claude-code", "--key", KEY], home)
            self.assertEqual(code, 0, out)
            self.assertIn("merged 2 agent hooks", out)
            data = json.loads(settings.read_text())
            post = data["hooks"]["PostToolUse"]
            self.assertEqual(len(post), 1)
            self.assertEqual(post[0]["hooks"][0]["type"], "command")
            self.assertEqual(post[0]["hooks"][0]["timeout"], 5)
            self.assertIn("coding hook --vendor claude-code --event PostToolUse", post[0]["hooks"][0]["command"])
            stop = data["hooks"]["SubagentStop"]
            self.assertEqual(stop[0], self.ORIGINAL["hooks"]["SubagentStop"][0])
            self.assertIn("--event SubagentStop", stop[1]["hooks"][0]["command"])
            self.assertEqual(data["hooks"]["PreToolUse"], self.ORIGINAL["hooks"]["PreToolUse"])
            self.assertNotIn("SubagentStart", data["hooks"])
            run(["coding", "install", "--vendor", "claude-code"], home)
            data = json.loads(settings.read_text())
            self.assertEqual(len(data["hooks"]["PostToolUse"]), 1)
            self.assertEqual(len(data["hooks"]["SubagentStop"]), 2)
            code, out = run(["coding", "status"], home)
            self.assertIn("agent hooks: on", out)
            code, out = run(["coding", "uninstall", "--vendor", "claude-code"], home)
            self.assertEqual(code, 0, out)
            self.assertIn("removed 2 agent hooks", out)
            self.assertEqual(json.loads(settings.read_text()), self.ORIGINAL)

    def test_reinstall_drops_owned_hooks_of_events_no_longer_installed(self):
        with TemporaryDirectory() as home:
            settings = self._settings(home)
            data = json.loads(settings.read_text())
            stale = f"{sys.executable} -m evalkit_cli.main coding hook --vendor claude-code --event SubagentStart"
            data["hooks"]["SubagentStart"] = [{"hooks": [{"type": "command", "command": stale, "timeout": 5}]}]
            settings.write_text(json.dumps(data))
            code, out = run(["coding", "install", "--vendor", "claude-code", "--key", KEY], home)
            self.assertEqual(code, 0, out)
            data = json.loads(settings.read_text())
            self.assertNotIn("SubagentStart", data["hooks"])
            self.assertEqual(sorted(data["hooks"]), ["PostToolUse", "PreToolUse", "SubagentStop"])
            self.assertEqual(data["hooks"]["PreToolUse"], self.ORIGINAL["hooks"]["PreToolUse"])
            run(["coding", "uninstall", "--vendor", "claude-code"], home)
            self.assertEqual(json.loads(settings.read_text()), self.ORIGINAL)

    def test_no_subagents_removes_hooks_and_status_says_off(self):
        with TemporaryDirectory() as home:
            settings = self._settings(home)
            run(["coding", "install", "--vendor", "claude-code", "--key", KEY], home)
            code, out = run(["coding", "install", "--vendor", "claude-code", "--no-subagents"], home)
            self.assertEqual(code, 0, out)
            self.assertIn("removed 2 agent hooks", out)
            data = json.loads(settings.read_text())
            self.assertEqual(data["hooks"], self.ORIGINAL["hooks"])
            code, out = run(["coding", "status"], home)
            self.assertIn("agent hooks: off", out)
            code, out = run(["coding", "uninstall", "--vendor", "claude-code"], home)
            self.assertEqual(json.loads(settings.read_text()), self.ORIGINAL)

    def test_fresh_settings_without_hooks_object(self):
        with TemporaryDirectory() as home:
            code, out = run(["coding", "install", "--vendor", "claude-code", "--key", KEY, "--dry-run"], home)
            self.assertEqual(code, 0, out)
            self.assertFalse((Path(home) / ".claude" / "settings.json").exists())
            run(["coding", "install", "--vendor", "claude-code", "--key", KEY], home)
            data = json.loads((Path(home) / ".claude" / "settings.json").read_text())
            self.assertEqual(sorted(data["hooks"]), ["PostToolUse", "SubagentStop"])
            run(["coding", "uninstall", "--vendor", "claude-code"], home)
            self.assertNotIn("hooks", json.loads((Path(home) / ".claude" / "settings.json").read_text()))


class TestHookRecords(unittest.TestCase):
    PAYLOAD = {
        "session_id": "sess-1",
        "transcript_path": "/tmp/t.jsonl",
        "cwd": "/work",
        "permission_mode": "default",
        "hook_event_name": "SubagentStop",
        "agent_id": "agent-42",
        "agent_type": "Explore",
        "last_assistant_message": "done, found 3 files",
    }

    def _attrs(self, record):
        rec = record["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
        return {a["key"]: a["value"] for a in rec["attributes"]}

    def test_claude_subagent_events_map_to_receiver_names(self):
        record = record_of("claude-code", "SubagentStop", self.PAYLOAD, False)
        attrs = self._attrs(record)
        self.assertEqual(attrs["event.name"], {"stringValue": "claude_code.subagent_stop"})
        self.assertEqual(attrs["session.id"], {"stringValue": "sess-1"})
        self.assertEqual(attrs["agent.id"], {"stringValue": "agent-42"})
        self.assertEqual(attrs["agent.type"], {"stringValue": "Explore"})
        self.assertNotIn("agent.last_message", attrs)
        self.assertNotIn("transcript_path", attrs)
        resource = {a["key"]: a["value"] for a in record["resourceLogs"][0]["resource"]["attributes"]}
        self.assertEqual(resource["service.name"], {"stringValue": "claude-code"})
        start = self._attrs(record_of("claude-code", "SubagentStart", self.PAYLOAD, False))
        self.assertEqual(start["event.name"], {"stringValue": "claude_code.subagent_start"})

    def test_full_capture_keeps_last_message_and_unknown_event_is_dropped(self):
        attrs = self._attrs(record_of("claude-code", "SubagentStop", self.PAYLOAD, True))
        self.assertEqual(attrs["agent.last_message"], {"stringValue": "done, found 3 files"})
        self.assertEqual(build_records("claude-code", "PreToolUse", self.PAYLOAD, True), [])
        cursor = self._attrs(record_of("cursor", "stop", {"conversation_id": "c1", "generation_id": "g1"}, False))
        self.assertEqual(cursor["event.name"], {"stringValue": "evalkit.enrichment"})

    def test_claude_hook_process_posts_marker_and_stays_silent(self):
        srv, received = _capture_server()
        with TemporaryDirectory() as home:
            try:
                endpoint = f"http://127.0.0.1:{srv.server_address[1]}/v1/otlp"
                run(["coding", "install", "--vendor", "claude-code", "--key", KEY, "--endpoint", endpoint], home)
                proc = subprocess.run(
                    [sys.executable, "-m", "evalkit_cli.main", "coding", "hook", "--vendor", "claude-code", "--event", "SubagentStart"],
                    input=json.dumps(self.PAYLOAD), capture_output=True, text=True, env={**os.environ, "HOME": home}, timeout=20,
                )
            finally:
                srv.shutdown()
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(proc.stdout, "")
            self.assertEqual(received[0]["path"], "/v1/otlp/v1/logs")
            self.assertEqual(received[0]["auth"], f"Bearer {KEY}")
            self.assertEqual(self._attrs(received[0]["body"])["event.name"], {"stringValue": "claude_code.subagent_start"})


class TestInstallerStateFixes(unittest.TestCase):
    def test_uninstall_restores_a_key_that_already_matched_our_value(self):
        with TemporaryDirectory() as home:
            settings = Path(home) / ".claude" / "settings.json"
            settings.parent.mkdir()
            original = {"env": {"CLAUDE_CODE_ENABLE_TELEMETRY": "1", "OTEL_METRICS_EXPORTER": "otlp", "FOO": "bar"}}
            settings.write_text(json.dumps(original))
            run(["coding", "install", "--vendor", "claude-code", "--key", KEY], home)
            code, out = run(["coding", "uninstall", "--vendor", "claude-code"], home)
            self.assertEqual(code, 0, out)
            self.assertEqual(json.loads(settings.read_text())["env"], original["env"])

    def test_installing_a_second_scope_still_remembers_the_first(self):
        with TemporaryDirectory() as home, TemporaryDirectory() as project:
            run(["coding", "install", "--vendor", "claude-code", "--key", KEY], home)
            run(["coding", "install", "--vendor", "claude-code", "--scope", "project"], home, cwd=project)
            user_settings = Path(home) / ".claude" / "settings.json"
            project_settings = Path(project) / ".claude" / "settings.json"
            state = json.loads((Path(home) / ".config" / "evalkit" / "state.json").read_text())
            self.assertEqual(sorted(state["vendors"]["claude-code"]["files"]), sorted([str(user_settings), str(project_settings)]))
            run(["coding", "uninstall", "--vendor", "claude-code"], home)
            for path in (user_settings, project_settings):
                self.assertNotIn(KEY, path.read_text(), path)

    def test_uninstall_never_touches_foreign_hook_shapes(self):
        with TemporaryDirectory() as home:
            settings = Path(home) / ".claude" / "settings.json"
            settings.parent.mkdir()
            original = {"hooks": {"Notification": {"legacy": "shape"}, "Stop": [], "PreToolUse": [{"hooks": [{"type": "command", "command": "echo pre"}]}]}}
            settings.write_text(json.dumps(original))
            run(["coding", "install", "--vendor", "claude-code", "--key", KEY], home)
            run(["coding", "uninstall", "--vendor", "claude-code"], home)
            self.assertEqual(json.loads(settings.read_text()), original)

    def test_install_refuses_a_non_list_hook_entry_instead_of_replacing_it(self):
        with TemporaryDirectory() as home:
            settings = Path(home) / ".claude" / "settings.json"
            settings.parent.mkdir()
            original = json.dumps({"hooks": {"PostToolUse": {"legacy": "shape"}}})
            settings.write_text(original)
            code, _ = run(["coding", "install", "--vendor", "claude-code", "--key", KEY], home)
            self.assertEqual(code, 1)
            self.assertEqual(settings.read_text(), original)


class TestToolOutputHookRecord(unittest.TestCase):
    PAYLOAD = {
        "session_id": "sess-1",
        "transcript_path": "/tmp/t.jsonl",
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "tool_use_id": "toolu_7",
        "duration_ms": 412,
        "agent_id": "agent-42",
        "agent_type": "Explore",
        "tool_input": {"command": "ls"},
        "tool_response": "a.go\nb.go",
    }

    def _attrs(self, record):
        rec = record["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
        return {a["key"]: a["value"] for a in rec["attributes"]}

    def test_post_tool_use_maps_to_tool_output(self):
        attrs = self._attrs(record_of("claude-code", "PostToolUse", self.PAYLOAD, False))
        self.assertEqual(attrs["event.name"], {"stringValue": "claude_code.tool_output"})
        self.assertEqual(attrs["session.id"], {"stringValue": "sess-1"})
        self.assertEqual(attrs["tool_use_id"], {"stringValue": "toolu_7"})
        self.assertEqual(attrs["tool_name"], {"stringValue": "Bash"})
        self.assertEqual(attrs["duration_ms"], {"intValue": "412"})
        self.assertEqual(attrs["agent.id"], {"stringValue": "agent-42"})
        self.assertEqual(attrs["agent.type"], {"stringValue": "Explore"})
        self.assertNotIn("tool_output", attrs)
        self.assertNotIn("transcript_path", attrs)
        self.assertNotIn("tool_input", attrs)

    def test_full_capture_sends_the_tool_response(self):
        attrs = self._attrs(record_of("claude-code", "PostToolUse", self.PAYLOAD, True))
        self.assertEqual(attrs["tool_output"], {"stringValue": "a.go\nb.go"})
        structured = self._attrs(record_of("claude-code", "PostToolUse", {**self.PAYLOAD, "tool_response": {"ok": True}}, True))
        self.assertEqual(structured["tool_output"], {"stringValue": '{"ok":true}'})


class TestCursorSpans(unittest.TestCase):
    BASE = {"conversation_id": "conv-1", "generation_id": "gen-1", "model": "claude-sonnet", "user_email": "dev@example.com"}
    TRACE = "cbfa0a89b7de3bd0f6a1b2c1f2fc7f0f"

    def _span(self, event, extra, capture_full=True):
        records = build_records("cursor", event, {**self.BASE, **extra}, capture_full, NOW_NS)
        self.assertEqual(len(records), 1)
        signal, payload = records[0]
        self.assertEqual(signal, "traces")
        return payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]

    def _log(self, event, extra, capture_full=True):
        records = build_records("cursor", event, {**self.BASE, **extra}, capture_full, NOW_NS)
        self.assertEqual(len(records), 1)
        signal, payload = records[0]
        self.assertEqual(signal, "logs")
        return payload["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]

    def test_ids_are_derived_exactly_as_documented(self):
        from hashlib import sha256

        span = self._span("beforeSubmitPrompt", {"prompt": "fix the flaky test"})
        self.assertEqual(span["traceId"], sha256(b"conv-1gen-1").hexdigest()[:32])
        self.assertEqual(span["spanId"], sha256(b"gen-1").hexdigest()[:16])
        self.assertNotIn("parentSpanId", span)
        tool = self._span("postToolUse", {"tool_name": "Bash", "tool_use_id": "toolu_9", "duration": 250})
        self.assertEqual(tool["spanId"], sha256(b"toolu_9").hexdigest()[:16])
        self.assertEqual(tool["parentSpanId"], span["spanId"])
        self.assertEqual(int(tool["endTimeUnixNano"]) - int(tool["startTimeUnixNano"]), 250 * 1_000_000)
        self.assertEqual(tool["endTimeUnixNano"], str(NOW_NS))

    def test_turn_root_carries_the_model_table_and_hides_content_unless_full(self):
        span = self._span("beforeSubmitPrompt", {"prompt": "fix the flaky test"})
        attrs = _attrs(span["attributes"])
        self.assertEqual(span["name"], "fix the flaky test")
        self.assertEqual(attrs["evalkit.span_type"], "agent")
        self.assertEqual(attrs["gen_ai.operation.name"], "invoke_agent")
        self.assertEqual(attrs["agent.vendor"], "cursor")
        self.assertEqual(attrs["gen_ai.conversation.id"], "conv-1")
        self.assertEqual(attrs["agent.turn.id"], "gen-1")
        self.assertEqual(attrs["agent.role"], "main")
        self.assertEqual(attrs["gen_ai.agent.name"], "Cursor")
        self.assertEqual(attrs["evalkit.attribution"], "hooks")
        self.assertEqual(attrs["gen_ai.request.model"], "claude-sonnet")
        self.assertEqual(attrs["prompt"], "fix the flaky test")
        self.assertEqual(attrs["user.id"], "dev@example.com")
        minimal = _attrs(self._span("beforeSubmitPrompt", {"prompt": "fix the flaky test"}, capture_full=False)["attributes"])
        self.assertNotIn("prompt", minimal)
        self.assertNotIn("user.id", minimal)
        self.assertEqual(self._span("beforeSubmitPrompt", {"prompt": "x"}, capture_full=False)["name"], "turn")

    def test_response_and_stop_complete_the_turn_through_an_enrichment(self):
        record = self._log("afterAgentResponse", {"text": "all green"})
        attrs = _attrs(record["attributes"])
        self.assertEqual(record["traceId"], self._span("beforeSubmitPrompt", {"prompt": "p"})["traceId"])
        self.assertEqual(record["spanId"], self._span("beforeSubmitPrompt", {"prompt": "p"})["spanId"])
        self.assertEqual(attrs["event.name"], "evalkit.enrichment")
        self.assertTrue(attrs["evalkit.enrichment"])
        self.assertEqual(attrs["evalkit.end_time_unix_nano"], str(NOW_NS))
        self.assertEqual(attrs["completion"], "all green")
        self.assertNotIn("completion", _attrs(self._log("afterAgentResponse", {"text": "all green"}, capture_full=False)["attributes"]))
        stop = _attrs(self._log("stop", {})["attributes"])
        self.assertEqual(stop["evalkit.end_time_unix_nano"], str(NOW_NS))

    def test_subagent_spans_hang_off_the_agent_tool_call(self):
        from hashlib import sha256

        span = self._span("subagentStart", {"subagent_id": "sub-7", "subagent_type": "Explore", "task": "find the callers", "tool_call_id": "toolu_9", "subagent_model": "haiku", "is_parallel_worker": True})
        attrs = _attrs(span["attributes"])
        self.assertEqual(span["spanId"], sha256(b"sub-7").hexdigest()[:16])
        self.assertEqual(span["parentSpanId"], sha256(b"toolu_9").hexdigest()[:16])
        self.assertEqual(span["name"], "subagent:Explore")
        self.assertEqual(attrs["evalkit.span_type"], "agent")
        self.assertEqual(attrs["agent.role"], "subagent")
        self.assertEqual(attrs["agent.subagent.id"], "sub-7")
        self.assertEqual(attrs["agent.subagent.type"], "Explore")
        self.assertEqual(attrs["gen_ai.agent.name"], "Explore")
        self.assertEqual(attrs["gen_ai.request.model"], "haiku")
        self.assertEqual(attrs["prompt"], "find the callers")
        stop = self._log("subagentStop", {"subagent_id": "sub-7", "subagent_type": "Explore", "summary": "3 callers", "tool_call_count": 4, "message_count": 9, "status": "completed"})
        stop_attrs = _attrs(stop["attributes"])
        self.assertEqual(stop["spanId"], span["spanId"])
        self.assertEqual(stop_attrs["agent.subagent.total_tool_uses"], "4")
        self.assertEqual(stop_attrs["agent.subagent.message_count"], "9")
        self.assertEqual(stop_attrs["completion"], "3 callers")

    def test_a_subagent_tool_call_parents_under_the_subagent(self):
        from hashlib import sha256

        span = self._span("postToolUse", {"tool_name": "Read", "tool_use_id": "toolu_x", "duration": 5, "subagent_id": "sub-7", "subagent_type": "Explore"})
        self.assertEqual(span["parentSpanId"], sha256(b"sub-7").hexdigest()[:16])
        attrs = _attrs(span["attributes"])
        self.assertEqual(attrs["agent.role"], "subagent")
        self.assertEqual(attrs["gen_ai.agent.name"], "Explore")

    def test_failures_are_errors_and_session_events_are_noise(self):
        span = self._span("postToolUseFailure", {"tool_name": "Bash", "tool_use_id": "toolu_b", "duration": 3})
        self.assertEqual(span["status"], {"code": 2})
        self.assertFalse(_attrs(span["attributes"])["success"])
        ok = self._span("afterShellExecution", {"command": "ls", "output": "a.go", "duration": 7, "tool_use_id": "toolu_c"})
        self.assertEqual(ok["status"], {"code": 1})
        self.assertEqual(ok["name"], "Shell")
        self.assertEqual(_attrs(ok["attributes"])["tool.arguments"], "ls")
        session = _attrs(self._log("sessionEnd", {"duration_ms": 900, "final_status": "completed"})["attributes"])
        self.assertEqual(session["event.name"], "cursor.sessionEnd")
        self.assertTrue(session["evalkit.noise"])
        self.assertEqual(session["duration_ms"], "900")

    def test_events_without_a_turn_or_a_mapping_send_nothing(self):
        self.assertEqual(build_records("cursor", "postToolUse", {"conversation_id": "c"}, True, NOW_NS), [])
        self.assertEqual(build_records("cursor", "beforeShellExecution", {**self.BASE}, True, NOW_NS), [])
        self.assertEqual(build_records("nope", "postToolUse", {**self.BASE}, True, NOW_NS), [])


class TestWindsurfSpans(unittest.TestCase):
    BASE = {"trajectory_id": "traj-1", "execution_id": "exec-1", "model_name": "swe-1"}

    def _records(self, event, extra, capture_full=True):
        return build_records("windsurf", event, {**self.BASE, **extra}, capture_full, NOW_NS)

    def test_turn_root_and_completion_use_trajectory_and_execution_ids(self):
        from hashlib import sha256

        signal, payload = self._records("pre_user_prompt", {"prompt": "add a test"})[0]
        span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        self.assertEqual(signal, "traces")
        self.assertEqual(span["traceId"], sha256(b"traj-1exec-1").hexdigest()[:32])
        self.assertEqual(span["spanId"], sha256(b"exec-1").hexdigest()[:16])
        attrs = _attrs(span["attributes"])
        self.assertEqual(attrs["gen_ai.conversation.id"], "traj-1")
        self.assertEqual(attrs["agent.turn.id"], "exec-1")
        self.assertEqual(attrs["gen_ai.agent.name"], "Windsurf Cascade")
        self.assertEqual(attrs["gen_ai.request.model"], "swe-1")
        signal, payload = self._records("post_cascade_response", {"response": "done"})[0]
        record = payload["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
        self.assertEqual(signal, "logs")
        self.assertEqual(record["spanId"], span["spanId"])
        self.assertEqual(_attrs(record["attributes"])["completion"], "done")

    def test_tool_fields_are_read_out_of_tool_info(self):
        signal, payload = self._records("post_run_command", {"tool_info": {"command": "go test", "output": "ok", "duration_ms": 40, "tool_call_id": "call-9"}})[0]
        span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        from hashlib import sha256

        self.assertEqual(span["spanId"], sha256(b"call-9").hexdigest()[:16])
        attrs = _attrs(span["attributes"])
        self.assertEqual(attrs["gen_ai.tool.name"], "Command")
        self.assertEqual(attrs["tool.arguments"], "go test")
        self.assertEqual(attrs["tool.result"], "ok")
        self.assertEqual(int(span["endTimeUnixNano"]) - int(span["startTimeUnixNano"]), 40 * 1_000_000)

    def test_a_tool_call_without_an_id_still_gets_a_stable_span_id(self):
        first = self._records("post_read_code", {"tool_info": {"file_path": "a.go"}})[0][1]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        again = self._records("post_read_code", {"tool_info": {"file_path": "a.go"}})[0][1]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        other = build_records("windsurf", "post_write_code", {**self.BASE, "tool_info": {"file_path": "a.go"}}, True, NOW_NS)[0][1]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        self.assertEqual(first["spanId"], again["spanId"])
        self.assertNotEqual(first["spanId"], other["spanId"])

    def test_only_the_events_we_map_are_installed(self):
        self.assertEqual(WINDSURF_EVENTS, ["pre_user_prompt", "post_cascade_response", "post_run_command", "post_write_code", "post_read_code", "post_mcp_tool_use"])
        self.assertEqual(
            CURSOR_EVENTS,
            [
                "sessionStart",
                "sessionEnd",
                "beforeSubmitPrompt",
                "afterAgentResponse",
                "stop",
                "postToolUse",
                "postToolUseFailure",
                "afterShellExecution",
                "afterMCPExecution",
                "afterFileEdit",
                "subagentStart",
                "subagentStop",
            ],
        )


class TestOpenCode(unittest.TestCase):
    def test_install_writes_the_plugin_and_registers_it(self):
        from evalkit_cli.opencode_plugin import PLUGIN_SOURCE, PLUGIN_SOURCE_SHA256
        from hashlib import sha256

        with TemporaryDirectory() as home:
            config = Path(home) / ".config" / "opencode" / "opencode.json"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({"theme": "system", "plugin": ["other-plugin"]}))
            code, out = run(["coding", "install", "--vendor", "opencode", "--key", KEY], home)
            self.assertEqual(code, 0, out)
            data = json.loads(config.read_text())
            self.assertEqual(data["theme"], "system")
            self.assertEqual(data["plugin"], ["other-plugin", "./plugins/evalkit.ts"])
            plugin = config.parent / "plugins" / "evalkit.ts"
            self.assertEqual(plugin.read_text(), PLUGIN_SOURCE)
            self.assertEqual(sha256(plugin.read_text().encode()).hexdigest(), PLUGIN_SOURCE_SHA256)
            self.assertIn("evalkit.enrichment", plugin.read_text())
            self.assertNotIn(KEY, plugin.read_text())
            before = config.read_text()
            run(["coding", "install", "--vendor", "opencode"], home)
            self.assertEqual(config.read_text(), before)
            code, out = run(["coding", "status"], home)
            self.assertIn("opencode: installed (user)", out)
            code, out = run(["coding", "uninstall", "--vendor", "opencode"], home)
            self.assertEqual(code, 0, out)
            self.assertEqual(json.loads(config.read_text()), {"theme": "system", "plugin": ["other-plugin"]})
            self.assertFalse(plugin.exists())
            self.assertFalse(plugin.parent.exists())

    def test_project_scope_and_a_foreign_plugin_value(self):
        with TemporaryDirectory() as home, TemporaryDirectory() as project:
            code, out = run(["coding", "install", "--vendor", "opencode", "--key", KEY, "--scope", "project"], home, cwd=project)
            self.assertEqual(code, 0, out)
            data = json.loads((Path(project) / "opencode.json").read_text())
            self.assertEqual(data["plugin"], ["./.opencode/plugins/evalkit.ts"])
            self.assertTrue((Path(project) / ".opencode" / "plugins" / "evalkit.ts").exists())
            run(["coding", "uninstall", "--vendor", "opencode"], home)
            self.assertEqual(json.loads((Path(project) / "opencode.json").read_text()), {})
        with TemporaryDirectory() as home:
            config = Path(home) / ".config" / "opencode" / "opencode.json"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({"plugin": "single-plugin"}))
            code, _ = run(["coding", "install", "--vendor", "opencode", "--key", KEY], home)
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(config.read_text()), {"plugin": "single-plugin"})


class TestHookVendorMigration(unittest.TestCase):
    def test_reinstall_drops_owned_hooks_of_events_we_no_longer_use(self):
        with TemporaryDirectory() as home:
            hooks = Path(home) / ".cursor" / "hooks.json"
            hooks.parent.mkdir()
            stale = {"command": "evalkit coding hook --vendor cursor --event preToolUse", "timeout": 5000}
            hooks.write_text(json.dumps({"version": 1, "hooks": {"preToolUse": [stale], "stop": [{"command": "notify-send done"}]}}))
            code, out = run(["coding", "install", "--vendor", "cursor", "--key", KEY], home)
            self.assertEqual(code, 0, out)
            data = json.loads(hooks.read_text())
            self.assertNotIn("preToolUse", data["hooks"])
            self.assertEqual(sorted(data["hooks"]), sorted(CURSOR_EVENTS))
            run(["coding", "uninstall", "--vendor", "cursor"], home)
            self.assertEqual(json.loads(hooks.read_text()), {"version": 1, "hooks": {"stop": [{"command": "notify-send done"}]}})

    def test_uninstall_removes_an_empty_hooks_table(self):
        with TemporaryDirectory() as home:
            run(["coding", "install", "--vendor", "windsurf", "--key", KEY], home)
            path = Path(home) / ".codeium" / "windsurf" / "hooks.json"
            self.assertEqual(len(json.loads(path.read_text())["hooks"]), len(WINDSURF_EVENTS))
            run(["coding", "uninstall", "--vendor", "windsurf"], home)
            self.assertEqual(json.loads(path.read_text()), {})
