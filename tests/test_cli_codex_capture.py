import io
import json
import os
import subprocess
import sys
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from evalkit_cli import codex_capture, codex_rollout
from evalkit_cli.config import ConfigStore, SharedConfig
from evalkit_cli.main import main

KEY = "tk_test_codex_capture"
THREAD = "01a0aacb-5769-7cc3-b65c-aed8532d8a1a"
TURN_ONE = "01a0aacb-615e-7a43-9b3a-ad376ab59d18"
TURN_TWO = "01a0aacc-111d-7822-bfc8-8f893141d389"
ROLLOUT = f"rollout-2026-09-16T20-47-28-{THREAD}.jsonl"
TRUNCATED = '{"timestamp":"2026-09-16T15:19:00.000Z","ordinal":11,"type":"response_item","payload":{"type":"mess'
TRACE_ONE = "caafe5aac2b449dc38be92c4393676a0"


def _line(ordinal, at, kind, payload):
    return json.dumps({"timestamp": at, "ordinal": ordinal, "type": kind, "payload": payload})


def fixture_lines():
    return [
        _line(1, "2026-09-16T15:17:30.618Z", "event_msg", {"type": "task_started", "turn_id": TURN_ONE}),
        _line(2, "2026-09-16T15:17:30.681Z", "response_item", {"type": "message", "id": "msg_user", "role": "user", "content": [{"type": "input_text", "text": "hey"}]}),
        _line(
            3,
            "2026-09-16T15:17:42.012Z",
            "response_item",
            {
                "type": "message",
                "id": "msg_first",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "first part"}, {"type": "output_text", "text": "second part"}],
                "phase": "commentary",
                "internal_chat_message_metadata_passthrough": {"turn_id": TURN_ONE},
            },
        ),
        _line(
            4,
            "2026-09-16T15:17:42.012Z",
            "response_item",
            {
                "type": "reasoning",
                "id": "rs_first",
                "summary": [{"type": "summary_text", "text": "weighing options"}],
                "encrypted_content": "rsn_secret_blob",
                "internal_chat_message_metadata_passthrough": {"turn_id": TURN_ONE},
            },
        ),
        _line(
            5,
            "2026-09-16T15:18:19.960Z",
            "response_item",
            {
                "type": "function_call",
                "id": "fc_1",
                "name": "exec_command",
                "arguments": '{"cmd":"pwd"}',
                "call_id": "call_1",
                "internal_chat_message_metadata_passthrough": {"turn_id": TURN_ONE},
            },
        ),
        _line(
            6,
            "2026-09-16T15:18:20.228Z",
            "response_item",
            {"type": "function_call_output", "id": "fco_1", "call_id": "call_1", "output": "/work", "internal_chat_message_metadata_passthrough": {"turn_id": TURN_ONE}},
        ),
        '{"timestamp":"2026-09-16T15:18:21.000Z","ordinal":7,"type":"response_item","payload":{"type":',
        _line(
            8,
            "2026-09-16T15:18:22.293Z",
            "token_usage_record",
            {"thread_id": THREAD, "turn_id": TURN_ONE, "response_id": "resp_1", "usage": {"input_tokens": 13909, "output_tokens": 12}},
        ),
        json.dumps(
            {
                "timestamp": "2026-09-16T15:18:47.090Z",
                "ordinal": 9,
                "type": "message",
                "id": "msg_final",
                "role": "assistant",
                "phase": "final_answer",
                "turn_id": TURN_TWO,
                "content": [{"type": "output_text", "text": "all done"}],
            }
        ),
    ]


def traced_lines():
    lines = fixture_lines()
    lines[0] = _line(1, "2026-09-16T15:17:30.618Z", "event_msg", {"type": "task_started", "turn_id": TURN_ONE, "trace_id": TRACE_ONE})
    return lines


def write_fixture(home, lines=None, truncated=True):
    directory = Path(home) / ".codex" / "sessions" / "2026" / "09" / "16"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ROLLOUT
    body = "\n".join(fixture_lines() if lines is None else lines) + "\n"
    path.write_text(body + (TRUNCATED if truncated else ""))
    index = Path(home) / ".codex" / "session_index.jsonl"
    index.write_text(json.dumps({"id": THREAD, "thread_name": "Respond to greeting", "updated_at": "2026-09-16T15:19:00.000Z"}) + "\n")
    return path


@contextmanager
def codex_home(home):
    previous = os.environ.get("CODEX_HOME")
    os.environ["CODEX_HOME"] = str(Path(home) / ".codex")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CODEX_HOME", None)
        else:
            os.environ["CODEX_HOME"] = previous


def run(argv, home):
    out = io.StringIO()
    with codex_home(home):
        code = main(argv, home=Path(home), out=out)
    return code, out.getvalue()


def receiver(status=200):
    posts = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            posts.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": json.loads(self.rfile.read(length) or b"{}")})
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, posts


def records_of(posts):
    out = []
    for post in posts:
        for resource in post["body"].get("resourceLogs", []):
            attrs = {a["key"]: list(a["value"].values())[0] for a in resource["resource"]["attributes"]}
            for scope in resource.get("scopeLogs", []):
                for record in scope.get("logRecords", []):
                    out.append(
                        {
                            "resource": attrs,
                            "at": record["timeUnixNano"],
                            "trace": record.get("traceId"),
                            "attrs": {a["key"]: list(a["value"].values())[0] for a in record["attributes"]},
                        }
                    )
    return out


def install(home, endpoint, capture="full"):
    return run(["coding", "install", "--vendor", "codex", "--key", KEY, "--endpoint", endpoint, "--capture", capture], home)


class TestCodexRollout(unittest.TestCase):
    def test_normalises_known_kinds_and_skips_the_rest(self):
        with TemporaryDirectory() as home:
            path = write_fixture(home)
            chunk = codex_rollout.read(path)
            kinds = [(item.kind, item.item_id) for item in chunk.items]
            self.assertEqual(
                kinds,
                [
                    ("assistant_message", "msg_first"),
                    ("reasoning", "rs_first"),
                    ("tool_call", "call_1"),
                    ("tool_output", "call_1"),
                    ("usage", "resp_1"),
                    ("assistant_message", "msg_final"),
                ],
            )
            first, reasoning, call, output, usage, final = chunk.items
            self.assertEqual(first.text, "first part\nsecond part")
            self.assertEqual(first.turn_id, TURN_ONE)
            self.assertEqual(first.phase, "commentary")
            self.assertEqual(first.at, "2026-09-16T15:17:42.012Z")
            self.assertEqual(first.ordinal, 3)
            self.assertEqual(reasoning.text, "weighing options")
            self.assertNotIn("rsn_secret_blob", reasoning.text)
            self.assertEqual((call.tool_name, call.text), ("exec_command", '{"cmd":"pwd"}'))
            self.assertEqual(output.text, "/work")
            self.assertEqual((usage.tokens_in, usage.tokens_out), (13909, 12))
            self.assertEqual((final.turn_id, final.phase, final.text), (TURN_TWO, "final_answer", "all done"))

    def test_a_truncated_last_line_is_left_for_the_next_read(self):
        with TemporaryDirectory() as home:
            path = write_fixture(home)
            chunk = codex_rollout.read(path)
            self.assertEqual(chunk.offset, len(path.read_text()) - len(TRUNCATED))
            self.assertEqual(codex_rollout.read(path, chunk.offset).items, [])
            rest = 'age","id":"msg_late","role":"assistant","content":[{"type":"output_text","text":"late"}],"turn_id":"' + TURN_TWO + '"}}\n'
            with open(path, "a") as handle:
                handle.write(rest)
            resumed = codex_rollout.read(path, chunk.offset)
            self.assertEqual([item.item_id for item in resumed.items], ["msg_late"])
            self.assertEqual(resumed.offset, path.stat().st_size)

    def test_malformed_json_and_unknown_shapes_never_raise(self):
        with TemporaryDirectory() as home:
            path = write_fixture(home, lines=["", "not json at all", "[1,2,3]", json.dumps({"type": "world_state", "payload": {"state": {}}})], truncated=False)
            self.assertEqual(codex_rollout.read(path).items, [])
            self.assertIsNone(codex_rollout.normalise("nope"))
            self.assertEqual(codex_rollout.read(Path(home) / "missing.jsonl").offset, 0)

    def test_thread_id_comes_from_the_file_name(self):
        self.assertEqual(codex_rollout.thread_id(Path("/x/" + ROLLOUT)), THREAD)


class TestCodexRecords(unittest.TestCase):
    def _built(self, capture_full):
        with TemporaryDirectory() as home:
            chunk = codex_rollout.read(write_fixture(home))
        return codex_capture.build(chunk.items, THREAD, capture_full)

    def test_full_capture_emits_the_receiver_contract(self):
        built = self._built(True)
        self.assertEqual((built.messages, built.reasoning), (2, 1))
        names = [next(a["value"]["stringValue"] for a in record["attributes"] if a["key"] == "event.name") for record in built.records]
        self.assertEqual(names, ["codex.assistant_message", "codex.assistant_reasoning", "codex.assistant_message"])
        attrs = {a["key"]: a["value"]["stringValue"] for a in built.records[0]["attributes"]}
        self.assertEqual(attrs["conversation.id"], THREAD)
        self.assertEqual(attrs["turn.id"], TURN_ONE)
        self.assertEqual(attrs["item.id"], "msg_first")
        self.assertEqual(attrs["completion"], "first part\nsecond part")
        reasoning = {a["key"]: a["value"]["stringValue"] for a in built.records[1]["attributes"]}
        self.assertEqual(reasoning["completion"], "weighing options")

    def test_every_record_gets_its_own_nanosecond_stamp(self):
        built = self._built(True)
        stamps = [record["timeUnixNano"] for record in built.records]
        self.assertEqual(len(set(stamps)), len(stamps))
        self.assertTrue(stamps[0].startswith("1789571862012"))
        self.assertEqual(int(stamps[1]) - int(stamps[0]), 1)

    def test_minimal_capture_keeps_the_metadata_and_drops_the_text(self):
        built = self._built(False)
        self.assertEqual((built.messages, built.reasoning), (2, 0))
        for record in built.records:
            keys = {a["key"] for a in record["attributes"]}
            self.assertNotIn("completion", keys)
            self.assertIn("item.id", keys)

    def test_task_started_pins_the_turn_trace_and_a_missing_one_is_tolerated(self):
        with TemporaryDirectory() as home:
            chunk = codex_rollout.read(write_fixture(home, lines=traced_lines()))
        self.assertEqual(chunk.traces, {TURN_ONE: TRACE_ONE})
        self.assertEqual([item.trace_id for item in chunk.items if item.turn_id == TURN_ONE], [TRACE_ONE] * 5)
        self.assertEqual([item.trace_id for item in chunk.items if item.turn_id == TURN_TWO], [""])
        built = codex_capture.build(chunk.items, THREAD, True)
        self.assertEqual([record.get("traceId") for record in built.records], [TRACE_ONE, TRACE_ONE, None])
        for record in built.records:
            self.assertNotIn("spanId", record)
            attrs = {a["key"] for a in record["attributes"]}
            self.assertTrue({"conversation.id", "turn.id", "item.id"} <= attrs)

    def test_a_turn_without_task_started_emits_with_no_trace_id(self):
        with TemporaryDirectory() as home:
            chunk = codex_rollout.read(write_fixture(home))
        self.assertEqual(chunk.traces, {})
        built = codex_capture.build(chunk.items, THREAD, True)
        self.assertEqual(len(built.records), 3)
        for record in built.records:
            self.assertNotIn("traceId", record)
            self.assertNotIn("spanId", record)
            attrs = {a["key"]: a["value"]["stringValue"] for a in record["attributes"]}
            self.assertEqual(attrs["conversation.id"], THREAD)
            self.assertTrue(attrs["turn.id"])
            self.assertTrue(attrs["item.id"])

    def test_a_resumed_read_finds_the_trace_id_earlier_in_the_file(self):
        with TemporaryDirectory() as home:
            lines = traced_lines()
            path = write_fixture(home, lines=lines)
            offset = len(("\n".join(lines[:2]) + "\n").encode())
            chunk = codex_rollout.read(path, offset)
            self.assertEqual(chunk.traces, {TURN_ONE: TRACE_ONE})
            first = next(item for item in chunk.items if item.turn_id == TURN_ONE)
            self.assertEqual(first.trace_id, TRACE_ONE)
            self.assertEqual(codex_capture.build(chunk.items, THREAD, True).records[0]["traceId"], TRACE_ONE)

    def test_an_item_without_an_id_falls_back_to_thread_turn_ordinal(self):
        item = codex_rollout.Item(codex_rollout.ASSISTANT_MESSAGE, TURN_ONE, "", "hi", ordinal=7)
        self.assertEqual(codex_capture.item_id(item, THREAD), f"{THREAD}:{TURN_ONE}:7")


class TestCodexBackfill(unittest.TestCase):
    def test_offsets_keep_a_second_run_from_re_emitting(self):
        server, posts = receiver()
        with TemporaryDirectory() as home:
            try:
                write_fixture(home)
                endpoint = f"http://127.0.0.1:{server.server_address[1]}"
                install(home, endpoint)
                code, out = run(["coding", "backfill", "--vendor", "codex"], home)
                self.assertEqual(code, 0, out)
                self.assertIn(THREAD, out)
                self.assertIn("Respond to greeting", out)
                self.assertIn("2 messages", out)
                self.assertIn("bytes", out)
                emitted = records_of(posts)
                self.assertEqual(len(emitted), 3)
                self.assertEqual(emitted[0]["resource"]["service.name"], "codex_cli_rs")
                self.assertEqual(emitted[0]["resource"]["agent.vendor"], "codex")
                self.assertEqual(posts[0]["path"], "/v1/logs")
                self.assertEqual(posts[0]["auth"], f"Bearer {KEY}")
                code, again = run(["coding", "backfill", "--vendor", "codex"], home)
                self.assertEqual(code, 0, again)
                self.assertIn("0 messages", again)
                self.assertEqual(len(records_of(posts)), 3)
                code, forced = run(["coding", "backfill", "--vendor", "codex", "--force"], home)
                self.assertEqual(code, 0, forced)
                self.assertIn("2 messages", forced)
                self.assertEqual(len(records_of(posts)), 6)
            finally:
                server.shutdown()

    def test_dry_run_prints_a_summary_and_emits_nothing(self):
        server, posts = receiver()
        with TemporaryDirectory() as home:
            try:
                write_fixture(home)
                install(home, f"http://127.0.0.1:{server.server_address[1]}")
                code, out = run(["coding", "backfill", "--vendor", "codex", "--dry-run"], home)
                self.assertEqual(code, 0, out)
                self.assertTrue(out.startswith("dry-run: "), out)
                self.assertIn("2 messages", out)
                self.assertEqual(posts, [])
                self.assertFalse((Path(home) / ".config" / "evalkit" / codex_capture.OFFSETS_FILE).exists())
                code, live = run(["coding", "backfill", "--vendor", "codex"], home)
                self.assertEqual(code, 0, live)
                self.assertEqual(len(records_of(posts)), 3)
            finally:
                server.shutdown()

    def test_minimal_capture_says_so_and_sends_no_text(self):
        server, posts = receiver()
        with TemporaryDirectory() as home:
            try:
                write_fixture(home)
                install(home, f"http://127.0.0.1:{server.server_address[1]}", capture="minimal")
                code, out = run(["coding", "backfill", "--vendor", "codex"], home)
                self.assertEqual(code, 0, out)
                self.assertIn("text off (capture minimal)", out)
                for record in records_of(posts):
                    self.assertNotIn("completion", record["attrs"])
            finally:
                server.shutdown()

    def test_a_rejected_post_keeps_the_offset_and_reports_failure(self):
        server, posts = receiver(status=500)
        with TemporaryDirectory() as home:
            try:
                write_fixture(home)
                install(home, f"http://127.0.0.1:{server.server_address[1]}")
                code, out = run(["coding", "backfill", "--vendor", "codex"], home)
                self.assertEqual(code, 1)
                self.assertIn("post failed, offset kept", out)
                self.assertFalse((Path(home) / ".config" / "evalkit" / codex_capture.OFFSETS_FILE).exists())
            finally:
                server.shutdown()

    def test_selection_flags_and_an_empty_codex_home(self):
        with TemporaryDirectory() as home:
            code, out = run(["coding", "backfill", "--vendor", "codex", "--dry-run"], home)
            self.assertEqual(code, 0)
            self.assertIn("no codex sessions found", out)
            write_fixture(home)
            code, out = run(["coding", "backfill", "--vendor", "codex", "--dry-run", "--session", "nope"], home)
            self.assertEqual((code, out), (0, "no codex session nope found under " + str(Path(home) / ".codex" / "sessions") + "\n"))
            code, out = run(["coding", "backfill", "--vendor", "codex", "--dry-run", "--session", THREAD], home)
            self.assertIn(THREAD, out)
            code, out = run(["coding", "backfill", "--vendor", "codex", "--dry-run", "--all", "--since", "2027-01-01T00:00:00Z"], home)
            self.assertIn("no codex sessions found", out)
            code, out = run(["coding", "backfill", "--vendor", "codex", "--dry-run", "--last", "5"], home)
            self.assertEqual(len(out.strip().splitlines()), 1)

    def test_backfill_without_a_key_asks_for_one(self):
        with TemporaryDirectory() as home:
            write_fixture(home)
            code, _ = run(["coding", "backfill", "--vendor", "codex"], home)
            self.assertEqual(code, 2)


class TestCodexHook(unittest.TestCase):
    def _hook(self, home, payload, event="Stop"):
        return subprocess.run(
            [sys.executable, "-m", "evalkit_cli.main", "coding", "hook", "--vendor", "codex", "--event", event],
            input=payload,
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": home, "CODEX_HOME": str(Path(home) / ".codex")},
            timeout=30,
        )

    def test_the_hook_reads_the_transcript_and_stays_silent(self):
        server, posts = receiver()
        with TemporaryDirectory() as home:
            try:
                path = write_fixture(home)
                install(home, f"http://127.0.0.1:{server.server_address[1]}")
                payload = json.dumps(
                    {
                        "hook_event_name": "Stop",
                        "session_id": THREAD,
                        "turn_id": TURN_TWO,
                        "transcript_path": str(path),
                        "last_assistant_message": "all done",
                        "model": "gpt-6",
                        "cwd": home,
                    }
                )
                proc = self._hook(home, payload)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stdout, "")
                emitted = records_of(posts)
                self.assertEqual([record["attrs"]["item.id"] for record in emitted], ["msg_first", "rs_first", "msg_final"])
                self.assertEqual(emitted[-1]["attrs"]["completion"], "all done")
                proc = self._hook(home, payload)
                self.assertEqual(proc.returncode, 0)
                self.assertEqual(len(records_of(posts)), 3)
            finally:
                server.shutdown()

    def test_the_hook_falls_back_to_the_last_assistant_message(self):
        server, posts = receiver()
        with TemporaryDirectory() as home:
            try:
                install(home, f"http://127.0.0.1:{server.server_address[1]}")
                payload = json.dumps({"hook_event_name": "Stop", "session_id": THREAD, "turn_id": TURN_TWO, "transcript_path": None, "last_assistant_message": "spoken but never written"})
                proc = self._hook(home, payload)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stdout, "")
                emitted = records_of(posts)
                self.assertEqual(len(emitted), 1)
                self.assertEqual(emitted[0]["attrs"]["completion"], "spoken but never written")
                self.assertEqual(emitted[0]["attrs"]["turn.id"], TURN_TWO)
                identifier = emitted[0]["attrs"]["item.id"]
                self.assertTrue(identifier.startswith(f"{THREAD}:{TURN_TWO}:"), identifier)
                proc = self._hook(home, payload)
                self.assertEqual(proc.returncode, 0)
                self.assertEqual(len(records_of(posts)), 1)
            finally:
                server.shutdown()

    def test_the_fallback_record_inherits_the_turn_trace_id(self):
        server, posts = receiver()
        with TemporaryDirectory() as home:
            try:
                path = write_fixture(home, lines=traced_lines(), truncated=False)
                store = ConfigStore(Path(home))
                cfg = SharedConfig(endpoint=f"http://127.0.0.1:{server.server_address[1]}", key=KEY, capture="full")
                codex_capture.Offsets(store.dir).commit(path, path.stat().st_size)
                payload = {"session_id": THREAD, "turn_id": TURN_ONE, "transcript_path": str(path), "last_assistant_message": "written after the hook fired"}
                with codex_home(home):
                    self.assertEqual(codex_capture.hook_emit(store, cfg, payload), 1)
                emitted = records_of(posts)
                self.assertEqual(len(emitted), 1)
                self.assertEqual(emitted[0]["attrs"]["completion"], "written after the hook fired")
                self.assertEqual(emitted[0]["trace"], TRACE_ONE)
                self.assertEqual(emitted[0]["attrs"]["turn.id"], TURN_ONE)
            finally:
                server.shutdown()

    def test_a_garbage_payload_exits_zero_and_says_nothing(self):
        server, posts = receiver()
        with TemporaryDirectory() as home:
            try:
                install(home, f"http://127.0.0.1:{server.server_address[1]}")
                for payload in ("", "not json at all", "[]", json.dumps({"hook_event_name": "Stop"})):
                    proc = self._hook(home, payload)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    self.assertEqual(proc.stdout, "")
                self.assertEqual(posts, [])
            finally:
                server.shutdown()

    def test_an_unreachable_endpoint_never_blocks_codex(self):
        with TemporaryDirectory() as home:
            write_fixture(home)
            install(home, "http://127.0.0.1:9")
            proc = self._hook(home, json.dumps({"session_id": THREAD, "turn_id": TURN_TWO, "last_assistant_message": "hi"}))
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(proc.stdout, "")

    def test_hook_emit_keeps_the_offset_when_the_post_fails(self):
        with TemporaryDirectory() as home:
            path = write_fixture(home)
            store = ConfigStore(Path(home))
            cfg = SharedConfig(endpoint="http://127.0.0.1:9", key=KEY, capture="full")
            with codex_home(home):
                self.assertEqual(codex_capture.hook_emit(store, cfg, {"session_id": THREAD, "transcript_path": str(path)}), 0)
            self.assertEqual(codex_capture.Offsets(store.dir).get(path), 0)


class TestCodexHooksInstall(unittest.TestCase):
    def _hooks(self, home):
        return json.loads((Path(home) / ".codex" / "hooks.json").read_text())

    def test_full_capture_writes_the_verified_codex_hook_shape(self):
        with TemporaryDirectory() as home:
            hooks = Path(home) / ".codex" / "hooks.json"
            hooks.parent.mkdir()
            hooks.write_text(json.dumps({"description": "mine", "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "/bin/true"}]}], "PostToolUse": []}}))
            code, out = install(home, "https://x.test/otlp")
            self.assertEqual(code, 0, out)
            self.assertIn("hooks.json", out)
            self.assertIn("merged 2 turn hooks (Stop, SessionEnd)", out)
            data = self._hooks(home)
            self.assertEqual(data["description"], "mine")
            self.assertEqual(data["hooks"]["Stop"][0], {"hooks": [{"type": "command", "command": "/bin/true"}]})
            entry = data["hooks"]["Stop"][1]["hooks"][0]
            self.assertEqual(entry["type"], "command")
            self.assertEqual(entry["timeout"], 5)
            self.assertTrue(entry["command"].endswith("coding hook --vendor codex --event Stop"))
            session_end = data["hooks"]["SessionEnd"][0]["hooks"][0]
            self.assertTrue(session_end["command"].endswith("--event SessionEnd"))
            self.assertEqual(session_end["timeout"], 3)
            self.assertNotIn(KEY, json.dumps(data))
            before = hooks.read_text()
            install(home, "https://x.test/otlp")
            self.assertEqual(hooks.read_text(), before)
            code, out = run(["coding", "status"], home)
            self.assertIn("hooks: 2 events", out)
            code, out = run(["coding", "uninstall", "--vendor", "codex"], home)
            self.assertEqual(code, 0, out)
            self.assertIn("removed 2 turn hooks", out)
            self.assertEqual(self._hooks(home), {"description": "mine", "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "/bin/true"}]}], "PostToolUse": []}})
            self.assertEqual((Path(home) / ".codex" / "config.toml").read_text(), "")

    def test_minimal_capture_writes_no_hooks_and_removes_ours(self):
        with TemporaryDirectory() as home:
            code, out = install(home, "https://x.test/otlp", capture="minimal")
            self.assertEqual(code, 0, out)
            self.assertFalse((Path(home) / ".codex" / "hooks.json").exists())
            install(home, "https://x.test/otlp", capture="full")
            self.assertIn("Stop", self._hooks(home)["hooks"])
            code, out = install(home, "https://x.test/otlp", capture="minimal")
            self.assertIn("removed evalkit turn hooks", out)
            self.assertEqual(self._hooks(home), {})
            code, out = run(["coding", "status"], home)
            self.assertNotIn("hooks:", out)

    def test_a_foreign_hook_shape_is_refused_instead_of_replaced(self):
        with TemporaryDirectory() as home:
            hooks = Path(home) / ".codex" / "hooks.json"
            hooks.parent.mkdir()
            original = json.dumps({"hooks": {"Stop": {"legacy": "shape"}}})
            hooks.write_text(original)
            code, _ = install(home, "https://x.test/otlp")
            self.assertEqual(code, 1)
            self.assertEqual(hooks.read_text(), original)


if __name__ == "__main__":
    unittest.main()
