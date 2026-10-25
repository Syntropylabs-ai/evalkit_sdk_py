import logging
import unittest

import evalkit
from tests.helpers import fresh, reset_context


def _install_handler(client, name):
    from evalkit.instrumentation.runtime.logging import EvalKitLoggingHandler

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = [h for h in logger.handlers if not isinstance(h, EvalKitLoggingHandler)]
    handler = EvalKitLoggingHandler(client)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class TestEvalKitLogging(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_warning_emits_log_span(self):
        logger = _install_handler(self.client, "app.test.warn")
        tid, end, ctx = self.client.start_trace("req")
        with evalkit.with_trace(ctx):
            logger.warning("something happened")
        end("OK")
        self.client.flush()
        logs = [
            e for e in self.captured
            if e.span_type == "log" and "something happened" in (e.attributes or {}).get("log.message", "")
        ]
        self.assertTrue(logs, "warning should emit a log span")
        self.assertEqual(logs[0].trace_id, tid)

    def test_error_log_has_error_status(self):
        logger = _install_handler(self.client, "app.test.error")
        logger.error("it broke")
        self.client.flush()
        logs = [
            e for e in self.captured
            if e.span_type == "log" and "it broke" in (e.attributes or {}).get("log.message", "")
        ]
        self.assertTrue(logs)
        self.assertEqual(logs[0].status, "ERROR")

    def test_log_level_in_attributes(self):
        logger = _install_handler(self.client, "app.test.level")
        logger.info("info message")
        logger.warning("warn message")
        self.client.flush()
        logs = [
            e for e in self.captured
            if e.span_type == "log" and "message" in (e.attributes or {}).get("log.message", "")
        ]
        levels = {e.attributes["log.level"] for e in logs if "log.level" in (e.attributes or {})}
        self.assertIn("INFO", levels)
        self.assertIn("WARN", levels)

    def test_install_logging_handler_export(self):
        tid, end, ctx = self.client.start_trace("logging-req")
        logger = logging.getLogger("app.test.install")
        # The logger's level is the application's to set; install_logging_handler leaves it alone.
        logger.setLevel(logging.INFO)
        with evalkit.with_trace(ctx):
            evalkit.install_logging_handler(self.client, logger_name="app.test.install")
            logger.info("via install_logging_handler")
        end("OK")
        self.client.flush()
        logs = [
            e for e in self.captured
            if e.span_type == "log"
            and "via install_logging_handler" in (e.attributes or {}).get("log.message", "")
        ]
        self.assertTrue(logs)



_APP = r"""
import logging, logging.config, os, sys
mode, with_init = sys.argv[1], sys.argv[2] == "1"
FMT = "%(levelname)s|%(name)s|%(message)s"

def configure():
    if mode == "basic":
        logging.basicConfig(level=logging.INFO, format=FMT, stream=sys.stderr)
    elif mode == "dict":
        logging.config.dictConfig({"version": 1, "formatters": {"f": {"format": FMT}},
            "handlers": {"h": {"class": "logging.StreamHandler", "formatter": "f", "stream": "ext://sys.stderr"}},
            "root": {"level": "INFO", "handlers": ["h"]}})
    elif mode == "handler":
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter(FMT))
        logging.getLogger("app").addHandler(h)
        logging.getLogger("app").setLevel(logging.INFO)

if mode != "basic_after":
    configure()
if with_init:
    import evalkit
    client = evalkit.init(subscription_key="tk_test", base_url="http://127.0.0.1:1", service_name="logcheck",
                          function_tracing=False)
    client._exporter.export = lambda batch: True
if mode == "basic_after":
    mode = "basic"
    configure()
log = logging.getLogger("app")
log.debug("debug line")
log.info("info line")
log.warning("warning line %s", 1)
logging.getLogger("app.sub").error("error line")
logging.warning("root warning")
try:
    1 / 0
except ZeroDivisionError:
    log.exception("with traceback")
print("ROOT", logging.getLogger().level, len(logging.getLogger().handlers), file=sys.stdout)
"""


class TestInitLeavesTheAppsLoggingAlone(unittest.TestCase):
    """init() must not change what an application's logging prints."""

    def _run(self, mode, with_init):
        import subprocess
        import sys as _sys
        import os as _os

        env = {**_os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True",
               "PYTHONPATH": _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))}
        out = subprocess.run([_sys.executable, "-c", _APP, mode, "1" if with_init else "0"],
                             capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        stderr = "\n".join(line for line in out.stderr.splitlines() if 'File "<string>"' not in line)
        return stderr, out.stdout

    def test_output_is_identical_with_and_without_init(self):
        for mode in ("basic", "basic_after", "dict", "handler", "unconfigured"):
            with self.subTest(mode=mode):
                without = self._run(mode, False)
                with_init = self._run(mode, True)
                self.assertEqual(
                    with_init, without,
                    "init() changed the application's log output: it replaced handlers' formatters, set a "
                    "level, or added a handler that made a later basicConfig() a no-op",
                )
                self.assertIn("warning line 1", without[0])


class TestLogCapture(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        from evalkit.instrumentation.runtime.logging import install_log_capture

        install_log_capture(self.client)

    def _logs(self, text):
        self.client.flush()
        return [e for e in self.captured
                if e.span_type == "log" and text in (e.attributes or {}).get("log.message", "")]

    def test_records_reaching_the_root_logger_become_log_spans(self):
        root = logging.getLogger()
        handlers, level = list(root.handlers), root.level
        tid, end, ctx = self.client.start_trace("req")
        with evalkit.with_trace(ctx):
            logging.getLogger("capture.app").warning("captured without a handler")
        end("OK")
        logs = self._logs("captured without a handler")
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0].trace_id, tid)
        self.assertEqual(root.handlers, handlers, "capture must not add a handler to the root logger")
        self.assertEqual(root.level, level)

    def test_a_logger_that_does_not_propagate_is_not_captured(self):
        logger = logging.getLogger("capture.private")
        logger.propagate = False
        logger.addHandler(logging.NullHandler())
        try:
            logger.warning("kept private")
        finally:
            logger.propagate = True
        self.assertEqual(self._logs("kept private"), [])

    def test_records_carry_trace_ids_for_formats_that_ask(self):
        import io

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(trace_id)s %(message)s"))
        logger = logging.getLogger("capture.ids")
        logger.addHandler(handler)
        try:
            tid, end, ctx = self.client.start_trace("req")
            with evalkit.with_trace(ctx):
                logger.warning("with ids")
                logger.warning("own id", extra={"trace_id": "mine"})
            end("OK")
        finally:
            logger.removeHandler(handler)
        lines = stream.getvalue().splitlines()
        self.assertEqual(lines[0], f"{tid} with ids")
        self.assertEqual(lines[1], "mine own id", "a trace_id the caller passed in extra= wins")

    def test_an_explicit_handler_and_the_capture_record_a_record_once(self):
        logger = logging.getLogger("capture.both")
        evalkit.install_logging_handler(self.client, logger_name="capture.both")
        logger.warning("recorded once")
        self.assertEqual(len(self._logs("recorded once")), 1)

    def test_install_logging_handler_leaves_levels_and_formatters_alone(self):
        logger = logging.getLogger("capture.untouched")
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        try:
            evalkit.install_logging_handler(self.client, logger_name="capture.untouched")
            self.assertEqual(logger.level, logging.NOTSET)
            self.assertIs(handler.formatter, formatter)
            self.assertEqual(handler.filters, [])
        finally:
            logger.handlers = []

if __name__ == "__main__":
    unittest.main(verbosity=2)
