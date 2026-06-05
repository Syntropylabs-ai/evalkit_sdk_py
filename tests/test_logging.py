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
        logger = _install_handler(self.client, "evalkit.test.warn")
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
        logger = _install_handler(self.client, "evalkit.test.error")
        logger.error("it broke")
        self.client.flush()
        logs = [
            e for e in self.captured
            if e.span_type == "log" and "it broke" in (e.attributes or {}).get("log.message", "")
        ]
        self.assertTrue(logs)
        self.assertEqual(logs[0].status, "ERROR")

    def test_log_level_in_attributes(self):
        logger = _install_handler(self.client, "evalkit.test.level")
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
        logger = logging.getLogger("evalkit.test.install")
        with evalkit.with_trace(ctx):
            evalkit.install_logging_handler(self.client, logger_name="evalkit.test.install")
            logger.info("via install_logging_handler")
        end("OK")
        self.client.flush()
        logs = [
            e for e in self.captured
            if e.span_type == "log"
            and "via install_logging_handler" in (e.attributes or {}).get("log.message", "")
        ]
        self.assertTrue(logs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
