import unittest

from evalkit.instrumentation.db import _common
from evalkit.instrumentation.db import sqlalchemy as sa


class _Opts:
    def __init__(self, limit):
        self.max_body_bytes = limit
        self.service_name = "svc"
        self.environment = "test"
        self.app_version = None
        self.device_id = None


class _Client:
    def __init__(self, limit):
        self._opts = _Opts(limit)
        self.spans = []

    def _meta_attrs(self):
        return {}

    def enqueue(self, envelope):
        self.spans.append(envelope)


class TestStatementCap(unittest.TestCase):
    def setUp(self):
        self.saved = _common.stmt_max()

    def tearDown(self):
        _common._STMT_MAX = self.saved

    def test_the_cap_follows_max_body_bytes(self):
        _common.set_stmt_max(50_000)
        client = _Client(50_000)
        end = _common.start_db(client, "postgresql", statement="SELECT " + "a" * 40_000)
        end("OK")
        self.assertEqual(
            len(client.spans[0].attributes["db.statement"]), 40_007,
            "a 2 KB statement cap silently truncated the long analytical queries that are exactly the ones worth "
            "tracing; the db path must honour max_body_bytes like the http path does",
        )

    def test_a_statement_past_the_cap_is_still_clipped(self):
        _common.set_stmt_max(100)
        client = _Client(100)
        end = _common.start_db(client, "postgresql", statement="x" * 500)
        end("OK")
        self.assertEqual(
            len(client.spans[0].attributes["db.statement"]), 100,
            "honouring the option cannot mean dropping the bound; an unbounded statement would put an arbitrary blob "
            "on the wire",
        )

    def test_a_nonsense_cap_is_ignored(self):
        _common.set_stmt_max(0)
        self.assertEqual(_common.stmt_max(), self.saved)
        _common.set_stmt_max(-1)
        self.assertEqual(_common.stmt_max(), self.saved)
        _common.set_stmt_max(None)
        self.assertEqual(
            _common.stmt_max(), self.saved,
            "a bad option value must leave the working default alone rather than turn statement capture into a "
            "zero-length string",
        )


class TestSqlalchemyCaps(unittest.TestCase):
    def setUp(self):
        self.saved = (sa._BODY_MAX, sa._PARAMS_MAX, _common.stmt_max())

    def tearDown(self):
        sa._BODY_MAX, sa._PARAMS_MAX, _common._STMT_MAX = self.saved

    def test_setting_caps_lifts_both_the_statement_and_the_parameter_limit(self):
        sa._set_caps(_Client(65_536))
        self.assertEqual(sa._BODY_MAX, 65_536)
        self.assertEqual(
            sa._PARAMS_MAX, 65_536,
            "bound parameters were hardcoded to 1 KB, so a bulk insert recorded a statement with its values cut off - "
            "the half that actually tells you what went into the database",
        )
        self.assertEqual(
            _common.stmt_max(), 65_536,
            "patch_sqlalchemy can be called on its own without auto-instrumentation, so it has to lift the shared "
            "db cap too or the two tracers disagree about the same option",
        )

    def test_a_missing_option_leaves_the_caps_untouched(self):
        class _Bare:
            _opts = object()

        sa._BODY_MAX, sa._PARAMS_MAX = 111, 222
        sa._set_caps(_Bare())
        self.assertEqual(
            (sa._BODY_MAX, sa._PARAMS_MAX), (111, 222),
            "reading the option must never raise or zero the caps; instrumentation that crashes on a hand-built "
            "client is worse than instrumentation with a small cap",
        )


if __name__ == "__main__":
    unittest.main()
