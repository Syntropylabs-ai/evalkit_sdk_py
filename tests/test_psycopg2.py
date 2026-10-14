"""psycopg2 tracing.

psycopg2's connection and cursor are C types, so the SDK traces them through the connection and
cursor factories. Most tests run against a stand-in module with psycopg2's factory contract; the
last class runs against the real psycopg2 when it is installed (no server needed), and against a
real server when EVALKIT_TEST_POSTGRES_DSN names one.
"""
import importlib
import os
import sys
import types
import unittest

import evalkit
from tests.helpers import fresh, reset_context


def _fake_psycopg2():
    """psycopg2's connect() and factory contract, in Python."""
    pkg = types.ModuleType("psycopg2")
    ext = types.ModuleType("psycopg2.extensions")
    extras = types.ModuleType("psycopg2.extras")
    sql = types.ModuleType("psycopg2.sql")

    class OperationalError(Exception):
        pass

    class ProgrammingError(Exception):
        pass

    class cursor:
        def __init__(self, conn, name=None):
            self.connection = conn
            self.name = name
            self.rowcount = -1
            self.query = None
            self._rows = []
            self.closed = False

        def execute(self, query, vars=None):
            if "boom" in str(query):
                raise ProgrammingError("syntax error at or near \"boom\"")
            self.query = query
            self._rows = [(1, "ada")]
            self.rowcount = 1
            return None

        def executemany(self, query, vars_list):
            self.query = query
            self.rowcount = len(list(vars_list))
            return None

        def fetchall(self):
            return list(self._rows)

        def fetchone(self):
            return self._rows[0] if self._rows else None

        def mogrify(self, query, vars=None):
            return b"mogrified"

        def close(self):
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    class connection:
        def __init__(self, dsn, async_=0):
            if "port=1 " in dsn + " ":
                raise OperationalError('connection to server at "127.0.0.1", port 1 failed: Connection refused')
            self.dsn = dsn
            self.cursor_factory = None
            self.info = types.SimpleNamespace(dbname="orders", host="db.internal", port=5432)
            self.closed = 0

        def cursor(self, name=None, cursor_factory=None, withhold=False, scrollable=None):
            factory = cursor_factory or self.cursor_factory or cursor
            return factory(self, name)

        def close(self):
            self.closed = 1

    class DictCursor(cursor):
        def execute(self, query, vars=None):
            self.dict_executed = True
            return super().execute(query, vars)

    class RealDictCursor(cursor):
        pass

    class DictConnection(connection):
        def cursor(self, *args, **kwargs):
            kwargs.setdefault("cursor_factory", self.cursor_factory or DictCursor)
            return super().cursor(*args, **kwargs)

    class Composed:
        def __init__(self, text):
            self.text = text

        def as_string(self, context):
            return self.text

    def connect(dsn=None, connection_factory=None, cursor_factory=None, **kwargs):
        dsn = dsn or " ".join(f"{k}={v}" for k, v in kwargs.items())
        conn = (connection_factory or connection)(dsn)
        if cursor_factory is not None:
            conn.cursor_factory = cursor_factory
        return conn

    ext.connection, ext.cursor = connection, cursor
    extras.DictCursor, extras.RealDictCursor, extras.DictConnection = DictCursor, RealDictCursor, DictConnection
    sql.Composed = Composed
    pkg.extensions, pkg.extras, pkg.sql = ext, extras, sql
    pkg.connect = connect
    pkg.OperationalError, pkg.ProgrammingError = OperationalError, ProgrammingError
    return pkg


def _reset_module():
    from evalkit.instrumentation.db import psycopg2 as ek_pg2

    ek_pg2._mixin = None
    ek_pg2._cursor_classes.clear()
    ek_pg2._connection_classes.clear()
    return ek_pg2


class _Base(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._saved_modules = {n: sys.modules.get(n) for n in
                               ("psycopg2", "psycopg2.extensions", "psycopg2.extras", "psycopg2.sql")}
        self.pg = _fake_psycopg2()
        sys.modules.update({"psycopg2": self.pg, "psycopg2.extensions": self.pg.extensions,
                            "psycopg2.extras": self.pg.extras, "psycopg2.sql": self.pg.sql})
        self.ek = _reset_module()
        self.assertTrue(self.ek.patch_psycopg2(self.client))

    def tearDown(self):
        for name, mod in self._saved_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        _reset_module()

    def _db(self):
        self.client.flush()
        return [e for e in self.captured if e.span_type == "db_query"]


class TestPsycopg2Tracing(_Base):
    def test_connect_execute_and_executemany_are_recorded(self):
        tid, end, ctx = self.client.start_trace("req")
        with evalkit.with_trace(ctx):
            conn = self.pg.connect("dbname=orders user=app password=pw-secret host=db.internal")
            with conn.cursor() as cur:
                self.assertIsNone(cur.execute("SELECT id, name FROM customers WHERE id = %s", (1,)))
                self.assertEqual(cur.fetchall(), [(1, "ada")])
                cur.executemany("INSERT INTO orders (id) VALUES (%s)", [(1,), (2,), (3,)])
        end("OK")
        spans = self._db()
        self.assertEqual([s.operation for s in spans], ["postgresql CONNECT", "SELECT customers", "INSERT orders"])
        connect, select, insert = spans
        self.assertEqual(connect.attributes["db.system"], "postgresql")
        self.assertEqual(connect.attributes["db.operation"], "CONNECT")
        self.assertEqual(connect.attributes["db.name"], "orders")
        self.assertEqual(connect.attributes["server.address"], "db.internal")
        self.assertEqual(connect.attributes["server.port"], 5432)
        self.assertNotIn("pw-secret", repr(connect.attributes), "the DSN, and its password, is never recorded")
        self.assertEqual(select.attributes["db.statement"], "SELECT id, name FROM customers WHERE id = %s")
        self.assertEqual(select.attributes["db.sql.table"], "customers")
        self.assertEqual(select.attributes["db.rows_affected"], 1)
        self.assertNotIn("db.params", select.attributes, "bound values are not recorded, as with psycopg 3")
        self.assertEqual(insert.attributes["db.operation"], "INSERT")
        self.assertEqual(insert.attributes["db.rows_affected"], 3)
        self.assertTrue(all(s.trace_id == tid for s in spans))

    def test_cursor_classes_are_kept(self):
        extras = self.pg.extras
        conn = self.pg.connect("dbname=orders")
        plain = conn.cursor()
        real_dict = conn.cursor(cursor_factory=extras.RealDictCursor)
        positional = conn.cursor(None, extras.RealDictCursor)
        self.assertIsInstance(plain, self.pg.extensions.cursor)
        self.assertIsInstance(real_dict, extras.RealDictCursor)
        self.assertIsInstance(positional, extras.RealDictCursor)
        self.assertEqual(type(real_dict).__name__, "RealDictCursor")
        self.assertEqual(real_dict.mogrify("SELECT 1"), b"mogrified")
        real_dict.execute("SELECT 1")
        positional.execute("SELECT 2")
        self.assertEqual(len(self._db()), 3, "connect plus both cursors' statements")

    def test_a_connection_level_cursor_factory_is_kept(self):
        extras = self.pg.extras
        conn = self.pg.connect("dbname=orders", cursor_factory=extras.RealDictCursor)
        cur = conn.cursor()
        self.assertIsInstance(cur, extras.RealDictCursor)
        cur.execute("SELECT 1")
        self.assertEqual(len(self._db()), 2)

    def test_a_connection_subclass_keeps_its_own_default_cursor(self):
        extras = self.pg.extras
        conn = self.pg.connect("dbname=orders", connection_factory=extras.DictConnection)
        self.assertIsInstance(conn, extras.DictConnection)
        cur = conn.cursor()
        self.assertIsInstance(cur, extras.DictCursor, "DictConnection hands out DictCursors; tracing must not change that")
        cur.execute("SELECT 1")
        self.assertTrue(cur.dict_executed)
        self.assertEqual([s.operation for s in self._db()], ["postgresql CONNECT", "postgresql SELECT"])

    def test_a_cursor_factory_function_is_used_as_given(self):
        made = []

        def factory(conn, name=None):
            made.append(name)
            return self.pg.extensions.cursor(conn, name)

        conn = self.pg.connect("dbname=orders")
        cur = conn.cursor(cursor_factory=factory)
        cur.execute("SELECT 1")
        self.assertEqual(made, [None])

    def test_a_composed_query_is_recorded_as_sent(self):
        conn = self.pg.connect("dbname=orders")
        conn.cursor().execute(self.pg.sql.Composed("SELECT * FROM refunds"))
        self.assertEqual(self._db()[-1].attributes["db.statement"], "SELECT * FROM refunds")

    def test_a_failed_statement_raises_the_drivers_exception(self):
        conn = self.pg.connect("dbname=orders")
        cur = conn.cursor()
        with self.assertRaises(self.pg.ProgrammingError) as raised:
            cur.execute("SELECT boom")
        self.assertIn("boom", str(raised.exception))
        span = self._db()[-1]
        self.assertEqual(span.status, "ERROR")
        self.assertIn("syntax error", span.status_message)

    def test_a_failed_connect_is_recorded_and_raised(self):
        with self.assertRaises(self.pg.OperationalError):
            self.pg.connect(host="127.0.0.1", port=1, dbname="orders")
        span = self._db()[-1]
        self.assertEqual((span.operation, span.status), ("postgresql CONNECT", "ERROR"))
        self.assertIn("Connection refused", span.status_message)

    def test_capture_content_off_drops_the_query_text(self):
        saved = self.client._opts.capture_content
        self.client._opts.capture_content = False
        try:
            self.pg.connect("dbname=orders").cursor().execute("SELECT * FROM customers WHERE email = 'ada@x.io'")
            span = self._db()[-1]
        finally:
            self.client._opts.capture_content = saved
        self.assertNotIn("db.statement", span.attributes)
        self.assertEqual(span.attributes["db.operation"], "SELECT")
        self.assertEqual(span.attributes["db.system"], "postgresql")

    def test_a_failure_in_the_tracer_leaves_the_query_alone(self):
        from evalkit.instrumentation.db import psycopg2 as ek_pg2

        original = ek_pg2.start_db

        def broken(*a, **k):
            raise RuntimeError("tracer bug")

        ek_pg2.start_db = broken
        try:
            conn = self.pg.connect("dbname=orders")
            cur = conn.cursor()
            self.assertIsNone(cur.execute("SELECT 1"))
            self.assertEqual(cur.fetchone(), (1, "ada"))
        finally:
            ek_pg2.start_db = original

    def test_patching_twice_wraps_once(self):
        connect = self.pg.connect
        self.assertTrue(self.ek.patch_psycopg2(self.client))
        self.assertIs(self.pg.connect, connect)
        self.pg.connect("dbname=orders").cursor().execute("SELECT 1")
        self.assertEqual(len(self._db()), 2)

    def test_statements_sqlalchemy_already_records_are_not_recorded_again(self):
        from evalkit.instrumentation.db._common import _orm_execution

        conn = self.pg.connect("dbname=orders")
        self.client.flush()
        self.captured.clear()
        recorded_above = types.SimpleNamespace(_otel_span=object())
        token = _orm_execution.set(recorded_above)
        try:
            conn.cursor().execute("SELECT 1")
        finally:
            _orm_execution.reset(token)
        self.assertEqual(self._db(), [], "SQLAlchemy's own span is the record of this statement")
        token = _orm_execution.set(types.SimpleNamespace(_otel_span=None, root_connection=None))
        try:
            conn.cursor().execute("SELECT 2")
        finally:
            _orm_execution.reset(token)
        self.assertEqual(len(self._db()), 1, "SQLAlchemy untraced: the driver span is the only record")


class TestOwnedScopes(unittest.TestCase):
    """OpenTelemetry's psycopg2 spans are dropped only when the SDK records the same queries itself."""

    def setUp(self):
        reset_context()
        self.client = fresh([])
        from evalkit.core import otel_provider

        self.owned = otel_provider._OWNED_SCOPES
        self.saved_owned = set(self.owned)
        self.saved_module = sys.modules.get("psycopg2", "absent")

    def tearDown(self):
        self.owned.clear()
        self.owned.update(self.saved_owned)
        if self.saved_module == "absent":
            sys.modules.pop("psycopg2", None)
        else:
            sys.modules["psycopg2"] = self.saved_module
        for name in ("psycopg2.extensions", "psycopg2.extras", "psycopg2.sql"):
            if self.saved_module == "absent":
                sys.modules.pop(name, None)
        _reset_module()

    def test_scope_is_not_owned_when_psycopg2_is_not_traced(self):
        from evalkit.instrumentation.auto import _try_patch_databases

        self.owned.discard("opentelemetry.instrumentation.psycopg2")
        sys.modules["psycopg2"] = None
        _try_patch_databases(self.client)
        self.assertNotIn("opentelemetry.instrumentation.psycopg2", self.owned)

    def test_scope_is_owned_once_psycopg2_is_traced(self):
        from evalkit.instrumentation.auto import _try_patch_databases

        self.owned.discard("opentelemetry.instrumentation.psycopg2")
        pg = _fake_psycopg2()
        sys.modules.update({"psycopg2": pg, "psycopg2.extensions": pg.extensions,
                            "psycopg2.extras": pg.extras, "psycopg2.sql": pg.sql})
        _reset_module()
        _try_patch_databases(self.client)
        self.assertIn("opentelemetry.instrumentation.psycopg2", self.owned)


def _real_psycopg2():
    try:
        saved = {n: sys.modules.pop(n) for n in list(sys.modules) if n == "psycopg2" or n.startswith("psycopg2.")}
        try:
            return importlib.import_module("psycopg2"), saved
        except ImportError:
            sys.modules.update(saved)
            return None, None
    except Exception:
        return None, None


class TestRealPsycopg2(unittest.TestCase):
    """The real driver's C types, when psycopg2 is installed."""

    @classmethod
    def setUpClass(cls):
        cls.pg, cls.saved = _real_psycopg2()
        if cls.pg is None:
            raise unittest.SkipTest("psycopg2 is not installed")

    @classmethod
    def tearDownClass(cls):
        for name in [n for n in sys.modules if n == "psycopg2" or n.startswith("psycopg2.")]:
            sys.modules.pop(name, None)
        sys.modules.update(cls.saved or {})
        _reset_module()

    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self.ek = _reset_module()
        self.original_connect = getattr(self.pg.connect, "__wrapped__", self.pg.connect)
        self.pg.connect = self.original_connect
        self.assertTrue(self.ek.patch_psycopg2(self.client))

    def tearDown(self):
        self.pg.connect = self.original_connect

    def test_traced_classes_sit_on_the_real_c_types(self):
        import psycopg2.extensions as ext
        import psycopg2.extras as extras

        mro = [k.__name__ for k in self.ek.traced_connection_class(extras.DictConnection, ext).__mro__]
        self.assertEqual(mro[:4], ["DictConnection", "DictConnection", "connection", "connection"])
        for base in (ext.cursor, extras.DictCursor, extras.RealDictCursor, extras.NamedTupleCursor):
            traced = self.ek.traced_cursor_class(base)
            self.assertTrue(issubclass(traced, base))
            self.assertEqual(traced.__name__, base.__name__)

    def test_a_refused_connection_is_recorded(self):
        with self.assertRaises(self.pg.OperationalError):
            self.pg.connect(host="127.0.0.1", port=1, dbname="x", user="u", password="pw-secret", connect_timeout=2)
        self.client.flush()
        spans = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual([(s.operation, s.status) for s in spans], [("postgresql CONNECT", "ERROR")])
        self.assertNotIn("pw-secret", repr(spans[0].attributes))

    def test_against_a_server(self):
        dsn = os.environ.get("EVALKIT_TEST_POSTGRES_DSN")
        if not dsn:
            self.skipTest("EVALKIT_TEST_POSTGRES_DSN is not set")
        import psycopg2.extras as extras
        import psycopg2.pool as pool
        from psycopg2 import sql

        conn = self.pg.connect(dsn)
        with conn, conn.cursor() as cur:
            self.assertIsNone(cur.execute("CREATE TEMP TABLE t_orders (id int, item text)"))
            cur.executemany("INSERT INTO t_orders VALUES (%s, %s)", [(1, "blender"), (2, "kettle")])
            cur.execute("SELECT id, item FROM t_orders ORDER BY id")
            self.assertEqual(cur.fetchall(), [(1, "blender"), (2, "kettle")])
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute(sql.SQL("SELECT {} FROM t_orders WHERE id = %s").format(sql.Identifier("item")), (2,))
            self.assertEqual(cur.fetchone(), {"item": "kettle"})
        extras.execute_values(conn.cursor(), "INSERT INTO t_orders VALUES %s", [(3, "toaster")])
        conn.close()
        dict_conn = self.pg.connect(dsn, connection_factory=extras.DictConnection)
        with dict_conn.cursor() as cur:
            cur.execute("SELECT 1 AS one")
            self.assertEqual(cur.fetchone()["one"], 1)
        dict_conn.close()
        p = pool.SimpleConnectionPool(1, 1, dsn)
        pooled = p.getconn()
        pooled.cursor().execute("SELECT 2")
        p.putconn(pooled)
        p.closeall()

        self.client.flush()
        spans = [e for e in self.captured if e.span_type == "db_query"]
        ops = [s.operation for s in spans]
        self.assertEqual(ops.count("postgresql CONNECT"), 3)
        self.assertIn("INSERT t_orders", ops)
        self.assertIn('SELECT t_orders', ops)
        self.assertTrue(any(s.attributes.get("db.statement") == 'SELECT "item" FROM t_orders WHERE id = %s'
                            for s in spans), [s.attributes.get("db.statement") for s in spans])
        self.assertTrue(all(s.status == "OK" for s in spans))


if __name__ == "__main__":
    unittest.main()
