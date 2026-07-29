import asyncio
import sys
import types
import unittest

import evalkit
from tests.helpers import fresh, reset_context


class TestDBCommon(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_start_db_span(self):
        from evalkit.instrumentation.db._common import start_db

        tid, end_tr, ctx = self.client.start_trace("req")
        with evalkit.with_trace(ctx):
            end = start_db(self.client, "postgresql", statement="SELECT * FROM users WHERE id=1")
            end("OK", rows=1)
        end_tr("OK")
        self.client.flush()
        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertTrue(db)
        a = db[0].attributes
        self.assertEqual(a["db.system"], "postgresql")
        self.assertEqual(a["db.operation"], "SELECT")
        self.assertEqual(a["db.sql.table"], "users")
        self.assertEqual(a["db.rows_affected"], 1)
        self.assertEqual(db[0].trace_id, tid)


class TestSQLAlchemy(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_sqlalchemy_db_query(self):
        try:
            from sqlalchemy import create_engine, text
        except ImportError:
            self.skipTest("sqlalchemy not installed")

        eng = create_engine("sqlite:///:memory:")
        evalkit.patch_sqlalchemy_engine(eng)
        tid, end, ctx = self.client.start_trace("req")
        with evalkit.with_trace(ctx):
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
        end("OK")
        self.client.flush()
        eng.dispose()
        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertTrue(db, "expected a db_query span")
        self.assertEqual(db[0].trace_id, tid)
        self.assertIn("db.statement", db[0].attributes)

    def test_skips_adk_session_tables_but_not_others(self):
        # Regression test: an ADK-based agent's own DatabaseSessionService
        # queries `sessions`/`app_states`/`user_states` several times per
        # turn — near-zero-duration bookkeeping that used to flood a trace
        # with low-signal spans around every real tool/model call.
        try:
            from sqlalchemy import create_engine, text
        except ImportError:
            self.skipTest("sqlalchemy not installed")

        eng = create_engine("sqlite:///:memory:")
        evalkit.patch_sqlalchemy_engine(eng)
        tid, end, ctx = self.client.start_trace("req")
        with evalkit.with_trace(ctx):
            with eng.connect() as conn:
                conn.execute(text("CREATE TABLE sessions (id INTEGER)"))
                conn.execute(text("CREATE TABLE app_states (id INTEGER)"))
                conn.execute(text("CREATE TABLE user_states (id INTEGER)"))
                conn.execute(text("CREATE TABLE widgets (id INTEGER)"))
                conn.execute(text("SELECT * FROM sessions"))
                conn.execute(text("SELECT * FROM app_states"))
                conn.execute(text("SELECT * FROM user_states"))
                conn.execute(text("SELECT * FROM widgets"))
        end("OK")
        self.client.flush()
        eng.dispose()
        tables = {e.attributes.get("db.sql.table") for e in self.captured if e.span_type == "db_query"}
        self.assertNotIn("sessions", tables)
        self.assertNotIn("app_states", tables)
        self.assertNotIn("user_states", tables)
        self.assertIn("widgets", tables)


class TestRedis(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._injected = []

    def tearDown(self):
        for n in self._injected:
            sys.modules.pop(n, None)

    def test_redis_sync_and_async(self):
        redis_mod = types.ModuleType("redis")
        client_mod = types.ModuleType("redis.client")

        class FakeRedis:
            def execute_command(self, *args, **kw):
                return "OK"

        client_mod.Redis = FakeRedis
        redis_mod.client = client_mod
        asyncio_mod = types.ModuleType("redis.asyncio")
        asyncio_client_mod = types.ModuleType("redis.asyncio.client")

        class FakeAsyncRedis:
            async def execute_command(self, *a, **k):
                return "OK"

        asyncio_client_mod.Redis = FakeAsyncRedis
        asyncio_mod.client = asyncio_client_mod

        for n, m in (
            ("redis", redis_mod), ("redis.client", client_mod),
            ("redis.asyncio", asyncio_mod), ("redis.asyncio.client", asyncio_client_mod),
        ):
            sys.modules[n] = m
            self._injected.append(n)

        from evalkit.instrumentation.db.redis import patch_redis
        patch_redis(self.client)

        FakeRedis().execute_command("GET", "mykey")
        asyncio.run(FakeAsyncRedis().execute_command("SET", "k", "v"))
        self.client.flush()
        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 2)
        ops = {e.attributes["db.operation"] for e in db}
        self.assertEqual(ops, {"GET", "SET"})
        self.assertTrue(all(e.attributes["db.system"] == "redis" for e in db))


class TestAsyncpg(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._injected = []

    def tearDown(self):
        for n in self._injected:
            sys.modules.pop(n, None)

    def test_asyncpg_execute(self):
        asyncpg_mod = types.ModuleType("asyncpg")
        conn_mod = types.ModuleType("asyncpg.connection")

        class FakeConn:
            async def execute(self, query, *a, **k):
                return "INSERT 0 1"

        conn_mod.Connection = FakeConn
        asyncpg_mod.connection = conn_mod
        sys.modules["asyncpg"] = asyncpg_mod
        sys.modules["asyncpg.connection"] = conn_mod
        self._injected += ["asyncpg", "asyncpg.connection"]

        from evalkit.instrumentation.db.asyncpg import patch_asyncpg
        patch_asyncpg(self.client)
        asyncio.run(FakeConn().execute("SELECT * FROM accounts"))
        self.client.flush()
        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertTrue(db)
        self.assertEqual(db[0].attributes["db.system"], "postgresql")
        self.assertEqual(db[0].attributes["db.operation"], "SELECT")
        self.assertEqual(db[0].attributes["db.sql.table"], "accounts")


class TestPsycopg(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._injected = []

    def tearDown(self):
        for n in self._injected:
            sys.modules.pop(n, None)

    def test_psycopg_execute(self):
        # patch_psycopg patches psycopg.Cursor.execute — NOT Connection.execute
        psycopg_mod = types.ModuleType("psycopg")

        class FakeCursor:
            rowcount = 3

            def execute(self, query, params=None):
                return None

        psycopg_mod.Cursor = FakeCursor
        psycopg_mod.AsyncCursor = FakeCursor  # same class is fine for this test
        sys.modules["psycopg"] = psycopg_mod
        self._injected.append("psycopg")

        from evalkit.instrumentation.db.psycopg import patch_psycopg
        patch_psycopg(self.client)

        tid, end_tr, ctx = self.client.start_trace("req")
        with evalkit.with_trace(ctx):
            FakeCursor().execute("INSERT INTO orders (id) VALUES (%s)", (1,))
        end_tr("OK")
        self.client.flush()
        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertTrue(db, "expected a db_query span from psycopg")
        self.assertEqual(db[0].trace_id, tid)
        self.assertEqual(db[0].attributes.get("db.system"), "postgresql")


class TestPyMySQL(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._injected = []

    def tearDown(self):
        for n in self._injected:
            sys.modules.pop(n, None)

    def test_pymysql_execute(self):
        pymysql_mod = types.ModuleType("pymysql")
        cursors_mod = types.ModuleType("pymysql.cursors")

        class FakeCursor:
            rowcount = 2

            def execute(self, query, args=None):
                return None

        cursors_mod.Cursor = FakeCursor
        pymysql_mod.cursors = cursors_mod
        sys.modules["pymysql"] = pymysql_mod
        sys.modules["pymysql.cursors"] = cursors_mod
        self._injected += ["pymysql", "pymysql.cursors"]

        from evalkit.instrumentation.db.pymysql import patch_pymysql
        patch_pymysql(self.client)

        tid, end_tr, ctx = self.client.start_trace("req")
        with evalkit.with_trace(ctx):
            FakeCursor().execute("SELECT id FROM products LIMIT 10")
        end_tr("OK")
        self.client.flush()
        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertTrue(db, "expected a db_query span from pymysql")
        self.assertEqual(db[0].attributes.get("db.system"), "mysql")


class TestPyMongo(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)
        self._injected = []

    def tearDown(self):
        for n in self._injected:
            sys.modules.pop(n, None)

    def test_pymongo_find_one(self):
        # patch_pymongo patches "find_one", "insert_one", etc. — not bare "find"
        pymongo_mod = types.ModuleType("pymongo")
        collection_mod = types.ModuleType("pymongo.collection")

        class FakeCollection:
            name = "users"

            def find_one(self, *a, **k):
                return {"_id": 1}

            def insert_one(self, *a, **k):
                return types.SimpleNamespace(inserted_id=1)

        collection_mod.Collection = FakeCollection
        pymongo_mod.collection = collection_mod
        sys.modules["pymongo"] = pymongo_mod
        sys.modules["pymongo.collection"] = collection_mod
        self._injected += ["pymongo", "pymongo.collection"]

        from evalkit.instrumentation.db.pymongo import patch_pymongo
        patch_pymongo(self.client)

        tid, end_tr, ctx = self.client.start_trace("req")
        with evalkit.with_trace(ctx):
            FakeCollection().find_one({"active": True})
        end_tr("OK")
        self.client.flush()
        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertTrue(db, "expected a db_query span from pymongo")
        self.assertEqual(db[0].attributes.get("db.system"), "mongodb")

    def test_pymongo_insert_one(self):
        pymongo_mod = types.ModuleType("pymongo")
        collection_mod = types.ModuleType("pymongo.collection")

        class FakeCollection:
            name = "orders"

            def insert_one(self, *a, **k):
                return types.SimpleNamespace(inserted_id=42)

        collection_mod.Collection = FakeCollection
        pymongo_mod.collection = collection_mod
        sys.modules["pymongo"] = pymongo_mod
        sys.modules["pymongo.collection"] = collection_mod
        self._injected += ["pymongo", "pymongo.collection"]

        from evalkit.instrumentation.db.pymongo import patch_pymongo
        patch_pymongo(self.client)

        FakeCollection().insert_one({"item": "book"})
        self.client.flush()
        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertTrue(db)
        self.assertEqual(db[0].attributes.get("db.operation"), "insert_one")
        self.assertEqual(db[0].attributes.get("db.sql.table"), "orders")


class TestAsyncSQLAlchemy(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_async_sqlalchemy_engine_db_query(self):
        try:
            from sqlalchemy.ext.asyncio import create_async_engine
        except ImportError:
            self.skipTest("sqlalchemy[asyncio] not installed")

        async def run():
            engine = create_async_engine("sqlite+aiosqlite:///:memory:")
            evalkit.patch_async_sqlalchemy_engine(engine)
            tid, end, ctx = self.client.start_trace("async-req")
            with evalkit.with_trace(ctx):
                async with engine.begin() as conn:
                    from sqlalchemy import text
                    await conn.execute(text("SELECT 1"))
            end("OK")
            await engine.dispose()

        try:
            asyncio.run(run())
        except Exception as exc:
            # aiosqlite may not be installed — skip gracefully
            if "aiosqlite" in str(exc) or "No module" in str(exc):
                self.skipTest("aiosqlite not installed")
            raise

        self.client.flush()
        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertTrue(db, "expected a db_query span from async engine")
        self.assertIn("db.statement", db[0].attributes)

    def test_patch_async_engine_idempotent(self):
        from evalkit.instrumentation.db.sqlalchemy import patch_sqlalchemy_async

        # Simulate an already-instrumented async engine: the guard must bail out
        # without calling patch_sqlalchemy (which requires a real Engine target).
        class FakeAsyncEngine:
            _evalkit_instrumented = True  # pre-set — already patched
            sync_engine = None  # would be None if guard somehow missed

        engine = FakeAsyncEngine()
        # Must not raise even though sync_engine is None
        patch_sqlalchemy_async(self.client, engine)
        self.assertTrue(engine._evalkit_instrumented)


class TestDBNoCrash(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.client = fresh([])

    def test_try_patch_databases_no_crash(self):
        from evalkit.instrumentation.auto import _try_patch_databases
        _try_patch_databases(self.client)


if __name__ == "__main__":
    unittest.main(verbosity=2)
