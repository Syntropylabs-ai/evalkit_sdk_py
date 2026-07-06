import asyncio
import unittest
import uuid

from weaviate.collections.classes.batch import BatchObjectReturn
from weaviate.collections.classes.internal import MetadataReturn, Object, QueryReturn

from tests.helpers import fresh, reset_context


def _make_object(oid: str, distance: float) -> Object:
    return Object(
        uuid=uuid.uuid4(),
        metadata=MetadataReturn(distance=distance),
        properties={},
        references=None,
        vector={},
        collection="my-collection",
    )


class TestWeaviateSync(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_near_vector_emits_db_query_span(self):
        from evalkit.instrumentation.vectordb.weaviate import _patch_near_vector_sync

        class FakeQuery:
            _name = "my-collection"

            def near_vector(self, near_vector, *, limit=10, **kwargs):
                return QueryReturn(objects=[_make_object("a", 0.1), _make_object("b", 0.2)])

        _patch_near_vector_sync(self.client, FakeQuery)
        FakeQuery().near_vector([0.1, 0.2, 0.3], limit=5)
        self.client.flush()

        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 1)
        self.assertEqual(db[0].attributes["db.system"], "weaviate")
        self.assertEqual(db[0].attributes["db.operation"], "query")
        self.assertEqual(db[0].attributes["db.vector.collection"], "my-collection")
        self.assertEqual(db[0].attributes["db.vector.top_k"], 5)
        self.assertEqual(db[0].attributes["db.vector.match_count"], 2)
        self.assertNotIn("0.1", db[0].attributes.get("gen_ai.request.body", ""))

    def test_near_vector_error_emits_error_span(self):
        from evalkit.instrumentation.vectordb.weaviate import _patch_near_vector_sync

        class FakeQuery:
            _name = "my-collection"

            def near_vector(self, near_vector, *, limit=10, **kwargs):
                raise RuntimeError("boom")

        _patch_near_vector_sync(self.client, FakeQuery)
        with self.assertRaises(RuntimeError):
            FakeQuery().near_vector([0.1])
        self.client.flush()
        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 1)
        self.assertEqual(db[0].status, "ERROR")

    def test_insert_emits_db_query_span(self):
        from evalkit.instrumentation.vectordb.weaviate import _patch_data_sync

        class FakeData:
            _name = "my-collection"

            def insert(self, properties, **kwargs):
                return uuid.uuid4()

            def insert_many(self, objects, **kwargs):
                return BatchObjectReturn()

        _patch_data_sync(self.client, FakeData)
        FakeData().insert({"text": "hello"})
        self.client.flush()

        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 1)
        self.assertEqual(db[0].attributes["db.operation"], "insert")
        self.assertEqual(db[0].attributes["db.vector.count"], 1)

    def test_insert_many_emits_db_query_span(self):
        from evalkit.instrumentation.vectordb.weaviate import _patch_data_sync

        class FakeData:
            _name = "my-collection"

            def insert(self, properties, **kwargs):
                return uuid.uuid4()

            def insert_many(self, objects, **kwargs):
                return BatchObjectReturn()

        _patch_data_sync(self.client, FakeData)
        FakeData().insert_many([{"text": "a"}, {"text": "b"}])
        self.client.flush()

        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 1)
        self.assertEqual(db[0].attributes["db.operation"], "insert_many")
        self.assertEqual(db[0].attributes["db.vector.count"], 2)
        self.assertEqual(db[0].status, "OK")


class TestWeaviateAsync(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_near_vector_async_emits_db_query_span(self):
        from evalkit.instrumentation.vectordb.weaviate import _patch_near_vector_async

        class FakeQueryAsync:
            _name = "my-collection"

            def near_vector(self, near_vector, *, limit=10, **kwargs):
                async def _inner():
                    return QueryReturn(objects=[_make_object("a", 0.1)])
                return _inner()

        _patch_near_vector_async(self.client, FakeQueryAsync)
        asyncio.run(FakeQueryAsync().near_vector([0.1, 0.2], limit=3))
        self.client.flush()

        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 1)
        self.assertEqual(db[0].attributes["db.vector.match_count"], 1)

    def test_insert_async_emits_db_query_span(self):
        from evalkit.instrumentation.vectordb.weaviate import _patch_data_async

        class FakeDataAsync:
            _name = "my-collection"

            def insert(self, properties, **kwargs):
                async def _inner():
                    return uuid.uuid4()
                return _inner()

            def insert_many(self, objects, **kwargs):
                async def _inner():
                    return BatchObjectReturn()
                return _inner()

        _patch_data_async(self.client, FakeDataAsync)
        asyncio.run(FakeDataAsync().insert({"text": "hello"}))
        self.client.flush()

        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 1)
        self.assertEqual(db[0].attributes["db.operation"], "insert")


if __name__ == "__main__":
    unittest.main(verbosity=2)
