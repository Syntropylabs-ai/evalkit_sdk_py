import unittest

from qdrant_client.http.models.models import QueryResponse, ScoredPoint, UpdateResult

from tests.helpers import fresh, reset_context


class TestQdrant(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_query_points_emits_db_query_span(self):
        from evalkit.instrumentation.vectordb.qdrant import _patch_query

        class FakeQdrantClient:
            def query_points(self, collection_name, *, query=None, limit=10, **kwargs):
                return QueryResponse(points=[
                    ScoredPoint(id="doc-1", version=1, score=0.95),
                    ScoredPoint(id="doc-2", version=1, score=0.80),
                ])

        _patch_query(self.client, FakeQdrantClient)
        FakeQdrantClient().query_points("my-collection", query=[0.1, 0.2, 0.3], limit=5)
        self.client.flush()

        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 1)
        self.assertEqual(db[0].attributes["db.system"], "qdrant")
        self.assertEqual(db[0].attributes["db.operation"], "query")
        self.assertEqual(db[0].attributes["db.vector.collection"], "my-collection")
        self.assertEqual(db[0].attributes["db.vector.top_k"], 5)
        self.assertEqual(db[0].attributes["db.vector.match_count"], 2)
        self.assertIn("gen_ai.response.body", db[0].attributes)
        self.assertNotIn("0.1", db[0].attributes.get("gen_ai.request.body", ""))

    def test_query_error_emits_error_span(self):
        from evalkit.instrumentation.vectordb.qdrant import _patch_query

        class FakeQdrantClient:
            def query_points(self, collection_name, *, query=None, limit=10, **kwargs):
                raise RuntimeError("collection not found")

        _patch_query(self.client, FakeQdrantClient)
        with self.assertRaises(RuntimeError):
            FakeQdrantClient().query_points("missing", query=[0.1])
        self.client.flush()
        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 1)
        self.assertEqual(db[0].status, "ERROR")

    def test_upsert_emits_db_query_span(self):
        from evalkit.instrumentation.vectordb.qdrant import _patch_upsert

        class FakeQdrantClient:
            def upsert(self, collection_name, points, **kwargs):
                return UpdateResult(status="completed")

        _patch_upsert(self.client, FakeQdrantClient)
        FakeQdrantClient().upsert("my-collection", [{"id": "p1", "vector": [0.1, 0.2]}, {"id": "p2", "vector": [0.3, 0.4]}])
        self.client.flush()

        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 1)
        self.assertEqual(db[0].attributes["db.operation"], "upsert")
        self.assertEqual(db[0].attributes["db.vector.count"], 2)
        self.assertEqual(db[0].attributes["db.vector.status"], "completed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
