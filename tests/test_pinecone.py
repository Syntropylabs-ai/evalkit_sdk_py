import unittest

from pinecone.models.vectors.responses import QueryResponse, UpsertResponse
from pinecone.models.vectors.vector import ScoredVector

from tests.helpers import fresh, reset_context


class TestPinecone(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_query_emits_db_query_span(self):
        from evalkit.instrumentation.vectordb.pinecone import _patch_query

        class FakeIndex:
            def query(self, *, top_k, vector=None, namespace="", **kwargs):
                return QueryResponse(
                    matches=[ScoredVector(id="doc-1", score=0.95), ScoredVector(id="doc-2", score=0.82)],
                    namespace=namespace,
                )

        _patch_query(self.client, FakeIndex)
        FakeIndex().query(top_k=5, vector=[0.1, 0.2, 0.3], namespace="my-ns")
        self.client.flush()

        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 1)
        self.assertEqual(db[0].attributes["db.system"], "pinecone")
        self.assertEqual(db[0].attributes["db.operation"], "query")
        self.assertEqual(db[0].attributes["db.vector.namespace"], "my-ns")
        self.assertEqual(db[0].attributes["db.vector.top_k"], 5)
        self.assertEqual(db[0].attributes["db.vector.match_count"], 2)
        self.assertIn("gen_ai.response.body", db[0].attributes)
        # Raw vector floats must never be captured.
        self.assertNotIn("0.1", db[0].attributes.get("gen_ai.request.body", ""))

    def test_query_error_emits_error_span(self):
        from evalkit.instrumentation.vectordb.pinecone import _patch_query

        class FakeIndex:
            def query(self, *, top_k, vector=None, namespace="", **kwargs):
                raise RuntimeError("index not found")

        _patch_query(self.client, FakeIndex)
        with self.assertRaises(RuntimeError):
            FakeIndex().query(top_k=5, vector=[0.1])
        self.client.flush()
        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 1)
        self.assertEqual(db[0].status, "ERROR")

    def test_upsert_emits_db_query_span(self):
        from evalkit.instrumentation.vectordb.pinecone import _patch_upsert

        class FakeIndex:
            def upsert(self, *, vectors, namespace="", **kwargs):
                return UpsertResponse(upserted_count=len(vectors))

        _patch_upsert(self.client, FakeIndex)
        FakeIndex().upsert(vectors=[("v1", [0.1, 0.2]), ("v2", [0.3, 0.4])], namespace="my-ns")
        self.client.flush()

        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 1)
        self.assertEqual(db[0].attributes["db.operation"], "upsert")
        self.assertEqual(db[0].attributes["db.vector.count"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
