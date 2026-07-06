import unittest

from tests.helpers import fresh, reset_context


class TestChroma(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def test_query_emits_db_query_span(self):
        from evalkit.instrumentation.vectordb.chroma import _patch_query

        class FakeCollection:
            name = "my-collection"

            def query(self, *, query_embeddings=None, n_results=10, **kwargs):
                return {
                    "ids": [["doc-1", "doc-2"]],
                    "distances": [[0.1, 0.2]],
                    "documents": [["hello", "world"]],
                }

        _patch_query(self.client, FakeCollection)
        FakeCollection().query(query_embeddings=[[0.1, 0.2, 0.3]], n_results=5)
        self.client.flush()

        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 1)
        self.assertEqual(db[0].attributes["db.system"], "chroma")
        self.assertEqual(db[0].attributes["db.operation"], "query")
        self.assertEqual(db[0].attributes["db.vector.collection"], "my-collection")
        self.assertEqual(db[0].attributes["db.vector.top_k"], 5)
        self.assertEqual(db[0].attributes["db.vector.match_count"], 2)
        self.assertIn("gen_ai.response.body", db[0].attributes)
        self.assertNotIn("0.1", db[0].attributes.get("gen_ai.request.body", ""))

    def test_query_error_emits_error_span(self):
        from evalkit.instrumentation.vectordb.chroma import _patch_query

        class FakeCollection:
            name = "my-collection"

            def query(self, *, query_embeddings=None, n_results=10, **kwargs):
                raise RuntimeError("collection not found")

        _patch_query(self.client, FakeCollection)
        with self.assertRaises(RuntimeError):
            FakeCollection().query(query_embeddings=[[0.1]])
        self.client.flush()
        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 1)
        self.assertEqual(db[0].status, "ERROR")

    def test_add_emits_db_query_span(self):
        from evalkit.instrumentation.vectordb.chroma import _patch_add

        class FakeCollection:
            name = "my-collection"

            def add(self, ids, documents=None, **kwargs):
                return None

        _patch_add(self.client, FakeCollection)
        FakeCollection().add(ids=["id1", "id2"], documents=["a", "b"])
        self.client.flush()

        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 1)
        self.assertEqual(db[0].attributes["db.operation"], "add")
        self.assertEqual(db[0].attributes["db.vector.count"], 2)

    def test_upsert_emits_db_query_span(self):
        from evalkit.instrumentation.vectordb.chroma import _patch_upsert

        class FakeCollection:
            name = "my-collection"

            def upsert(self, ids, documents=None, **kwargs):
                return None

        _patch_upsert(self.client, FakeCollection)
        FakeCollection().upsert(ids=["id1"], documents=["a"])
        self.client.flush()

        db = [e for e in self.captured if e.span_type == "db_query"]
        self.assertEqual(len(db), 1)
        self.assertEqual(db[0].attributes["db.operation"], "upsert")
        self.assertEqual(db[0].attributes["db.vector.count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
