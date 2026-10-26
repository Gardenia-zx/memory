"""Document lifecycle and retrieval functionality, without semantic/latency claims."""

from copy import deepcopy
import sqlite3
import unittest

from src.external_document_store import ExternalDocumentStore
from src.forgetter import MemoryForgetter
from src.hybrid_retriever import HashEmbedding, HybridRetriever
from src.message_store import MessageStore


class ExternalDocumentTests(unittest.TestCase):
    def setUp(self):
        self.messages = MessageStore(":memory:")
        self.addCleanup(self.messages.close)
        self.documents = ExternalDocumentStore(self.messages)

    def add(self, **changes):
        return self.documents.add_document(**{
            "user_id": "u1", "document_id": "d1", "content": "北京杭州咖啡茶",
            "chunk_size": 4, "overlap": 1, "metadata": {"title": "资料"}, **changes})

    def test_stable_chunks_offsets_metadata_and_idempotent_body_free_retry(self):
        first = self.add()
        self.assertEqual([(chunk["start_offset"], chunk["end_offset"], chunk["text"]) for chunk in first["chunks"]],
                         [(0, 4, "北京杭州"), (3, 7, "州咖啡茶")])
        self.assertEqual(self.documents.add_document(user_id="u1", document_id="d1"), first)
        self.assertEqual(self.add(content=None, chunk_size=False), first)
        self.assertEqual(self.documents.get_chunk(user_id="u1", document_id="d1", version=1, chunk_id="chunk:000001"), first["chunks"][1])
        first["metadata"]["title"] = "changed"
        self.assertEqual(self.documents.get_document(user_id="u1", document_id="d1")["metadata"]["title"], "资料")

    def test_versioning_retraction_and_users_do_not_revive_old_versions(self):
        first = self.add()
        second = self.add(version=2, content="新资料")
        self.assertEqual(self.documents.get_document(user_id="u1", document_id="d1")["status"], "superseded")
        self.assertEqual({chunk["version"] for chunk in self.documents.list_chunks(user_id="u1")}, {2})
        other = self.add(user_id="u2")
        self.assertEqual(other["status"], "active")
        retracted = self.documents.retract_document(user_id="u1", document_id="d1", expected_version=2)
        self.assertEqual(self.documents.retract_document(user_id="u1", document_id="d1", expected_version=2), retracted)
        self.assertEqual(self.documents.list_chunks(user_id="u1"), [])
        with self.assertRaises(ValueError):
            self.add(version=3)
        self.assertEqual(self.add(), {**first, "status": "superseded"})

    def test_scopes_and_half_open_times_filter_chunks(self):
        self.add(scope="session", session_id="s1", valid_from="2024-01-01T08:00:00+08:00", valid_to="2025-01-01T00:00:00+00:00")
        self.assertEqual(self.documents.list_chunks(user_id="u1", as_of="2024-02-01T00:00:00+00:00"), [])
        self.assertTrue(self.documents.list_chunks(user_id="u1", session_id="s1", as_of="2024-01-01T00:00:00+00:00"))
        self.assertEqual(self.documents.list_chunks(user_id="u1", session_id="s1", as_of="2025-01-01T00:00:00+00:00"), [])
        self.assertEqual(self.documents.list_chunks(user_id="u1", session_id="s2", as_of="2024-02-01T00:00:00+00:00"), [])

    def test_invalid_new_input_and_chunk_failure_roll_back_version_change(self):
        first = self.add()
        for changes in ({"version": True}, {"version": 3}, {"version": 2, "overlap": 4},
                        {"version": 2, "metadata": {"nan": float("nan")}},
                        {"version": 2, "valid_from": "2024-01-01"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.add(**changes)
        with self.messages.conn:
            self.messages.conn.execute("""CREATE TRIGGER fail_chunk BEFORE INSERT ON external_document_chunks
                WHEN NEW.version=2 AND NEW.chunk_id='chunk:000001'
                BEGIN SELECT RAISE(ABORT,'chunk failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.add(version=2)
        self.assertEqual(self.documents.get_latest_document(user_id="u1", document_id="d1"), first)

    def test_forget_cleans_documents_and_blocks_new_writes(self):
        self.add()
        self.add(user_id="u2")
        receipt = MemoryForgetter(self.messages).forget_user(user_id="u1", reason="test")
        self.assertEqual(receipt["deleted_rows"]["external_documents"], 1)
        self.assertEqual(receipt["deleted_rows"]["external_document_chunks"], 2)
        with self.assertRaises(ValueError):
            self.add()
        self.assertTrue(self.documents.list_chunks(user_id="u2"))


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.messages = MessageStore(":memory:")
        self.addCleanup(self.messages.close)
        self.documents = ExternalDocumentStore(self.messages)
        for user, session, message, content in (("u1", "s1", "m1", "我在杭州工作"), ("u1", "s2", "m2", "咖啡"), ("u2", "s1", "m1", "杭州秘密")):
            self.messages.append_message(user_id=user, session_id=session, message_id=message,
                                         occurred_at="2024-01-01T00:00:00+00:00", payload={"role": "user", "content": content})
        self.documents.add_document(user_id="u1", document_id="d1", content="杭州旅行资料", metadata={"title": "旅行"})
        self.retriever = HybridRetriever(self.messages, self.documents, embedding=HashEmbedding())

    def test_chinese_lexical_and_hybrid_results_keep_typed_evidence(self):
        for mode in ("lexical", "vector", "hybrid"):
            with self.subTest(mode=mode):
                hits = self.retriever.search(user_id="u1", query="杭州", mode=mode)
                self.assertEqual({hit["type"] for hit in hits}, {"message", "document_chunk"})
                self.assertTrue(all(hit["source"]["user_id"] == "u1" for hit in hits))
                for hit in hits:
                    if hit["type"] == "document_chunk":
                        self.assertNotIn("message_id", hit["source"])
                        self.assertEqual(hit["metadata"]["title"], "旅行")

    def test_session_time_empty_queries_and_stable_order(self):
        first = self.retriever.search(user_id="u1", query="杭州", session_id="s2")
        self.assertEqual([hit["type"] for hit in first], ["document_chunk"])
        self.assertEqual(first, self.retriever.search(user_id="u1", query="杭州", session_id="s2"))
        self.assertEqual(self.retriever.search(user_id="u1", query="杭州", as_of="2023-01-01T00:00:00+00:00", top_k=0), [])
        self.assertEqual(self.retriever.search(user_id="u1", query="   "), [])
        self.assertEqual(self.retriever.search(user_id="u1", query="unrelated-xyz"), [])

    def test_embedding_validation_and_explicit_vector_configuration(self):
        for embedding in (lambda text: [], lambda text: [True], lambda text: [float("nan")]):
            with self.subTest(embedding=embedding), self.assertRaises(ValueError):
                HybridRetriever(self.messages, embedding=embedding).search(user_id="u1", query="杭州", mode="vector")
        with self.assertRaises(ValueError):
            HybridRetriever(self.messages).search(user_id="u1", query="杭州", mode="hybrid")
        with self.assertRaises(ValueError):
            HybridRetriever(self.messages, embedding=lambda text: [1] if text == "query" else [1, 2]).search(user_id="u1", query="query", mode="vector")
        self.assertEqual(HashEmbedding()("杭州"), HashEmbedding()("杭州"))

    def test_retracted_documents_and_forgotten_users_are_not_retrieved(self):
        self.documents.retract_document(user_id="u1", document_id="d1", expected_version=1)
        self.assertEqual([hit["type"] for hit in self.retriever.search(user_id="u1", query="杭州")], ["message"])
        MemoryForgetter(self.messages).forget_user(user_id="u1", reason="test")
        with self.assertRaises(ValueError):
            self.retriever.search(user_id="u1", query="杭州")


if __name__ == "__main__":
    unittest.main()
