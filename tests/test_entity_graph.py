import unittest

from src.entity_graph import EntityGraphStore, GraphHistoryRetriever
from src.history_retriever import HistoryRetriever
from src.message_store import MessageStore


class EntityGraphTests(unittest.TestCase):
    def setUp(self):
        self.messages = MessageStore(":memory:")
        self.addCleanup(self.messages.close)
        self.graph = EntityGraphStore(self.messages)
        for user_id in ("u1", "u2"):
            for index, content in enumerate(("Alice works with Bob", "Bob lives in Hangzhou"), 1):
                self.messages.append_message(
                    user_id=user_id, session_id="s", message_id=f"m{index}",
                    occurred_at="2026-01-01T00:00:00+00:00",
                    payload={"role": "user", "content": content},
                )

    def source(self, message_id, user_id="u1"):
        return {"user_id": user_id, "session_id": "s", "message_id": message_id}

    def seed_graph(self, user_id="u1"):
        alice = self.graph.add_entity(
            user_id=user_id, entity_id="alice", entity_type="person",
            canonical_name="Alice", aliases=["ＡＬＩＣＥ"], attributes={},
        )
        bob = self.graph.add_entity(
            user_id=user_id, entity_id="bob", entity_type="person",
            canonical_name="Bob", aliases=["Robert"], attributes={},
        )
        city = self.graph.add_entity(
            user_id=user_id, entity_id="hz", entity_type="city",
            canonical_name="杭州", aliases=["Hangzhou"], attributes={},
        )
        self.graph.add_relation(
            user_id=user_id, relation_id="r1", subject_entity_id="alice",
            predicate="works_with", object_entity_id="bob", sources=[self.source("m1", user_id)],
        )
        self.graph.add_relation(
            user_id=user_id, relation_id="r2", subject_entity_id="bob",
            predicate="lives_in", object_entity_id="hz", sources=[self.source("m2", user_id)],
        )
        return alice, bob, city

    def test_alias_resolution_is_normalized_but_ambiguity_is_preserved(self):
        self.seed_graph()
        resolved = self.graph.resolve_entity(user_id="u1", name="  Ａlice ")
        self.assertEqual([item["entity_id"] for item in resolved], ["alice"])
        self.graph.add_entity(
            user_id="u1", entity_id="other-alice", entity_type="person",
            canonical_name="Other", aliases=["Alice"], attributes={},
        )
        self.assertEqual(
            [item["entity_id"] for item in self.graph.resolve_entity(user_id="u1", name="alice")],
            ["alice", "other-alice"],
        )

    def test_merge_redirects_alias_resolution_without_deleting_history(self):
        self.seed_graph()
        self.graph.add_entity(
            user_id="u1", entity_id="robert", entity_type="person",
            canonical_name="Robert Smith", aliases=["Bobby"], attributes={},
        )
        merged = self.graph.merge_entity(
            user_id="u1", source_entity_id="robert", target_entity_id="bob",
        )
        self.assertEqual(merged["status"], "merged")
        self.assertEqual(
            [item["entity_id"] for item in self.graph.resolve_entity(user_id="u1", name="Bobby")],
            ["bob"],
        )

    def test_bounded_expansion_handles_cycles_time_and_user_isolation(self):
        self.seed_graph()
        self.graph.add_relation(
            user_id="u1", relation_id="cycle", subject_entity_id="hz",
            predicate="mentions", object_entity_id="alice", sources=[self.source("m2")],
        )
        depth_one = self.graph.expand(
            user_id="u1", entity_ids=["alice"], max_depth=1, max_nodes=10,
        )
        self.assertEqual([item["entity_id"] for item in depth_one["entities"]], ["alice", "bob", "hz"])
        bounded = self.graph.expand(
            user_id="u1", entity_ids=["alice"], max_depth=5, max_nodes=2,
        )
        self.assertEqual(len(bounded["entities"]), 2)
        self.assertEqual(self.graph.resolve_entity(user_id="u2", name="Alice"), [])

    def test_graph_retrieval_can_be_compared_with_plain_history_retrieval(self):
        self.seed_graph()
        retriever = GraphHistoryRetriever(HistoryRetriever(self.messages), self.graph)
        paired = retriever.compare(user_id="u1", query="Alice 的居住关系", top_k=5)
        plain_ids = [hit["message"]["message_id"] for hit in paired["without_graph"]]
        graph_ids = [hit["message"]["message_id"] for hit in paired["with_graph"]]
        self.assertEqual(plain_ids, ["m1"])
        self.assertEqual(graph_ids, ["m1", "m2"])
        self.assertEqual(paired["with_graph"][1]["via"], "graph")

    def test_relation_requires_real_same_user_sources_and_id_retries_first_body(self):
        self.seed_graph()
        first = self.graph.get_relation(user_id="u1", relation_id="r1")
        self.assertEqual(self.graph.add_relation(
            user_id="u1", relation_id="r1", subject_entity_id=None, predicate=None,
            object_entity_id=None, sources=None,
        ), first)
        with self.assertRaises(ValueError):
            self.graph.add_relation(
                user_id="u1", relation_id="bad", subject_entity_id="alice",
                predicate="knows", object_entity_id="bob", sources=[self.source("m1", "u2")],
            )
        self.assertIsNone(self.graph.get_relation(user_id="u1", relation_id="bad"))

    def test_chained_merges_redirect_old_aliases_and_edges_without_rewriting_history(self):
        self.seed_graph()
        for entity_id, name in (("robert", "Robert Smith"), ("final", "Robert Final")):
            self.graph.add_entity(user_id="u1", entity_id=entity_id, entity_type="person", canonical_name=name)
        original = self.graph.get_relation(user_id="u1", relation_id="r1")
        first_merge = self.graph.merge_entity(user_id="u1", source_entity_id="bob", target_entity_id="robert")
        self.graph.merge_entity(user_id="u1", source_entity_id="robert", target_entity_id="final")
        self.assertEqual(self.graph.merge_entity(user_id="u1", source_entity_id="bob", target_entity_id="robert"), first_merge)
        self.assertEqual([item["entity_id"] for item in self.graph.resolve_entity(user_id="u1", name="Bob")], ["final"])
        self.assertEqual([item["entity_id"] for item in self.graph.find_query_entities(user_id="u1", query="Bob 的经历")], ["final"])
        graph = self.graph.expand(user_id="u1", entity_ids=["alice"], max_depth=2)
        self.assertEqual([item["entity_id"] for item in graph["entities"]], ["alice", "final", "hz"])
        by_id = {relation["relation_id"]: relation for relation in graph["relations"]}
        self.assertEqual(by_id["r1"]["object_entity_id"], "final")
        self.assertEqual(by_id["r2"]["subject_entity_id"], "final")
        self.assertEqual(self.graph.get_relation(user_id="u1", relation_id="r1"), original)
        self.assertEqual(self.messages.conn.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_bounded_expansion_returns_only_edges_whose_endpoints_were_admitted(self):
        self.seed_graph()
        one = self.graph.expand(user_id="u1", entity_ids=["alice"], max_nodes=1)
        self.assertEqual(one["relations"], [])
        graph = self.graph.expand(user_id="u1", entity_ids=["alice"], max_nodes=2, max_depth=4)
        ids = {entity["entity_id"] for entity in graph["entities"]}
        self.assertEqual(ids, {"alice", "bob"})
        self.assertEqual([relation["relation_id"] for relation in graph["relations"]], ["r1"])
        for relation in graph["relations"]:
            self.assertTrue({relation["subject_entity_id"], relation["object_entity_id"]} <= ids)

    def test_blank_query_preserves_baseline_empty_result(self):
        self.seed_graph()
        retriever = GraphHistoryRetriever(HistoryRetriever(self.messages), self.graph)
        for query in ("", "  \n\t"):
            with self.subTest(query=query):
                self.assertEqual(retriever.search(user_id="u1", query=query), [])

    def test_ignored_late_alias_rolls_back_the_entity_and_first_alias(self):
        with self.messages.conn:
            self.messages.conn.execute("""
                CREATE TRIGGER ignore_alias BEFORE INSERT ON entity_aliases
                WHEN NEW.normalized_alias = 'later' BEGIN SELECT RAISE(IGNORE); END
            """)
        with self.assertRaises(ValueError):
            self.graph.add_entity(user_id="u1", entity_id="new", entity_type="person",
                                  canonical_name="New", aliases=["Later"])
        self.assertIsNone(self.graph.get_entity(user_id="u1", entity_id="new"))
        self.assertEqual(self.graph.resolve_entity(user_id="u1", name="New"), [])

    def test_ignored_late_relation_source_rolls_back_relation_and_first_source(self):
        self.seed_graph()
        with self.messages.conn:
            self.messages.conn.execute("""
                CREATE TRIGGER ignore_relation_source BEFORE INSERT ON entity_relation_sources
                WHEN NEW.relation_id = 'new' AND NEW.source_message_id = 'm2'
                BEGIN SELECT RAISE(IGNORE); END
            """)
        with self.assertRaises(ValueError):
            self.graph.add_relation(user_id="u1", relation_id="new", subject_entity_id="alice",
                                    predicate="knows", object_entity_id="bob",
                                    sources=[self.source("m1"), self.source("m2")])
        self.assertIsNone(self.graph.get_relation(user_id="u1", relation_id="new"))
        self.assertEqual(self.messages.conn.execute("SELECT COUNT(*) FROM entity_relation_sources").fetchone()[0], 2)


if __name__ == "__main__":
    unittest.main()
