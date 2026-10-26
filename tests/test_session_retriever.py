"""Tests for the session-first retriever used by the LoCoMo benchmark."""

import unittest

from src.hybrid_retriever import HybridRetriever
from src.message_store import MessageStore
from src.session_retriever import SessionFirstRetriever


class SessionFirstRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.store = MessageStore(":memory:")
        self.addCleanup(self.store.close)
        for session, message, content in (
            ("s1", "a1", "I like tea."),
            ("s1", "a2", "Tea is warm."),
            ("s1", "a3", "Nothing relevant here."),
            ("s2", "b1", "My bicycle is red."),
            ("s2", "b2", "It cost 200 dollars."),
            ("s2", "b3", "I ride it daily."),
            ("s3", "c1", "The weather is nice."),
            ("s3", "c2", "I went hiking."),
        ):
            self.store.append_message(
                user_id="u1", session_id=session, message_id=message,
                occurred_at="2024-01-01T00:00:00+00:00",
                payload={"role": "user", "content": content},
            )
        self.store.append_message(
            user_id="u1", session_id="s4", message_id="d1",
            occurred_at="2024-06-01T00:00:00+00:00",
            payload={"role": "user", "content": "My bicycle is blue."},
        )
        self.retriever = SessionFirstRetriever(self.store)

    def test_best_session_contributes_all_its_turns(self):
        hits = self.retriever.search(
            user_id="u1", query="bicycle red",
            as_of="2024-01-02T00:00:00+00:00",
        )
        self.assertEqual(
            [hit["source"]["message_id"] for hit in hits], ["b1", "b2", "b3"]
        )
        self.assertTrue(all(hit["source"]["session_id"] == "s2" for hit in hits))
        self.assertGreater(hits[0]["score"], 0)
        self.assertEqual(hits[1]["score"], 0)
        self.assertEqual(hits[0]["session_score"], hits[0]["score"])

    def test_session_id_limits_candidates_to_one_session(self):
        hits = self.retriever.search(user_id="u1", query="bicycle", session_id="s1")
        self.assertEqual(hits, [])
        hits = self.retriever.search(user_id="u1", query="tea", session_id="s1")
        self.assertEqual([hit["source"]["message_id"] for hit in hits], ["a1", "a2", "a3"])

    def test_as_of_excludes_later_messages(self):
        early = self.retriever.search(
            user_id="u1", query="blue", as_of="2024-01-02T00:00:00+00:00"
        )
        late = self.retriever.search(
            user_id="u1", query="blue", as_of="2024-07-01T00:00:00+00:00"
        )
        self.assertEqual(early, [])
        self.assertEqual([hit["source"]["message_id"] for hit in late], ["d1"])

    def test_turns_per_session_with_neighbors_keeps_adjacent_turn(self):
        retriever = SessionFirstRetriever(
            self.store, turns_per_session=1, neighbors=True
        )
        hits = retriever.search(
            user_id="u1", query="bicycle red",
            as_of="2024-01-02T00:00:00+00:00",
        )
        self.assertEqual(
            [hit["source"]["message_id"] for hit in hits], ["b1", "b2"]
        )

    def test_turns_per_session_without_neighbors_keeps_only_best_turn(self):
        retriever = SessionFirstRetriever(self.store, turns_per_session=1)
        hits = retriever.search(
            user_id="u1", query="bicycle red",
            as_of="2024-01-02T00:00:00+00:00",
        )
        self.assertEqual([hit["source"]["message_id"] for hit in hits], ["b1"])

    def test_top_k_caps_flat_output(self):
        hits = self.retriever.search(
            user_id="u1", query="bicycle red", top_k=1,
            as_of="2024-01-02T00:00:00+00:00",
        )
        self.assertEqual([hit["source"]["message_id"] for hit in hits], ["b1"])
        self.assertEqual(
            self.retriever.search(user_id="u1", query="bicycle", top_k=0), []
        )

    def test_empty_query_and_unknown_user_return_empty(self):
        self.assertEqual(self.retriever.search(user_id="u1", query="   "), [])
        self.assertEqual(self.retriever.search(user_id="nobody", query="bicycle"), [])

    def test_hit_shape_carries_source_and_metadata(self):
        hits = self.retriever.search(
            user_id="u1", query="bicycle red",
            as_of="2024-01-02T00:00:00+00:00",
        )
        self.assertEqual(
            set(hits[0]),
            {"type", "text", "source", "metadata", "score", "session_score"},
        )
        self.assertEqual(hits[0]["type"], "message")
        self.assertEqual(hits[0]["metadata"]["role"], "user")
        self.assertEqual(
            hits[0]["metadata"]["occurred_at"], "2024-01-01T00:00:00+00:00"
        )
        self.assertEqual(
            hits[0]["source"],
            {"type": "message", "user_id": "u1", "session_id": "s2", "message_id": "b1"},
        )

    def test_scores_match_lexical_retriever_for_scored_messages(self):
        session_hits = self.retriever.search(
            user_id="u1", query="bicycle red",
            as_of="2024-01-02T00:00:00+00:00",
        )
        session_scores = {
            (hit["source"]["session_id"], hit["source"]["message_id"]): hit["score"]
            for hit in session_hits
        }
        lexical_hits = HybridRetriever(self.store).search(
            user_id="u1", query="bicycle red",
            top_k=len(self.store.list_user_messages(user_id="u1")) + 1,
            as_of="2024-01-02T00:00:00+00:00",
        )
        self.assertTrue(lexical_hits)
        for hit in lexical_hits:
            key = (hit["source"]["session_id"], hit["source"]["message_id"])
            self.assertIn(key, session_scores)
            self.assertAlmostEqual(hit["score"], session_scores[key])

    def test_repeated_search_is_deterministic(self):
        first = self.retriever.search(
            user_id="u1", query="bicycle red",
            as_of="2024-01-02T00:00:00+00:00",
        )
        self.assertEqual(
            first,
            self.retriever.search(
                user_id="u1", query="bicycle red",
                as_of="2024-01-02T00:00:00+00:00",
            ),
        )

    def test_invalid_configuration_and_arguments_raise(self):
        for kwargs in (
            {"session_count": 0},
            {"session_count": True},
            {"session_count": 1.5},
            {"turns_per_session": 0},
            {"turns_per_session": True},
            {"neighbors": "yes"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SessionFirstRetriever(self.store, **kwargs)
        with self.assertRaises(ValueError):
            self.retriever.search(user_id="", query="bicycle")
        with self.assertRaises(ValueError):
            self.retriever.search(user_id="u1", query=123)
        with self.assertRaises(ValueError):
            self.retriever.search(user_id="u1", query="bicycle", top_k=True)
        with self.assertRaises(ValueError):
            self.retriever.search(
                user_id="u1", query="bicycle", as_of="2024-01-01T00:00:00"
            )


if __name__ == "__main__":
    unittest.main()
