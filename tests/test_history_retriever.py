"""第三关：按约定验证原始历史的关键词检索行为。"""

import tempfile
import unittest
from pathlib import Path

from src.history_retriever import HistoryRetriever
from src.message_store import MessageStore


class HistoryRetrieverTests(unittest.TestCase):
    def setUp(self):
        project_root = Path(__file__).resolve().parents[1]
        temp_dir = tempfile.TemporaryDirectory(
            prefix=".history-retriever-test-", dir=project_root,
        )
        self.assertTrue(Path(temp_dir.name).resolve().is_relative_to(project_root))
        self.addCleanup(temp_dir.cleanup)
        self.store = MessageStore(Path(temp_dir.name) / "messages.sqlite")
        self.addCleanup(self.store.close)
        self.retriever = HistoryRetriever(self.store)

    def _append(self, message_id, payload, *, user_id="u1", session_id="s1"):
        return self.store.append_message(
            user_id=user_id,
            session_id=session_id,
            message_id=message_id,
            occurred_at="2026-10-20T10:00:00+08:00",
            payload=payload,
        )

    def test_empty_store_and_unknown_user_return_empty(self):
        self.assertEqual(
            self.retriever.search(user_id="u1", query="咖啡"), [],
        )
        self._append("m1", {"role": "user", "content": "喜欢咖啡"})
        self.assertEqual(
            self.retriever.search(user_id="unknown", query="咖啡"), [],
        )

    def test_search_crosses_sessions_but_never_crosses_users(self):
        first = self._append(
            "m1", {"role": "user", "content": "咖啡"}, session_id="s1",
        )
        other = self._append(
            "m1", {"role": "user", "content": "早餐咖啡"}, user_id="u2",
        )
        second = self._append(
            "m1", {"role": "assistant", "content": "早餐咖啡"},
            session_id="s2",
        )
        self._append(
            "m1", {"role": "user", "content": "早餐咖啡"}, user_id="u10",
        )

        self.assertEqual(
            self.retriever.search(user_id="u1", query="早餐 咖啡", top_k=2),
            [{"message": second, "score": 2}, {"message": first, "score": 1}],
        )
        self.assertEqual(
            self.retriever.search(user_id="u2", query="早餐 咖啡"),
            [{"message": other, "score": 2}],
        )

    def test_scores_count_distinct_query_terms_not_occurrences(self):
        repeated = self._append(
            "m1", {"role": "user", "content": "咖啡咖啡咖啡"},
        )
        both = self._append(
            "m2", {"role": "user", "content": "早餐喜欢喝咖啡"},
        )
        self._append("m3", {"role": "user", "content": "晚上去跑步"})

        self.assertEqual(
            self.retriever.search(user_id="u1", query="早餐 咖啡 咖啡"),
            [{"message": both, "score": 2}, {"message": repeated, "score": 1}],
        )
        self.assertEqual(
            self.retriever.search(user_id="u1", query="早餐 咖啡", top_k=1),
            [{"message": both, "score": 2}],
        )

    def test_query_splits_on_whitespace_without_creating_empty_terms(self):
        both = self._append(
            "m1", {"role": "user", "content": "早餐喝咖啡"},
        )
        one = self._append("m2", {"role": "user", "content": "咖啡"})
        self._append("m3", {"role": "user", "content": "跑步"})

        self.assertEqual(
            self.retriever.search(
                user_id="u1", query="\t 早餐  \n咖啡\r\n\u3000咖啡 ",
            ),
            [{"message": both, "score": 2}, {"message": one, "score": 1}],
        )

    def test_matching_casefolds_both_query_and_content(self):
        mixed = self._append(
            "m1", {"role": "user", "content": "Straße COFFEE"},
        )
        lower = self._append(
            "m2", {"role": "user", "content": "strasse coffee"},
        )

        expected = [{"message": mixed, "score": 2}, {"message": lower, "score": 2}]
        for query in ("STRASSE CoFfEe", "Straße coffee", "Straße STRASSE coffee COFFEE"):
            with self.subTest(query=query):
                self.assertEqual(
                    self.retriever.search(user_id="u1", query=query), expected,
                )

    def test_ties_preserve_session_and_sequence_order(self):
        # 写入顺序与会话排序不同，ID 排序也与会话内 seq 顺序不同。
        later_session = self._append(
            "a", {"role": "user", "content": "咖啡"}, session_id="s2",
        )
        first = self._append("z", {"role": "user", "content": "咖啡"})
        second = self._append("a", {"role": "user", "content": "咖啡"})

        self.assertEqual(
            self.retriever.search(user_id="u1", query="咖啡"),
            [
                {"message": first, "score": 1},
                {"message": second, "score": 1},
                {"message": later_session, "score": 1},
            ],
        )

    def test_default_and_explicit_top_k_limit_results(self):
        records = [
            self._append(f"m{index}", {"role": "user", "content": "咖啡"})
            for index in range(7)
        ]
        expected = [{"message": record, "score": 1} for record in records]

        self.assertEqual(
            self.retriever.search(user_id="u1", query="咖啡"), expected[:5],
        )
        self.assertEqual(
            self.retriever.search(user_id="u1", query="咖啡", top_k=2),
            expected[:2],
        )
        self.assertEqual(
            self.retriever.search(user_id="u1", query="咖啡", top_k=20),
            expected,
        )

    def test_empty_or_whitespace_query_returns_empty(self):
        self._append("m1", {"role": "user", "content": "咖啡"})
        for query in ("", " ", "\t\r\n", "\u3000"):
            with self.subTest(query=query):
                self.assertEqual(
                    self.retriever.search(user_id="u1", query=query), [],
                )

    def test_zero_top_k_returns_empty(self):
        self._append("m1", {"role": "user", "content": "咖啡"})
        self.assertEqual(
            self.retriever.search(user_id="u1", query="咖啡", top_k=0), [],
        )

    def test_invalid_query_raises_value_error(self):
        for query in (None, 123, True, ["咖啡"], {"query": "咖啡"}, b"coffee"):
            with self.subTest(query=query):
                with self.assertRaises(ValueError):
                    self.retriever.search(user_id="u1", query=query)

    def test_invalid_top_k_raises_value_error_and_rejects_booleans(self):
        for top_k in (-1, 1.5, "2", None, True, False):
            with self.subTest(top_k=top_k):
                with self.assertRaises(ValueError):
                    self.retriever.search(user_id="u1", query="咖啡", top_k=top_k)

    def test_user_id_is_validated_even_for_empty_queries_or_zero_top_k(self):
        for user_id in (None, "", " ", "\t", 123, True):
            for query, top_k in (("咖啡", 5), ("", 5), ("咖啡", 0)):
                with self.subTest(user_id=user_id, query=query, top_k=top_k):
                    with self.assertRaises(ValueError):
                        self.retriever.search(
                            user_id=user_id, query=query, top_k=top_k,
                        )

    def test_missing_and_non_string_content_are_skipped(self):
        payloads = [
            {"role": "assistant", "content": None},
            {"role": "user", "content": [{"type": "text", "text": "咖啡"}]},
            {"role": "tool", "content": {"text": "咖啡"}},
            {"role": "user", "content": 123},
            {"role": "user", "content": True},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "arguments": "咖啡"}],
            },
        ]
        for index, payload in enumerate(payloads):
            self._append(f"skip-{index}", payload)
        valid = self._append("valid", {"role": "user", "content": "只有咖啡"})

        self.assertEqual(
            self.retriever.search(user_id="u1", query="咖啡 123 true none"),
            [{"message": valid, "score": 1}],
        )

    def test_no_match_returns_empty(self):
        self._append("m1", {"role": "user", "content": "早餐喝咖啡"})
        self.assertEqual(
            self.retriever.search(user_id="u1", query="游泳"), [],
        )

    def test_results_preserve_complete_evidence_and_do_not_modify_history(self):
        saved = self._append(
            "tool-result",
            {
                "role": "tool",
                "content": "已查到咖啡信息",
                "tool_call_id": "call_1",
                "metadata": {"tags": ["原始"]},
            },
        )
        before = self.store.list_user_messages(user_id="u1")
        expected = [{"message": saved, "score": 1}]
        hits = self.retriever.search(user_id="u1", query="咖啡")

        self.assertEqual(hits, expected)
        self.assertNotIn("score", hits[0]["message"])
        self.assertEqual(self.store.list_user_messages(user_id="u1"), before)

        hits[0]["score"] = 99
        hits[0]["message"]["payload"]["content"] = "修改返回内容"
        hits[0]["message"]["payload"]["metadata"]["tags"].append("修改标签")
        hits.clear()

        self.assertEqual(self.store.list_user_messages(user_id="u1"), before)
        self.assertEqual(
            self.retriever.search(user_id="u1", query="咖啡"), expected,
        )
