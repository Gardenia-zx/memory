"""验证来源检索评测的指标、隔离和报告，不依赖模型或网络。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call

from src.eval.retrieval_evaluator import evaluate_retrieval
from src.history_retriever import HistoryRetriever
from src.message_store import MessageStore


class RetrievalEvaluatorTests(unittest.TestCase):
    @staticmethod
    def _source(message_id, *, user_id="u1", session_id="s1"):
        return {
            "user_id": user_id,
            "session_id": session_id,
            "message_id": message_id,
        }

    @staticmethod
    def _case(case_id, relevant_sources, *, user_id="u1", query="咖啡"):
        return {
            "case_id": case_id,
            "user_id": user_id,
            "query": query,
            "relevant_sources": relevant_sources,
        }

    @staticmethod
    def _hit(source, score=1):
        return {
            "message": {
                **source,
                "seq": 1,
                "occurred_at": "2026-10-20T02:00:00+00:00",
                "recorded_at": "2026-10-20T02:00:01+00:00",
                "payload": {"role": "user", "content": "早餐喝咖啡"},
            },
            "score": score,
        }

    def test_mixed_cases_use_macro_average_and_separate_no_answer_denominator(self):
        first, second, third = [self._source(f"m{i}") for i in range(1, 4)]
        cases = [
            self._case("half", [first, second]),
            self._case("third", [first, second, third]),
            self._case("miss", [first]),
            self._case("correct-empty", []),
            self._case("false-positive", []),
        ]
        retriever = Mock(spec=HistoryRetriever)
        retriever.search.side_effect = [
            [self._hit(first)],
            [self._hit(second)],
            [],
            [],
            [self._hit(third)],
        ]

        report = evaluate_retrieval(retriever, cases, top_k=3)

        self.assertEqual(report["top_k"], 3)
        self.assertEqual(report["answerable_count"], 3)
        self.assertEqual(report["no_answer_count"], 2)
        self.assertAlmostEqual(report["mean_recall_at_k"], (1 / 2 + 1 / 3) / 3)
        self.assertAlmostEqual(report["hit_rate_at_k"], 2 / 3)
        self.assertEqual(report["no_answer_empty_rate"], 0.5)
        self.assertEqual(len(report["details"]), 5)
        self.assertEqual(
            [detail["case_id"] for detail in report["details"]],
            [case["case_id"] for case in cases],
        )
        self.assertEqual(
            report["details"][0],
            {
                "case_id": "half",
                "user_id": "u1",
                "query": "咖啡",
                "relevant_sources": [first, second],
                "retrieved": [{"source": first, "score": 1}],
                "recall_at_k": 0.5,
                "hit_at_k": 1,
                "empty_return": False,
            },
        )
        self.assertEqual(report["details"][2]["recall_at_k"], 0.0)
        self.assertEqual(report["details"][2]["hit_at_k"], 0)
        self.assertTrue(report["details"][2]["empty_return"])
        for detail in report["details"][3:]:
            self.assertIsNone(detail["recall_at_k"])
            self.assertIsNone(detail["hit_at_k"])

    def test_only_answerable_cases_leave_no_answer_rate_none(self):
        retriever = Mock(spec=HistoryRetriever)
        retriever.search.return_value = []

        report = evaluate_retrieval(
            retriever, [self._case("miss", [self._source("m1")])],
        )

        self.assertEqual(report["top_k"], 5)
        self.assertEqual(report["answerable_count"], 1)
        self.assertEqual(report["no_answer_count"], 0)
        self.assertEqual(report["mean_recall_at_k"], 0.0)
        self.assertEqual(report["hit_rate_at_k"], 0.0)
        self.assertIsNone(report["no_answer_empty_rate"])

    def test_only_no_answer_cases_leave_recall_and_hit_none(self):
        retriever = Mock(spec=HistoryRetriever)
        retriever.search.return_value = []

        report = evaluate_retrieval(retriever, [self._case("empty", [])])

        self.assertEqual(report["answerable_count"], 0)
        self.assertEqual(report["no_answer_count"], 1)
        self.assertIsNone(report["mean_recall_at_k"])
        self.assertIsNone(report["hit_rate_at_k"])
        self.assertEqual(report["no_answer_empty_rate"], 1.0)
        self.assertTrue(report["details"][0]["empty_return"])

    def test_empty_cases_return_counts_zero_and_metrics_none_without_search(self):
        retriever = Mock(spec=HistoryRetriever)

        self.assertEqual(
            evaluate_retrieval(retriever, [], top_k=2),
            {
                "top_k": 2,
                "answerable_count": 0,
                "no_answer_count": 0,
                "mean_recall_at_k": None,
                "hit_rate_at_k": None,
                "no_answer_empty_rate": None,
                "details": [],
            },
        )
        retriever.search.assert_not_called()

    def test_duplicate_labels_and_results_do_not_inflate_scores(self):
        first, second = self._source("m1"), self._source("m2")
        relevant = [first, first, second]
        hits = [self._hit(first), self._hit(first), self._hit(second)]
        retriever = Mock(spec=HistoryRetriever)
        retriever.search.return_value = hits

        report = evaluate_retrieval(retriever, [self._case("duplicates", relevant)])

        self.assertEqual(report["mean_recall_at_k"], 1.0)
        self.assertEqual(report["hit_rate_at_k"], 1.0)
        detail = report["details"][0]
        self.assertEqual(detail["relevant_sources"], relevant)
        self.assertEqual(
            detail["retrieved"],
            [
                {"source": first, "score": 1},
                {"source": first, "score": 1},
                {"source": second, "score": 1},
            ],
        )

    def test_top_k_is_applied_before_deduplication_without_refilling(self):
        first, second = self._source("m1"), self._source("m2")
        retriever = Mock(spec=HistoryRetriever)
        retriever.search.return_value = [
            self._hit(first, 3), self._hit(first, 2), self._hit(second, 1),
        ]

        report = evaluate_retrieval(
            retriever, [self._case("cutoff", [first, second])], top_k=2,
        )

        self.assertEqual(report["mean_recall_at_k"], 0.5)
        self.assertEqual(report["hit_rate_at_k"], 1.0)
        self.assertEqual(
            report["details"][0]["retrieved"],
            [{"source": first, "score": 3}, {"source": first, "score": 2}],
        )

    def test_source_matching_uses_session_and_user_along_with_message_id(self):
        expected = self._source("shared", session_id="s2")
        other_session = self._source("shared", session_id="s1")
        other_user = self._source("shared", user_id="u2", session_id="s2")
        retriever = Mock(spec=HistoryRetriever)
        retriever.search.side_effect = [
            [self._hit(other_session)], [self._hit(other_user)],
        ]

        report = evaluate_retrieval(
            retriever,
            [
                self._case("session-miss", [expected]),
                self._case("own-user-hit", [other_user], user_id="u2"),
            ],
        )

        self.assertEqual(report["mean_recall_at_k"], 0.5)
        self.assertEqual(report["details"][0]["hit_at_k"], 0)
        self.assertEqual(report["details"][1]["hit_at_k"], 1)

    def test_labels_stay_out_of_search_and_report_copies_inputs_and_serializes(self):
        source = self._source("m1")
        cases = [self._case("source-trace", [source], query="早餐 咖啡")]
        hits = [self._hit(source, 2)]
        before_cases, before_hits = copy.deepcopy(cases), copy.deepcopy(hits)
        retriever = Mock(spec=HistoryRetriever)
        retriever.search.return_value = hits

        report = evaluate_retrieval(retriever, cases, top_k=4)

        self.assertEqual(
            retriever.search.call_args_list,
            [call(user_id="u1", query="早餐 咖啡", top_k=4)],
        )
        self.assertEqual(cases, before_cases)
        self.assertEqual(hits, before_hits)
        self.assertEqual(json.loads(json.dumps(report, ensure_ascii=False)), report)
        detail = report["details"][0]
        detail["relevant_sources"][0]["message_id"] = "changed-label"
        detail["retrieved"][0]["source"]["session_id"] = "changed-source"
        detail["retrieved"][0]["score"] = 99
        self.assertEqual(cases, before_cases)
        self.assertEqual(hits, before_hits)

    def test_invalid_top_k_raises_value_error_even_for_empty_cases(self):
        retriever = Mock(spec=HistoryRetriever)
        for top_k in (0, -1, 1.5, "2", None, True, False):
            with self.subTest(top_k=top_k):
                with self.assertRaises(ValueError):
                    evaluate_retrieval(retriever, [], top_k=top_k)
        retriever.search.assert_not_called()

    def test_search_exception_is_propagated_instead_of_scored_as_empty(self):
        retriever = Mock(spec=HistoryRetriever)
        error = RuntimeError("检索失败")
        retriever.search.side_effect = error

        with self.assertRaises(RuntimeError) as raised:
            evaluate_retrieval(retriever, [self._case("broken-search", [])])

        self.assertIs(raised.exception, error)

    def test_other_user_in_labels_or_retrieved_sources_raises_value_error(self):
        other_user = self._source("m1", user_id="u2")
        for location in ("label", "retrieved"):
            with self.subTest(location=location):
                retriever = Mock(spec=HistoryRetriever)
                retriever.search.return_value = (
                    [self._hit(other_user)] if location == "retrieved" else []
                )
                relevant = [other_user] if location == "label" else []
                with self.assertRaises(ValueError):
                    evaluate_retrieval(retriever, [self._case("isolated", relevant)])
                if location == "label":
                    retriever.search.assert_not_called()

    def test_real_store_and_retriever_preserve_sources_across_sessions(self):
        project_root = Path(__file__).resolve().parents[1]
        temp_dir = tempfile.TemporaryDirectory(
            prefix=".retrieval-evaluator-test-", dir=project_root,
        )
        self.assertTrue(Path(temp_dir.name).resolve().is_relative_to(project_root))
        self.addCleanup(temp_dir.cleanup)
        store = MessageStore(Path(temp_dir.name) / "messages.sqlite")
        self.addCleanup(store.close)
        for user_id, session_id, content in (
            ("u1", "s1", "早餐喝咖啡"),
            ("u1", "s2", "下午喝咖啡"),
            ("u2", "s1", "早餐喝咖啡"),
        ):
            store.append_message(
                user_id=user_id,
                session_id=session_id,
                message_id="shared",
                occurred_at="2026-10-20T10:00:00+08:00",
                payload={"role": "user", "content": content},
            )
        sources = [self._source("shared", session_id=session) for session in ("s1", "s2")]

        report = evaluate_retrieval(
            HistoryRetriever(store),
            [
                self._case("coffee", sources),
                self._case("no-swimming", [], query="游泳"),
            ],
            top_k=2,
        )

        self.assertEqual(report["mean_recall_at_k"], 1.0)
        self.assertEqual(report["hit_rate_at_k"], 1.0)
        self.assertEqual(report["no_answer_empty_rate"], 1.0)
        self.assertEqual(
            [hit["source"] for hit in report["details"][0]["retrieved"]], sources,
        )


if __name__ == "__main__":
    unittest.main()
