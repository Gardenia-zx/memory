"""Synthetic tests for eval.diagnostics; no real dataset or model is needed."""

import json
from pathlib import Path
import tempfile
import unittest

from eval.diagnostics import metrics
from eval.diagnostics.run_diagnostics import evaluate, prepare
from src.eval.benchmark_datasets import load_benchmark_dataset


SAMPLE = {
    "sample_id": "conv-test",
    "conversation": {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session_1_date_time": "2023-05-01",
        "session_1": [
            {"dia_id": "D1:1", "speaker": "Alice",
             "text": "I bought a red bicycle yesterday."},
            {"dia_id": "D1:2", "speaker": "Bob",
             "text": "How much did it cost?"},
            {"dia_id": "D1:3", "speaker": "Alice",
             "text": "It cost 200 dollars."},
        ],
        "session_2_date_time": "2023-06-01",
        "session_2": [
            {"dia_id": "D2:1", "speaker": "Bob",
             "text": "My sister visited me last week."},
            {"dia_id": "D2:2", "speaker": "Alice",
             "text": "Did she enjoy the city?"},
        ],
    },
    "qa": [
        {"question": "What color was the bicycle?", "answer": "red",
         "evidence": ["D1:1"], "category": 4},
        {"question": "When did Alice buy the bicycle?", "answer": "2023-05-01",
         "evidence": ["D1:1"], "category": 2},
        {"question": "Who visited Bob?", "answer": ["Bob's sister", "his sister"],
         "evidence": ["D2:1", "D2:2"], "category": 1},
        {"question": "What might Alice do with the bicycle?", "answer": "ride; commute",
         "evidence": ["D1:1"], "category": 3},
    ],
}


class DiagnosticHelpersTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "locomo.json"
        self.path.write_text(json.dumps([SAMPLE], ensure_ascii=False), encoding="utf-8")
        self.dataset = load_benchmark_dataset(
            self.path, benchmark="locomo", assumed_timezone="+00:00"
        )

    def test_bm25_scores_match_production_retriever(self):
        context = prepare(self.dataset, history="conv-test", limit=4)
        try:
            documents = [
                record["payload"].get("content", "") for record in context.records
            ]
            hits = context.retriever.search(
                user_id=context.history["user_id"], query="bicycle",
                top_k=len(documents) + 1,
            )
            computed = metrics.bm25_scores("bicycle", documents)
            self.assertEqual(
                sorted(round(score, 9) for score in computed if score > 0),
                sorted(round(hit["score"], 9) for hit in hits),
            )
        finally:
            context.store.close()

    def test_neighbor_distance_uses_session_and_seq(self):
        context = prepare(self.dataset, history="conv-test", limit=1)
        try:
            positions = metrics.message_positions(context.records)
            self.assertEqual(positions[("session_1", "D1:1")], 1)
            # A hand-built hit keeps this a unit test: the retriever would also
            # hit session_2 through the stopword "the" ("... the city").
            hits = [{
                "source": {
                    "type": "message", "user_id": context.history["user_id"],
                    "session_id": "session_1", "message_id": "D1:1",
                },
                "text": "I bought a red bicycle yesterday.",
            }]
            distance = metrics.neighbor_distance(
                {"session_id": "session_1", "message_id": "D1:2"}, hits, positions
            )
            self.assertEqual(distance, 1)
            self.assertIsNone(metrics.neighbor_distance(
                {"session_id": "session_2", "message_id": "D2:1"}, hits, positions
            ))
        finally:
            context.store.close()

    def test_session_documents_carry_date_and_rank(self):
        context = prepare(self.dataset, history="conv-test", limit=1)
        try:
            session_ids, documents = metrics.session_documents(
                context.records, with_date=True
            )
            self.assertEqual(session_ids, ["session_1", "session_2"])
            self.assertIn("1 May 2023", documents[0])
            scores = metrics.bm25_scores("bicycle", documents)
            ranks = {
                session_id: position
                for position, index in enumerate(metrics.ranked_indices(scores), 1)
                for session_id in [session_ids[index]]
            }
            self.assertEqual(ranks["session_1"], 1)
            self.assertEqual(
                metrics.coverage_at(ranks, 5, ["session_1"]), 1.0
            )
            self.assertIsNone(metrics.coverage_at(ranks, 5, []))
        finally:
            context.store.close()

    def test_answer_support_and_diagnostic_f1(self):
        support = metrics.answer_support(
            "Alice: I bought a red bicycle yesterday.", ["red"]
        )
        self.assertTrue(support["contains"])
        self.assertTrue(support["supported"])
        self.assertFalse(metrics.answer_support("nothing here", ["red"])["supported"])
        self.assertEqual(metrics.gold_answers(
            {"answer": "ride; commute", "category": "3"}
        ), ["ride", "commute"])
        self.assertAlmostEqual(metrics.token_f1("red bicycle", "red bicycle"), 1.0)
        self.assertEqual(metrics.token_f1("blue", "red"), 0.0)
        self.assertAlmostEqual(
            metrics.best_token_f1("red bike", ["blue", "red bike"]), 1.0
        )
        self.assertTrue(metrics.is_abstention("No information available"))

    def test_recall_ceiling_and_buckets(self):
        self.assertAlmostEqual(metrics.recall_ceiling(5, 2), 1.0)
        self.assertAlmostEqual(metrics.recall_ceiling(5, 10), 0.5)
        self.assertIsNone(metrics.recall_ceiling(5, 0))
        self.assertEqual(metrics.rank_bucket(1), "1-5")
        self.assertEqual(metrics.rank_bucket(6), "6-20")
        self.assertEqual(metrics.rank_bucket(21), "21-50")
        self.assertEqual(metrics.rank_bucket(None), "unscored")


class DiagnosticEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        path = Path(self.folder.name) / "locomo.json"
        path.write_text(json.dumps([SAMPLE], ensure_ascii=False), encoding="utf-8")
        self.dataset = load_benchmark_dataset(
            path, benchmark="locomo", assumed_timezone="+00:00"
        )

    def test_evaluate_selected_checks_without_report(self):
        result = evaluate(
            self.dataset, history="conv-test", limit=4, report=None,
            checks_names=["V0", "V1", "V2", "V3", "V4", "V9"],
        )
        self.assertEqual(len(result["dataset"]["question_ids"]), 4)
        self.assertEqual(result["checks"]["V0"]["status"], "skipped")
        aggregate = result["checks"]["V0"]["metrics"]["aggregate"]
        self.assertGreaterEqual(aggregate["source_recall_at_k"], 0.0)
        self.assertLessEqual(aggregate["source_recall_at_k"], 1.0)
        self.assertEqual(len(result["checks"]["V4"]["metrics"]["curve"]), 5)
        self.assertEqual(
            set(result["checks"]["V3"]["metrics"]["variants"]),
            {"text_only", "with_date"},
        )
        self.assertEqual(
            result["checks"]["V9"]["metrics"]["selected"]["category_counts"],
            {"4": 1, "2": 1, "1": 1, "3": 1},
        )
        self.assertFalse(result["summary"]["reproduction_ok"])

    def test_v0_flags_report_mismatch(self):
        report = {
            "dataset_sha256": self.dataset["dataset_sha256"],
            "overall": {"source_recall_at_k": 0.5, "session_recall_at_k": 0.5},
            "predictions": [
                {
                    "question_id": f"locomo:conv-test:{index}",
                    "source_recall_at_k": 0.123,
                    "session_recall_at_k": 0.0,
                    "retrieved_sources": [],
                }
                for index in range(4)
            ],
        }
        result = evaluate(
            self.dataset, history="conv-test", limit=4, report=report,
            checks_names=["V0"],
        )
        self.assertEqual(result["checks"]["V0"]["status"], "fail")
        kinds = {mismatch["kind"] for mismatch in result["checks"]["V0"]["mismatches"]}
        self.assertIn("source_recall", kinds)
        self.assertIn("sources", kinds)
        self.assertIn("aggregate", kinds)

    def test_v0_passes_on_matching_report(self):
        baseline = evaluate(
            self.dataset, history="conv-test", limit=4, report=None,
            checks_names=["V0"],
        )
        aggregate = baseline["checks"]["V0"]["metrics"]["aggregate"]
        report = {
            "dataset_sha256": self.dataset["dataset_sha256"],
            "overall": aggregate,
            "predictions": [
                {
                    "question_id": row["question_id"],
                    "source_recall_at_k": row["source_recall_at_k"],
                    "session_recall_at_k": row["session_recall_at_k"],
                    "retrieved_sources": [
                        {"type": "message", "user_id": key[0],
                         "session_id": key[1], "message_id": key[2]}
                        for key in row["sources"]
                    ],
                }
                for row in baseline["checks"]["V0"]["per_question"]
            ],
        }
        result = evaluate(
            self.dataset, history="conv-test", limit=4, report=report,
            checks_names=["V0"],
        )
        self.assertEqual(result["checks"]["V0"]["status"], "pass")
        self.assertTrue(result["summary"]["reproduction_ok"])

    def test_v6_and_v7_skip_without_report(self):
        result = evaluate(
            self.dataset, history="conv-test", limit=4, report=None,
            checks_names=["V6", "V7"],
        )
        self.assertEqual(result["checks"]["V6"]["status"], "skipped")
        self.assertEqual(result["checks"]["V7"]["status"], "skipped")

    def test_v10_session_first_simulation(self):
        result = evaluate(
            self.dataset, history="conv-test", limit=4, report=None,
            checks_names=["V0", "V10"],
        )
        metrics = result["checks"]["V10"]["metrics"]
        self.assertEqual(len(metrics["configs"]), 140)
        self.assertTrue(metrics["pareto"])
        self.assertIn("source_recall_at_k", metrics["recommended"])
        self.assertEqual(metrics["total_questions"], 4)
        self.assertEqual(result["checks"]["V10"]["status"], "info")
        # The session-first configs must actually recover evidence; a zero
        # maximum would mean the evidence keys and selected keys disagree.
        self.assertGreater(
            max(config["source_recall_at_k"] for config in metrics["configs"]),
            0.0,
        )
        flat = {row["name"]: row for row in metrics["references"]}
        self.assertAlmostEqual(
            flat["flat_top5"]["source_recall_at_k"],
            result["checks"]["V0"]["metrics"]["aggregate"]["source_recall_at_k"],
        )

    def test_limit_zero_means_all_questions(self):
        result = evaluate(
            self.dataset, history="conv-test", limit=0, report=None,
            checks_names=["V0"],
        )
        self.assertEqual(len(result["dataset"]["question_ids"]), 4)


if __name__ == "__main__":
    unittest.main()
