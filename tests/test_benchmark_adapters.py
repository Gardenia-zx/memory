import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from eval.score_locomo_official import score_predictions
from src.eval.benchmark_datasets import load_benchmark_dataset, timestamp
from src.eval.benchmark_runner import (
    export_locomo_predictions,
    export_predictions_jsonl,
    official_score_command,
    run_benchmark,
    save_benchmark_report,
)


FIXTURES = Path(__file__).parent / "fixtures"


class BenchmarkDatasetTests(unittest.TestCase):
    def test_longmemeval_groups_shared_history_without_ingesting_labels(self):
        dataset = load_benchmark_dataset(
            FIXTURES / "longmemeval_sample.json", benchmark="longmemeval",
        )
        self.assertEqual(1, len(dataset["histories"]))
        history = dataset["histories"][0]
        self.assertEqual(["first", "second"], [
            message["session_id"] for message in history["messages"]
        ])
        serialized = json.dumps(history["messages"], ensure_ascii=False)
        self.assertNotIn("Where did I move?", serialized)
        self.assertNotIn("Not stated", serialized)
        self.assertEqual(2, len(history["questions"]))
        self.assertEqual("turn:1", history["questions"][0]["evidence_sources"][0]["message_id"])
        self.assertFalse(history["questions"][1]["answerable"])

    def test_locomo_keeps_speaker_time_and_media_but_excludes_gold_summaries(self):
        dataset = load_benchmark_dataset(
            FIXTURES / "locomo_sample.json", benchmark="locomo",
            assumed_timezone="+08:00",
        )
        history = dataset["histories"][0]
        serialized = json.dumps(history["messages"], ensure_ascii=False)
        self.assertIn("Alice: I moved to Hangzhou.", serialized)
        self.assertIn("[Image caption] A cup of tea", serialized)
        self.assertNotIn("DO_NOT_INGEST", serialized)
        self.assertNotIn("WRONG_DISTRACTOR", serialized)
        self.assertEqual("2023-05-08T05:56:00+00:00", history["messages"][0]["occurred_at"])
        self.assertEqual("D1:1", history["questions"][0]["evidence_sources"][0]["message_id"])
        self.assertIsNone(history["questions"][1]["answer"])

    def test_locomo_normalizes_compound_evidence_and_reports_dangling_labels(self):
        raw = json.loads((FIXTURES / "locomo_sample.json").read_text(encoding="utf-8"))
        raw[0]["qa"][0]["evidence"] = [
            "D1:1; D1:2 D2:01", "D", "D9:9",
        ]
        raw[0]["qa"][1]["evidence"] = ["D:1:2"]
        from src.eval.benchmark_datasets import adapt_locomo

        history = adapt_locomo(raw)[0]
        question = history["questions"][0]
        self.assertEqual(["D1:1", "D1:2", "D2:1"], [
            source["message_id"] for source in question["evidence_sources"]
        ])
        self.assertEqual(["D", "D9:9"], question["unresolved_evidence"])
        self.assertEqual(raw[0]["qa"][0]["evidence"], question["official"]["evidence"])
        self.assertEqual("D1:2", history["questions"][1]["evidence_sources"][0]["message_id"])

    def test_timestamp_requires_explicit_timezone_assumption(self):
        self.assertEqual("2026-01-01T00:00:00+00:00", timestamp(
            "2026-01-01T08:00:00", "+08:00",
        ))
        with self.assertRaises(ValueError):
            timestamp("2026-01-01", "local")

    def test_longmemeval_preserves_repeated_official_sessions_with_unique_internal_ids(self):
        entry = {
            "question_id": "duplicate-session-shape",
            "question_type": "single-session-user",
            "question": "Where did I move?",
            "answer": "Hangzhou",
            "question_date": "2024/02/03 12:00",
            "haystack_session_ids": ["repeated", "repeated"],
            "haystack_dates": ["2024/02/01 08:00", "2024/02/02 09:00"],
            "haystack_sessions": [
                [{"role": "user", "content": "filler"}],
                [{"role": "user", "content": "I moved to Hangzhou", "has_answer": True}],
            ],
            "answer_session_ids": ["repeated"],
        }
        from src.eval.benchmark_datasets import adapt_longmemeval

        history = adapt_longmemeval([entry])[0]
        session_ids = [message["session_id"] for message in history["messages"]]
        self.assertEqual(2, len(set(session_ids)))
        self.assertEqual("repeated", session_ids[0])
        self.assertTrue(session_ids[1].startswith("repeated#occurrence-"))
        question = history["questions"][0]
        self.assertEqual([session_ids[1]], question["evidence_session_ids"])
        self.assertEqual(session_ids[1], question["evidence_sources"][0]["session_id"])


class BenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        self.longmem = load_benchmark_dataset(
            FIXTURES / "longmemeval_sample.json", benchmark="longmemeval",
        )
        self.locomo = load_benchmark_dataset(
            FIXTURES / "locomo_sample.json", benchmark="locomo",
        )

    def test_fixed_snapshot_label_isolation_metrics_costs_and_secret_redaction(self):
        build_calls, public_questions = [], []

        def build(store, messages):
            build_calls.append(len(messages))
            for message in messages:
                store.append_message(**message)
            return {"input_tokens": 10, "output_tokens": 2, "cost_usd": 0.01}

        def answer(connection, question):
            public_questions.append(question)
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM messages")
            row = connection.execute("""
                SELECT user_id,session_id,message_id FROM messages
                WHERE payload_json LIKE '%杭州%' LIMIT 1
            """).fetchone()
            return {
                "answer": "杭州" if "move" in question["question"] else "信息不足",
                "retrieved_sources": [] if row is None else [{
                    "type": "message", "user_id": row["user_id"],
                    "session_id": row["session_id"], "message_id": row["message_id"],
                }],
                "usage": {"input_tokens": 5, "output_tokens": 1, "cost_usd": 0.002},
            }

        def score(labelled, answer_text):
            return 1.0 if answer_text else 0.0

        report = run_benchmark(
            self.longmem, answer_question=answer, build_history=build,
            score_answer=score, top_k=2,
            run_config={
                "OPENAI_API_KEY": "do-not-save", "provider-token": "also-secret",
                "nested": [{"client_secret": "hidden"}], "model": "configured-model",
            },
        )
        self.assertEqual([2], build_calls)
        self.assertEqual(2, len(public_questions))
        self.assertTrue(all(set(question) == {"request_id", "question", "user_id", "as_of"}
                            for question in public_questions))
        self.assertEqual("[REDACTED]", report["run_config"]["OPENAI_API_KEY"])
        self.assertEqual("[REDACTED]", report["run_config"]["provider-token"])
        self.assertEqual("[REDACTED]", report["run_config"]["nested"][0]["client_secret"])
        self.assertEqual("configured-model", report["run_config"]["model"])
        self.assertEqual(1.0, report["overall"]["answer_score"])
        self.assertEqual(10, report["overall"]["query_usage"]["input_tokens"])
        self.assertEqual(10, report["build_usage"]["input_tokens"])
        self.assertEqual(1, len(report["snapshot_sha256"]))

    def test_exports_reports_and_official_commands_without_executing_scorers(self):
        report = run_benchmark(
            self.longmem,
            answer_question=lambda connection, question: {"answer": "answer"},
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            jsonl = root / "nested" / "predictions.jsonl"
            full = root / "nested" / "report.json"
            export_predictions_jsonl(report, jsonl)
            save_benchmark_report(report, full)
            rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({"question_id", "hypothesis"}, set(rows[0]))
            self.assertEqual("not_run", json.loads(full.read_text(encoding="utf-8"))["answer_metric"])

            repository = root / "LongMemEval"
            script = repository / "src" / "evaluation" / "evaluate_qa.py"
            script.parent.mkdir(parents=True)
            script.write_text("# official fixture", encoding="utf-8")
            command = official_score_command(
                benchmark="longmemeval", repository=repository,
                predictions_path=jsonl,
                dataset_path=FIXTURES / "longmemeval_sample.json",
                python_executable="python", judge_model="gpt-4o",
            )
            self.assertEqual([
                "python", str(script.resolve()), "gpt-4o", str(jsonl.resolve()),
                str((FIXTURES / "longmemeval_sample.json").resolve()),
            ], command["argv"])
            self.assertFalse(command["executed"])

    def test_locomo_export_and_official_bridge_with_injected_scorer(self):
        history = self.locomo["histories"][0]

        def answer(connection, question):
            first = connection.execute("""
                SELECT user_id,session_id,message_id FROM messages ORDER BY session_id,seq LIMIT 1
            """).fetchone()
            return {"answer": "Hangzhou", "retrieved_sources": [{
                "type": "message", "user_id": first["user_id"],
                "session_id": first["session_id"], "message_id": first["message_id"],
            }]}

        report = run_benchmark(self.locomo, answer_question=answer)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            predictions = root / "locomo-predictions.json"
            export_locomo_predictions(self.locomo, report, predictions)
            exported = json.loads(predictions.read_text(encoding="utf-8"))
            self.assertEqual(history["original_id"], exported[0]["sample_id"])
            self.assertEqual("Hangzhou", exported[0]["qa"][0]["memory_prediction"])
            self.assertEqual(["D1:1"], exported[0]["qa"][0]["memory_prediction_context"])

            repository = root / "locomo"
            script = repository / "task_eval" / "evaluation.py"
            script.parent.mkdir(parents=True)
            script.write_text("# official fixture", encoding="utf-8")

            def evaluator(qas, eval_key):
                self.assertEqual("memory_prediction", eval_key)
                return [1.0] * len(qas), 0.0, [0.5] * len(qas)

            scores = score_predictions(
                repository=repository, predictions=predictions,
                dataset=FIXTURES / "locomo_sample.json", evaluator=evaluator,
            )
            self.assertEqual(1.0, scores["overall"])
            self.assertEqual(0.5, scores["retrieval_recall"])
            self.assertEqual(2, scores["question_count"])

    def test_cross_user_retrieval_is_rejected(self):
        other = {"type": "message", "user_id": "other", "session_id": "s", "message_id": "m"}
        with self.assertRaises(ValueError):
            run_benchmark(
                self.longmem,
                answer_question=lambda connection, question: {
                    "answer": "x", "retrieved_sources": [other],
                },
            )


if __name__ == "__main__":
    unittest.main()
