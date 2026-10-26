"""验证固定数据加载和基线运行；运行函数保持只返回结果字典。"""

import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from eval.run_retrieval_eval import DEFAULT_DATASET_PATH, run_evaluation
from src.message_store import MessageStore


class RunRetrievalEvalTests(unittest.TestCase):
    def setUp(self):
        project_root = Path(__file__).resolve().parents[1]
        temp_dir = tempfile.TemporaryDirectory(
            prefix=".run-retrieval-eval-test-", dir=project_root,
        )
        self.assertTrue(Path(temp_dir.name).resolve().is_relative_to(project_root))
        self.addCleanup(temp_dir.cleanup)
        self.dataset_path = Path(temp_dir.name) / "dataset.json"

    @staticmethod
    def _source(message_id, *, user_id="u1", session_id="s1"):
        return {
            "user_id": user_id,
            "session_id": session_id,
            "message_id": message_id,
        }

    @classmethod
    def _dataset(cls):
        first = cls._source("m1")
        second = cls._source("m2", session_id="s2")
        other = cls._source("m1", user_id="u2")
        history = [
            {
                **source,
                "occurred_at": "2026-10-20T10:00:00+08:00",
                "payload": {"role": "user", "content": content},
            }
            for source, content in (
                (first, "早餐喝咖啡"),
                (second, "下午喝咖啡"),
                (other, "游泳"),
            )
        ]
        return {
            "dataset_id": "中文开发数据_v1",
            "history": history,
            "cases": [
                {
                    "case_id": "coffee",
                    "user_id": "u1",
                    "query": "咖啡",
                    "relevant_sources": [first, second],
                },
                {
                    "case_id": "other-user-only",
                    "user_id": "u1",
                    "query": "游泳",
                    "relevant_sources": [],
                },
            ],
        }

    def _write_dataset(self, dataset):
        self.dataset_path.write_text(
            json.dumps(dataset, ensure_ascii=False), encoding="utf-8",
        )
        return self.dataset_path

    def test_real_fixture_preserves_sources_metrics_and_metadata_without_files(self):
        dataset = self._dataset()
        dataset_path = self._write_dataset(dataset)

        report = run_evaluation(dataset_path, top_k=2)

        self.assertEqual(report["dataset_id"], dataset["dataset_id"])
        self.assertEqual(report["retriever"], "keyword_v1")
        self.assertEqual(report["top_k"], 2)
        self.assertEqual(report["answerable_count"], 1)
        self.assertEqual(report["no_answer_count"], 1)
        self.assertEqual(report["mean_recall_at_k"], 1.0)
        self.assertEqual(report["hit_rate_at_k"], 1.0)
        self.assertEqual(report["no_answer_empty_rate"], 1.0)
        self.assertEqual(
            report["details"][0]["retrieved"],
            [
                {"source": self._source("m1"), "score": 1},
                {"source": self._source("m2", session_id="s2"), "score": 1},
            ],
        )
        self.assertEqual(json.loads(json.dumps(report, ensure_ascii=False)), report)
        self.assertEqual(list(dataset_path.parent.iterdir()), [dataset_path])

    def test_repeated_runs_are_reproducible_and_do_not_reuse_previous_history(self):
        dataset_path = self._write_dataset(self._dataset())
        first = run_evaluation(str(dataset_path), top_k=2)
        self.assertEqual(run_evaluation(dataset_path, top_k=2), first)
        empty_dataset = {
            "dataset_id": "empty-history",
            "history": [],
            "cases": [{
                "case_id": "previous-history-must-disappear",
                "user_id": "u1",
                "query": "咖啡",
                "relevant_sources": [],
            }],
        }

        report = run_evaluation(self._write_dataset(empty_dataset))

        self.assertEqual(report["top_k"], 5)
        self.assertEqual(report["answerable_count"], 0)
        self.assertIsNone(report["mean_recall_at_k"])
        self.assertEqual(report["no_answer_empty_rate"], 1.0)
        self.assertEqual(report["details"][0]["retrieved"], [])

    def test_builtin_dataset_is_fixed_and_records_keyword_baseline_limits(self):
        report = run_evaluation(DEFAULT_DATASET_PATH, top_k=2)

        self.assertEqual(report["answerable_count"], 6)
        self.assertEqual(report["no_answer_count"], 2)
        self.assertEqual(len(report["details"]), 8)
        self.assertAlmostEqual(report["mean_recall_at_k"], 7 / 9)
        self.assertAlmostEqual(report["hit_rate_at_k"], 5 / 6)
        self.assertEqual(report["no_answer_empty_rate"], 1.0)
        self.assertEqual(run_evaluation(DEFAULT_DATASET_PATH, top_k=2), report)

    def test_invalid_top_level_data_raises_value_error(self):
        valid = self._dataset()
        invalid_datasets = [[], None]
        for field, value in (
            ("dataset_id", " "), ("dataset_id", 1),
            ("history", {}), ("cases", None),
        ):
            invalid_datasets.append({**valid, field: value})
        for field in ("dataset_id", "history", "cases"):
            invalid_datasets.append({key: value for key, value in valid.items() if key != field})
        for dataset in invalid_datasets:
            with self.subTest(dataset=dataset):
                with self.assertRaises(ValueError):
                    run_evaluation(self._write_dataset(dataset))

    def test_case_structure_and_duplicate_case_id_are_rejected(self):
        valid = self._dataset()
        invalid_cases = [None]
        for field, value in (
            ("case_id", " "), ("user_id", None),
            ("query", 12), ("relevant_sources", {}),
            ("relevant_sources", [{"user_id": "u1", "session_id": "s1"}]),
            ("relevant_sources", [self._source(" ")]),
        ):
            invalid_cases.append({**valid["cases"][0], field: value})
        missing_query = copy.deepcopy(valid["cases"][0])
        missing_query.pop("query")
        invalid_cases.append(missing_query)
        for case in invalid_cases:
            with self.subTest(case=case):
                dataset = {**valid, "cases": [case]}
                with self.assertRaises(ValueError):
                    run_evaluation(self._write_dataset(dataset))
        duplicate = copy.deepcopy(valid)
        duplicate["cases"].append(copy.deepcopy(duplicate["cases"][0]))
        with self.assertRaisesRegex(ValueError, "coffee"):
            run_evaluation(self._write_dataset(duplicate))

    def test_missing_or_foreign_labels_raise_value_error_with_case_id(self):
        for source in (self._source("missing"), self._source("m1", user_id="u2")):
            with self.subTest(source=source):
                dataset = self._dataset()
                dataset["cases"][0]["relevant_sources"] = [source]
                with self.assertRaisesRegex(ValueError, "coffee"):
                    run_evaluation(self._write_dataset(dataset))

    def test_duplicate_history_identity_is_not_silently_hidden_by_idempotency(self):
        dataset = self._dataset()
        duplicate = copy.deepcopy(dataset["history"][0])
        duplicate["payload"]["content"] = "同一来源却出现另一份内容"
        dataset["history"].append(duplicate)

        with self.assertRaises(ValueError):
            run_evaluation(self._write_dataset(dataset))

    def test_only_history_is_ingested_and_connection_closes_on_evaluation_failure(self):
        dataset = self._dataset()
        dataset_path = self._write_dataset(dataset)
        actual_store = MessageStore(":memory:")
        self.addCleanup(actual_store.close)
        store = Mock(wraps=actual_store)
        error = RuntimeError("评测失败")

        with (
            patch("eval.run_retrieval_eval.MessageStore", return_value=store) as factory,
            patch("eval.run_retrieval_eval.evaluate_retrieval", side_effect=error) as evaluate,
        ):
            with self.assertRaises(RuntimeError) as raised:
                run_evaluation(dataset_path, top_k=2)

        self.assertIs(raised.exception, error)
        factory.assert_called_once_with(":memory:")
        self.assertEqual(
            store.append_message.call_args_list,
            [call(**message) for message in dataset["history"]],
        )
        evaluate.assert_called_once()
        self.assertEqual(evaluate.call_args.args[1], dataset["cases"])
        self.assertEqual(evaluate.call_args.kwargs, {"top_k": 2})
        store.close.assert_called_once_with()
        with self.assertRaises(sqlite3.ProgrammingError):
            actual_store.conn.execute("SELECT 1")

    def test_top_k_must_be_positive_integer_even_when_cases_are_empty(self):
        dataset_path = self._write_dataset({
            "dataset_id": "empty", "history": [], "cases": [],
        })
        for top_k in (0, -1, 1.5, "2", None, True, False):
            with self.subTest(top_k=top_k):
                with self.assertRaises(ValueError):
                    run_evaluation(dataset_path, top_k=top_k)

    def test_json_decode_and_missing_file_errors_are_propagated(self):
        self.dataset_path.write_text("{broken JSON", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            run_evaluation(self.dataset_path)
        with self.assertRaises(FileNotFoundError):
            run_evaluation(self.dataset_path.with_name("missing.json"))


if __name__ == "__main__":
    unittest.main()
