from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval.run_benchmark import _model_adapter, load_config, main, run_from_config
from src.model_adapter import ModelNotConfiguredError


FIXTURES = Path(__file__).parent / "fixtures"


class FakeAnswerAdapter:
    def __init__(self):
        self.calls = []
        self.last_usage = {"input_tokens": 7, "output_tokens": 2}

    def complete_json(self, *, task, payload):
        if task != "benchmark_answer":
            raise AssertionError(task)
        self.calls.append(deepcopy(payload))
        return json.dumps({"answer": "Hangzhou"})


class RealBenchmarkCliTests(unittest.TestCase):
    def test_full_locomo_run_exports_predictions_report_and_unexecuted_command(self):
        adapter = FakeAnswerAdapter()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            repository = root / "locomo"
            script = repository / "task_eval" / "evaluation.py"
            script.parent.mkdir(parents=True)
            script.write_text("# official fixture", encoding="utf-8")
            report_path = root / "report.json"
            predictions_path = root / "predictions.json"
            command_path = root / "official-command.json"
            result = run_from_config({
                "benchmark": "locomo",
                "dataset_path": str((FIXTURES / "locomo_sample.json").resolve()),
                "model": "fake-model",
                "api_mode": "chat_completions",
                "base_url": "https://api.example.invalid/v1",
                "max_output_tokens": 321,
                "top_k": 2,
                "report_path": str(report_path),
                "predictions_path": str(predictions_path),
                "official_repository": str(repository),
                "official_command_path": str(command_path),
                "run_config": {"retriever": "lexical", "api_key": "redact-me"},
            }, adapter=adapter, allow_full=True)

            self.assertEqual(2, result["report"]["overall"]["question_count"])
            self.assertEqual(2, len(adapter.calls))
            first = adapter.calls[0]
            self.assertEqual("Where did Alice move?", first["question"])
            self.assertIn("occurred_at", first["instructions"])
            self.assertIn("not the question's as_of", first["instructions"])
            self.assertIn("yesterday", first["instructions"])
            self.assertIn("last week", first["instructions"])
            self.assertIn("next month", first["instructions"])
            self.assertIn("this month", first["instructions"])
            self.assertIn("Do not use ISO date format", first["instructions"])
            self.assertIn("duration", first["instructions"])
            self.assertIn("D1:1", [
                item["source"]["message_id"] for item in first["retrieved_context"]
            ])
            self.assertFalse({"answer", "evidence", "official"} & set(first))
            self.assertEqual("[REDACTED]", result["report"]["run_config"]["api_key"])
            self.assertEqual("locomo10", result["report"]["run_config"]["dataset_variant"])
            self.assertEqual(
                "raw-lexical-json-v2",
                result["report"]["run_config"]["answer_prompt_version"],
            )
            self.assertEqual(
                "chat_completions", result["report"]["run_config"]["api_mode"],
            )
            self.assertEqual(
                "https://api.example.invalid/v1",
                result["report"]["run_config"]["base_url"],
            )
            self.assertEqual(321, result["report"]["run_config"]["max_output_tokens"])
            self.assertEqual(14, result["report"]["overall"]["query_usage"]["input_tokens"])
            self.assertTrue(report_path.is_file())
            exported = json.loads(predictions_path.read_text(encoding="utf-8"))
            self.assertEqual("Hangzhou", exported[0]["qa"][0]["memory_prediction"])
            self.assertIn("D1:1", exported[0]["qa"][0]["memory_prediction_context"])
            command = json.loads(command_path.read_text(encoding="utf-8"))
            self.assertFalse(command["executed"])
            self.assertEqual(result["report"]["dataset_sha256"], command["dataset_sha256"])
            self.assertEqual(64, len(command["predictions_sha256"]))
            self.assertEqual(64, len(command["entrypoint_sha256"]))
            self.assertEqual("--output", command["argv"][-2])
            self.assertEqual(command["expected_output_path"], command["argv"][-1])

    def test_limited_smoke_runs_one_question_without_official_prediction_artifacts(self):
        adapter = FakeAnswerAdapter()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            predictions = root / "must-not-exist.json"
            command = root / "must-not-exist-command.json"
            result = run_from_config({
                "benchmark": "locomo",
                "dataset_path": "locomo_sample.json",
                "model": "fake-model",
                "top_k": 1,
                "max_context_characters": 5,
                "report_path": str(root / "smoke-report.json"),
                "predictions_path": str(predictions),
                "official_repository": str(root / "unused"),
                "official_command_path": str(command),
            }, adapter=adapter, config_dir=FIXTURES,
                limit_histories=1, limit_questions=1)

            self.assertEqual(1, result["report"]["overall"]["question_count"])
            self.assertEqual(1, len(adapter.calls))
            self.assertLessEqual(sum(len(item["content"]) for item in
                                     adapter.calls[0]["retrieved_context"]), 5)
            self.assertFalse(result["report"]["run_scope"]["full"])
            self.assertIsNone(result["predictions_path"])
            self.assertIsNone(result["official_command_path"])
            self.assertFalse(predictions.exists())
            self.assertFalse(command.exists())

    def test_full_longmemeval_run_exports_official_jsonl_shape(self):
        adapter = FakeAnswerAdapter()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            predictions = root / "predictions.jsonl"
            result = run_from_config({
                "benchmark": "longmemeval",
                "dataset_path": str((FIXTURES / "longmemeval_sample.json").resolve()),
                "model": "fake-model",
                "top_k": 1,
                "report_path": str(root / "report.json"),
                "predictions_path": str(predictions),
            }, adapter=adapter, allow_full=True)

            rows = [json.loads(line) for line in predictions.read_text(
                encoding="utf-8",
            ).splitlines()]
            self.assertEqual(2, len(rows))
            self.assertEqual({"question_id", "hypothesis"}, set(rows[0]))
            self.assertEqual("fixture-q1", rows[0]["question_id"])
            self.assertTrue(result["report"]["official_scoring"]["eligible"])
            self.assertFalse(result["report"]["official_scoring"]["command_generated"])

    def test_missing_credentials_and_invalid_limits_fail_before_model_calls(self):
        base = {
            "benchmark": "locomo",
            "dataset_path": str((FIXTURES / "locomo_sample.json").resolve()),
            "model": "fake-model",
            "api_key_env": "MEMORY_BENCHMARK_TEST_KEY",
        }
        with patch.dict(os.environ, {"MEMORY_BENCHMARK_TEST_KEY": ""}):
            with self.assertRaises(ModelNotConfiguredError):
                run_from_config(base, allow_full=True)
        with patch.dict(os.environ, {"MEMORY_BENCHMARK_TEST_KEY": "secret"}):
            configured = _model_adapter({
                **base,
                "base_url": "https://api.siliconflow.cn/v1",
                "api_mode": "chat_completions",
                "timeout_seconds": 180,
                "max_output_tokens": 2048,
            })
        self.assertEqual("chat_completions", configured.config.api_mode)
        self.assertEqual("https://api.siliconflow.cn/v1", configured.config.base_url)
        self.assertEqual(180, configured.config.timeout_seconds)
        self.assertEqual(2048, configured.config.max_output_tokens)
        adapter = FakeAnswerAdapter()
        with self.assertRaises(ValueError):
            run_from_config(base, adapter=adapter, limit_questions=0)
        with self.assertRaises(ValueError):
            run_from_config(base, adapter=adapter, limit_histories=1, limit_questions=2)
        self.assertEqual([], adapter.calls)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "duplicate.json"
            path.write_text('{"benchmark":"locomo","benchmark":"longmemeval"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)
        with patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--config", "not-read-without-run-scope.json"])
            with self.assertRaises(SystemExit):
                main(["--config", "not-read.json", "--limit-histories", "1"])
            with self.assertRaises(SystemExit):
                main(["--config", "not-read.json", "--full", "--limit-questions", "1"])


if __name__ == "__main__":
    unittest.main()
