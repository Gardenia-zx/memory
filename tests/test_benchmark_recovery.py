"""Retry and checkpoint behaviour for paid benchmark runs."""

from pathlib import Path
import tempfile
import unittest

from src.eval.benchmark_datasets import load_benchmark_dataset
from src.eval.benchmark_runner import run_benchmark


FIXTURES = Path(__file__).parent / "fixtures"


class BenchmarkRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.dataset = load_benchmark_dataset(
            FIXTURES / "locomo_sample.json", benchmark="locomo",
            assumed_timezone="+08:00",
        )

    def test_transient_error_is_retried(self):
        calls = []

        def answer(connection, question):
            calls.append(1)
            if len(calls) == 1:
                try:
                    raise TimeoutError("read operation timed out")
                except TimeoutError as error:
                    raise RuntimeError("wrapped network failure") from error
            return {"answer": "ok"}

        report = run_benchmark(
            self.dataset, answer_question=answer,
            answer_attempts=2, retry_delay_seconds=0,
        )
        self.assertEqual(2, report["overall"]["question_count"])
        self.assertEqual(3, len(calls))
        self.assertEqual(2, report["answer_attempts"])

    def test_non_transient_error_is_not_retried(self):
        calls = []

        def answer(connection, question):
            calls.append(1)
            raise ValueError("programming error")

        with self.assertRaises(ValueError):
            run_benchmark(
                self.dataset, answer_question=answer,
                answer_attempts=3, retry_delay_seconds=0,
            )
        self.assertEqual(1, len(calls))

    def test_checkpoint_resumes_completed_questions(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            checkpoint = Path(folder) / "answers.checkpoint.jsonl"
            first_calls = []

            def first(connection, question):
                first_calls.append(1)
                if len(first_calls) == 2:
                    raise RuntimeError("aborted after the first question")
                return {"answer": "first"}

            with self.assertRaises(RuntimeError):
                run_benchmark(
                    self.dataset, answer_question=first,
                    checkpoint_path=checkpoint,
                )
            self.assertTrue(checkpoint.is_file())

            second_calls = []

            def second(connection, question):
                second_calls.append(1)
                return {"answer": "second"}

            report = run_benchmark(
                self.dataset, answer_question=second,
                checkpoint_path=checkpoint,
            )
            self.assertEqual(1, len(second_calls))
            self.assertEqual(1, report["reused_answers"])
            self.assertEqual(2, report["overall"]["question_count"])
            self.assertEqual(
                ["first", "second"],
                [row["hypothesis"] for row in report["predictions"]],
            )

    def test_checkpoint_is_ignored_when_config_changes(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            checkpoint = Path(folder) / "answers.checkpoint.jsonl"
            run_benchmark(
                self.dataset,
                answer_question=lambda connection, question: {"answer": "first"},
                checkpoint_path=checkpoint,
            )
            calls = []

            def answer(connection, question):
                calls.append(1)
                return {"answer": "second"}

            report = run_benchmark(
                self.dataset, answer_question=answer,
                checkpoint_path=checkpoint,
                run_config={"model": "a different model"},
            )
            self.assertEqual(2, len(calls))
            self.assertEqual(0, report["reused_answers"])

    def test_invalid_recovery_configuration_raises(self):
        for kwargs in (
            {"answer_attempts": 0},
            {"answer_attempts": True},
            {"retry_delay_seconds": -1},
            {"retry_delay_seconds": "2"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_benchmark(
                    self.dataset,
                    answer_question=lambda connection, question: {"answer": "x"},
                    **kwargs,
                )


if __name__ == "__main__":
    unittest.main()
