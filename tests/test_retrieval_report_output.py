"""验证检索报告的 JSON 保存和命令入口输出。"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from eval import run_retrieval_eval as runner


class RetrievalReportOutputTests(unittest.TestCase):
    def setUp(self):
        project_root = Path(__file__).resolve().parents[1]
        temp_dir = tempfile.TemporaryDirectory(
            prefix=".retrieval-report-test-", dir=project_root,
        )
        self.temp_path = Path(temp_dir.name).resolve()
        self.assertTrue(self.temp_path.is_relative_to(project_root))
        self.addCleanup(temp_dir.cleanup)

    @staticmethod
    def _report():
        source = {
            "user_id": "用户一",
            "session_id": "会话一",
            "message_id": "消息一",
        }
        return {
            "dataset_id": "中文开发数据_v1",
            "retriever": "keyword_v1",
            "top_k": 2,
            "answerable_count": 1,
            "no_answer_count": 0,
            "mean_recall_at_k": 1.0,
            "hit_rate_at_k": 1.0,
            "no_answer_empty_rate": None,
            "details": [{
                "case_id": "早餐饮品",
                "user_id": "用户一",
                "query": "早餐 咖啡",
                "relevant_sources": [source],
                "retrieved": [{"source": dict(source), "score": 2}],
                "recall_at_k": 1.0,
                "hit_at_k": 1,
                "empty_return": False,
            }],
        }

    def test_save_creates_nested_directories_and_preserves_complete_utf8_report(self):
        report = self._report()
        report_path = self.temp_path / "多层" / "reports" / "baseline.json"

        saved_path = runner.save_report(report, str(report_path))

        self.assertEqual(saved_path, report_path)
        text = report_path.read_text(encoding="utf-8")
        self.assertIn("中文开发数据_v1", text)
        self.assertIn("早餐 咖啡", text)
        self.assertIn('"no_answer_empty_rate": null', text)
        self.assertTrue(text.endswith("\n"))
        self.assertEqual(json.loads(text), report)

    def test_saving_again_replaces_previous_report_without_stale_content(self):
        report_path = self.temp_path / "baseline.json"
        runner.save_report(self._report(), report_path)
        replacement = {"dataset_id": "下一版", "details": []}

        runner.save_report(replacement, report_path)

        self.assertEqual(
            json.loads(report_path.read_text(encoding="utf-8")), replacement,
        )
        self.assertNotIn("早餐饮品", report_path.read_text(encoding="utf-8"))

    def test_nonfinite_metrics_raise_without_changing_existing_report(self):
        report_path = self.temp_path / "baseline.json"
        runner.save_report(self._report(), report_path)
        original_bytes = report_path.read_bytes()

        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                invalid_report = self._report()
                invalid_report["mean_recall_at_k"] = value

                with self.assertRaises(ValueError):
                    runner.save_report(invalid_report, report_path)

                self.assertEqual(report_path.read_bytes(), original_bytes)

    def test_unserializable_report_raises_before_creating_or_overwriting_output(self):
        report_path = self.temp_path / "baseline.json"
        runner.save_report(self._report(), report_path)
        original_bytes = report_path.read_bytes()
        new_path = self.temp_path / "not-created" / "baseline.json"
        invalid_report = {"details": [{"unsupported": object()}]}

        for path in (report_path, new_path):
            with self.subTest(path=path):
                with self.assertRaises(TypeError):
                    runner.save_report(invalid_report, path)

        self.assertEqual(report_path.read_bytes(), original_bytes)
        self.assertFalse(new_path.parent.exists())

    def test_main_runs_builtin_dataset_and_saves_report_with_matching_console_summary(self):
        report_path = self.temp_path / "reports" / "baseline.json"
        output = io.StringIO()

        with (
            patch.object(runner, "DEFAULT_REPORT_PATH", report_path),
            patch.object(runner, "run_evaluation", wraps=runner.run_evaluation) as evaluate,
            redirect_stdout(output),
        ):
            runner.main()

        evaluate.assert_called_once_with(runner.DEFAULT_DATASET_PATH, top_k=2)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["dataset_id"], "retrieval_dev_v1")
        self.assertEqual(report["retriever"], "keyword_v1")
        self.assertEqual(report["top_k"], 2)
        self.assertEqual(len(report["details"]), 8)
        self.assertEqual(report["answerable_count"], 6)
        self.assertEqual(report["no_answer_count"], 2)
        console = output.getvalue()
        for expected in (
            "retrieval_dev_v1", "keyword_v1", "top_k：2",
            "案例数：8", "有答案 6", "无答案 2",
            "Recall@2：77.78%", "Hit@2：83.33%",
            "无答案返回空比例：100.00%", str(report_path.resolve()),
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, console)

    def test_main_displays_unavailable_metrics_as_na_and_saves_them_as_null(self):
        report = self._report()
        report.update(
            answerable_count=0,
            no_answer_count=0,
            mean_recall_at_k=None,
            hit_rate_at_k=None,
            no_answer_empty_rate=None,
            details=[],
        )
        report_path = self.temp_path / "empty" / "baseline.json"
        output = io.StringIO()

        with (
            patch.object(runner, "DEFAULT_REPORT_PATH", report_path),
            patch.object(runner, "run_evaluation", return_value=report),
            redirect_stdout(output),
        ):
            runner.main()

        self.assertEqual(
            json.loads(report_path.read_text(encoding="utf-8")), report,
        )
        console = output.getvalue()
        self.assertIn("案例数：0", console)
        for label in ("Recall@2", "Hit@2", "无答案返回空比例"):
            with self.subTest(label=label):
                self.assertIn(f"{label}：N/A", console)


if __name__ == "__main__":
    unittest.main()
