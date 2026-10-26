import sqlite3
import unittest

from src.message_store import MessageStore
from src.summary_generator import SummaryGenerator
from src.summary_store import SummaryStore


class FakeAdapter:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, *, task, payload):
        self.calls.append((task, payload))
        return self.responses.pop(0)


class SummaryGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.store = MessageStore(":memory:")
        self.summaries = SummaryStore(self.store)
        for index, content in enumerate(("第一条", "第二条", "第三条"), 1):
            self.store.append_message(
                user_id="u1", session_id="s1", message_id=f"m{index}",
                occurred_at=f"2026-01-0{index}T10:00:00+08:00",
                payload={"role": "user", "content": content},
            )

    def tearDown(self):
        self.store.close()

    def test_generates_from_exact_saved_range_and_retry_does_not_call_model(self):
        adapter = FakeAdapter(['{"summary":"用户连续提供了三条信息。"}'])
        generator = SummaryGenerator(self.summaries, adapter)
        saved = generator.generate(
            user_id="u1", session_id="s1", summary_id="sum1", start_seq=1, end_seq=3,
        )
        retried = generator.generate(
            user_id="u1", session_id="s1", summary_id="sum1", start_seq=99, end_seq=100,
        )
        self.assertEqual("用户连续提供了三条信息。", saved["content"])
        self.assertEqual(saved, retried)
        self.assertEqual(1, len(adapter.calls))
        self.assertEqual(["第一条", "第二条", "第三条"], [
            message["content"] for message in adapter.calls[0][1]["messages"]
        ])

    def test_rejects_bad_model_json_without_saving(self):
        adapter = FakeAdapter(['{"summary":"a","summary":"b"}'])
        with self.assertRaises(ValueError):
            SummaryGenerator(self.summaries, adapter).generate(
                user_id="u1", session_id="s1", summary_id="sum1", start_seq=1, end_seq=2,
            )
        self.assertIsNone(self.summaries.get_summary(
            user_id="u1", session_id="s1", summary_id="sum1",
        ))

    def test_failed_summary_insert_does_not_report_success(self):
        self.store.conn.execute("""
            CREATE TRIGGER ignore_summary BEFORE INSERT ON summaries
            BEGIN SELECT RAISE(IGNORE); END
        """)
        adapter = FakeAdapter(['{"summary":"摘要"}'])
        with self.assertRaises(ValueError):
            SummaryGenerator(self.summaries, adapter).generate(
                user_id="u1", session_id="s1", summary_id="sum1", start_seq=1, end_seq=2,
            )


if __name__ == "__main__":
    unittest.main()
