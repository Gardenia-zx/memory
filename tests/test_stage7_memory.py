"""第七关独立功能测试：摘要压缩、任务恢复与有证据的经历事件。"""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.context_compactor import ContextCompactor
from src.context_queue import ContextQueue
from src.error import ContextBudgetExceeded
from src.event_store import EventStore
from src.message_store import MessageStore
from src.summary_store import SummaryStore
from src.task_store import TaskStore


class Stage7MemoryTests(unittest.TestCase):
    def setUp(self):
        self.messages = MessageStore(":memory:")
        self.addCleanup(self.messages.close)
        self.summaries = SummaryStore(self.messages)
        self.tasks = TaskStore(self.messages)
        self.events = EventStore(self.messages)
        for user_id in ("u1", "u2"):
            for index in range(1, 7):
                self.messages.append_message(
                    user_id=user_id, session_id="chat", message_id=f"m{index}",
                    occurred_at=f"2026-10-25T10:00:0{index}+08:00",
                    payload={"role": "user", "content": f"原始记录 {user_id}/{index}"},
                )
        self.source = {"user_id": "u1", "session_id": "chat", "message_id": "m1"}

    def summary(self, summary_id="s1", **changes):
        return self.summaries.save_summary(**{
            "user_id": "u1", "session_id": "chat", "summary_id": summary_id,
            "content": "前四条消息的已确认摘要", "start_seq": 1, "end_seq": 4, **changes,
        })

    def event(self, event_id="e1", **changes):
        return self.events.record_event(**{
            "user_id": "u1", "event_id": event_id,
            "occurred_at": "2026-10-25T10:00:00+08:00", "text": "完成了代码审查",
            "payload": {"kind": "review", "result": ["通过"]}, "sources": [self.source], **changes,
        })

    def queue(self):
        queue = ContextQueue(self.messages, user_id="u1", session_id="chat")
        queue.refresh()
        return queue

    def test_summary_validates_real_contiguous_coverage_and_first_write_identity(self):
        first = self.summary()
        changes = self.messages.conn.total_changes
        self.assertEqual(self.summaries.save_summary(user_id="u1", session_id="chat", summary_id="s1"), first)
        self.assertEqual(self.summary(content=None, start_seq=False, end_seq=99), first)
        self.assertEqual(self.messages.conn.total_changes, changes)
        for changes in ({"start_seq": True}, {"start_seq": 0}, {"end_seq": 0},
                        {"end_seq": 7}, {"session_id": "absent"}, {"content": " "}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.summary("invalid", **changes)
        # 相同摘要 ID 在不同用户的真实会话中仍是独立记录。
        other = self.summary(user_id="u2", content="另一用户的摘要")
        self.assertEqual(other["user_id"], "u2")
        self.assertEqual(self.summaries.get_summary(user_id="u1", session_id="chat", summary_id="s1"), first)
        self.assertIsNone(self.summaries.get_summary(user_id="u1", session_id="other", summary_id="s1"))

    def test_summary_insert_ignored_or_failed_is_not_reported_as_saved(self):
        for mode in ("FAIL", "IGNORE"):
            abort = "RAISE(FAIL, 'summary failed')" if mode == "FAIL" else "RAISE(IGNORE)"
            with self.messages.conn:
                self.messages.conn.execute(f"""
                    CREATE TRIGGER fail_summary BEFORE INSERT ON summaries BEGIN
                        UPDATE messages SET occurred_at = 'side-effect' WHERE user_id = 'u2';
                        SELECT {abort}; END
                """)
            original = self.messages.list_user_messages(user_id="u2")
            try:
                with self.assertRaises(sqlite3.IntegrityError if mode == "FAIL" else ValueError):
                    self.summary()
                self.assertEqual(self.messages.list_user_messages(user_id="u2"), original)
                self.assertEqual(self.summaries.list_summaries(user_id="u1", session_id="chat"), [])
            finally:
                with self.messages.conn:
                    self.messages.conn.execute("DROP TRIGGER fail_summary")

    def test_summary_boundary_never_splits_an_assistant_tool_call_group(self):
        self.messages.append_message(
            user_id="u1", session_id="tools", message_id="call",
            occurred_at="2026-10-25T02:00:00Z",
            payload={"role": "assistant", "content": None, "tool_calls": [{"id": "t1"}]},
        )
        self.messages.append_message(
            user_id="u1", session_id="tools", message_id="result",
            occurred_at="2026-10-25T02:00:01Z",
            payload={"role": "tool", "content": "ok", "tool_call_id": "t1"},
        )
        with self.assertRaises(ValueError):
            self.summaries.save_summary(
                user_id="u1", session_id="tools", summary_id="split",
                content="不能只覆盖调用", start_seq=1, end_seq=1,
            )
        saved = self.summaries.save_summary(
            user_id="u1", session_id="tools", summary_id="whole",
            content="调用及结果", start_seq=1, end_seq=2,
        )
        self.assertEqual(saved["end_seq"], 2)

    def test_compactor_replaces_only_covered_prefix_and_keeps_raw_history_and_watermark(self):
        self.summary()
        queue = self.queue()
        original = self.messages.list_messages(user_id="u1", session_id="chat")
        compactor = ContextCompactor(queue, self.summaries)
        context = compactor.build_context(max_tokens=3, token_counter=len, keep_recent=2)
        coverage = json.loads(context[0]["content"])
        self.assertEqual((coverage["type"], coverage["start_seq"], coverage["end_seq"]), ("history_summary", 1, 4))
        self.assertEqual(context[1:], [record["payload"] for record in original[-2:]])
        self.assertEqual(queue.to_messages(), context)
        self.assertEqual(queue.last_seq, 6)
        self.assertEqual(queue.refresh(), 0)
        self.assertEqual(self.messages.list_messages(user_id="u1", session_id="chat"), original)
        context[0]["content"] = "调用方修改"
        self.assertNotEqual(queue.to_messages()[0]["content"], "调用方修改")
        self.assertEqual(len(queue.build_context(max_tokens=3, token_counter=len)), 3)

    def test_compactor_supports_refresh_and_a_new_summary_covering_previous_summary(self):
        self.summary()
        queue = self.queue()
        compactor = ContextCompactor(queue, self.summaries)
        compactor.build_context(max_tokens=3, token_counter=len)
        self.messages.append_message(user_id="u1", session_id="chat", message_id="m7",
                                     occurred_at="2026-10-25T03:00:00+00:00",
                                     payload={"role": "assistant", "content": "最新原文"})
        self.assertEqual(queue.refresh(), 1)
        self.summary("s2", end_seq=5, content="覆盖前五条的累计摘要")
        result = compactor.build_context(max_tokens=3, token_counter=len)
        self.assertEqual(json.loads(result[0]["content"])["summary_id"], "s2")
        self.assertEqual(result[-1], {"role": "assistant", "content": "最新原文"})
        self.assertEqual(queue.last_seq, 7)
        self.assertEqual(len(self.messages.list_messages(user_id="u1", session_id="chat")), 7)

    def test_compactor_does_nothing_within_budget_and_fails_without_losing_messages(self):
        queue = self.queue()
        compactor = ContextCompactor(queue, self.summaries)
        original = deepcopy(queue._records)
        self.assertEqual(compactor.build_context(max_tokens=6, token_counter=len), queue.to_messages())
        for summary_id in (None, "missing"):
            with self.assertRaises(ContextBudgetExceeded):
                compactor.build_context(max_tokens=3, token_counter=len, summary_id=summary_id)
            self.assertEqual(queue._records, original)
        self.summary("not-prefix", start_seq=2)
        self.summary("covers-recent", end_seq=5)
        for summary_id in ("not-prefix", "covers-recent"):
            with self.assertRaises(ContextBudgetExceeded):
                compactor.build_context(max_tokens=3, token_counter=len, summary_id=summary_id)
            self.assertEqual(queue._records, original)
        self.summary()
        with self.assertRaises(ContextBudgetExceeded):
            compactor.build_context(max_tokens=2, token_counter=len)
        self.assertEqual(queue._records, original)

    def test_compactor_validates_budget_and_counter_without_mutating_queue(self):
        queue = self.queue()
        compactor = ContextCompactor(queue, self.summaries)
        original = deepcopy(queue._records)
        for options in ({"max_tokens": True}, {"max_tokens": -1}, {"keep_recent": 0},
                        {"keep_recent": True}, {"token_counter": None},
                        {"token_counter": lambda messages: True}, {"token_counter": lambda messages: -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                compactor.build_context(**{"max_tokens": 6, "token_counter": len, **options})
            self.assertEqual(queue._records, original)
        def mutating_counter(messages):
            messages[0]["content"] = "counter 修改"
            return len(messages)
        compactor.build_context(max_tokens=6, token_counter=mutating_counter)
        self.assertEqual(queue._records, original)
        other = MessageStore(":memory:")
        try:
            with self.assertRaises(ValueError):
                ContextCompactor(queue, SummaryStore(other))
        finally:
            other.close()

    def test_task_lifecycle_recovery_identity_and_user_isolation(self):
        first = self.tasks.create_task(user_id="u1", task_id="t1", title="继续开发", payload={"steps": [1]})
        changes = self.messages.conn.total_changes
        self.assertEqual(self.tasks.create_task(user_id="u1", task_id="t1", title=None), first)
        self.assertEqual(self.tasks.set_status(user_id="u1", task_id="t1", status="pending"), first)
        self.assertEqual(self.messages.conn.total_changes, changes)
        done = self.tasks.set_status(user_id="u1", task_id="t1", status="done")
        self.assertEqual(done["status"], "done")
        self.assertGreaterEqual(done["updated_at"], first["updated_at"])
        self.assertEqual(self.tasks.list_tasks(user_id="u1", status="pending"), [])
        pending = self.tasks.set_status(user_id="u1", task_id="t1", status="pending")
        self.assertEqual(self.tasks.list_tasks(user_id="u1", status="pending"), [pending])
        other = self.tasks.create_task(user_id="u2", task_id="t1", title="其他用户")
        self.tasks.set_status(user_id="u1", task_id="t1", status="cancelled")
        self.assertEqual(self.tasks.get_task(user_id="u2", task_id="t1"), other)
        pending["payload"]["steps"].append(2)
        self.assertEqual(self.tasks.get_task(user_id="u1", task_id="t1")["payload"], {"steps": [1]})

    def test_task_rejects_invalid_inputs_and_status_without_creating_work(self):
        for fields in ({"title": " "}, {"payload": []}, {"payload": {"x": float("nan")}}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.tasks.create_task(**{"user_id": "u1", "task_id": "bad", "title": "任务", **fields})
        for status in (None, True, "running", ""):
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.tasks.set_status(user_id="u1", task_id="absent", status=status)
        with self.assertRaises(ValueError):
            self.tasks.set_status(user_id="u1", task_id="absent", status="done")
        self.assertEqual(self.tasks.list_tasks(user_id="u1"), [])

    def test_task_failed_or_ignored_update_rolls_back_side_effects(self):
        first = self.tasks.create_task(user_id="u1", task_id="t1", title="任务")
        for mode in ("FAIL", "IGNORE"):
            abort = "RAISE(FAIL, 'task failed')" if mode == "FAIL" else "RAISE(IGNORE)"
            with self.messages.conn:
                self.messages.conn.execute(f"""
                    CREATE TRIGGER fail_task BEFORE UPDATE ON tasks BEGIN
                        UPDATE messages SET occurred_at = 'side-effect' WHERE user_id = 'u2';
                        SELECT {abort}; END
                """)
            original = self.messages.list_user_messages(user_id="u2")
            try:
                with self.assertRaises(sqlite3.IntegrityError if mode == "FAIL" else ValueError):
                    self.tasks.set_status(user_id="u1", task_id="t1", status="done")
                self.assertEqual(self.tasks.get_task(user_id="u1", task_id="t1"), first)
                self.assertEqual(self.messages.list_user_messages(user_id="u2"), original)
            finally:
                with self.messages.conn:
                    self.messages.conn.execute("DROP TRIGGER fail_task")

    def test_events_keep_real_sources_normalized_time_and_first_write_snapshot(self):
        source = {**self.source, "label": "不保存额外字段"}
        first = self.event(sources=[source, self.source])
        self.assertEqual(first["occurred_at"], "2026-10-25T02:00:00+00:00")
        self.assertEqual(first["sources"], [self.source])
        changes = self.messages.conn.total_changes
        self.assertEqual(self.events.record_event(user_id="u1", event_id="e1"), first)
        self.assertEqual(self.event(occurred_at=None, text=None, payload=[], sources=[]), first)
        self.assertEqual(self.messages.conn.total_changes, changes)
        first["payload"]["result"].append("外部修改")
        self.assertEqual(self.events.get_event(user_id="u1", event_id="e1")["payload"]["result"], ["通过"])
        structured = self.event("structured", text="", payload={"kind": "execution", "ok": True})
        self.assertEqual(structured["text"], "")

    def test_events_distinguish_plan_execution_and_verified_results_with_participants(self):
        planned = self.event(
            "planned", phase="planned", participants=["user:u1", "agent", "user:u1"],
        )
        executed = self.event("executed", phase="executed", participants=["agent"])
        verified = self.event("verified", phase="verified", participants=["reviewer"])
        self.assertEqual(planned["participants"], ["user:u1", "agent"])
        self.assertEqual([planned["phase"], executed["phase"], verified["phase"]],
                         ["planned", "executed", "verified"])
        for phase in (None, True, "done"):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                self.event(f"bad-{phase}", phase=phase)

    def test_event_queries_use_half_open_occurrence_times_and_never_other_users(self):
        first = self.event()
        second = self.event("e2", occurred_at="2026-10-25T02:00:01+00:00")
        self.event(user_id="u2", sources=[{**self.source, "user_id": "u2"}])
        self.assertEqual(self.events.list_events(user_id="u1", start_at="2026-10-25T10:00:00+08:00",
                                                end_at="2026-10-25T02:00:01+00:00"), [first])
        self.assertEqual(self.events.list_events(user_id="u1", start_at="2026-10-25T02:00:01+00:00"), [second])
        self.assertEqual(self.events.list_events(user_id="u1"), [first, second])
        self.assertIsNone(self.events.get_event(user_id="u1", event_id="absent"))

    def test_events_reject_fake_cross_user_sources_and_invalid_time_or_payload(self):
        for fields in ({"sources": []}, {"sources": [None]},
                       {"sources": [{**self.source, "message_id": "missing"}]},
                       {"sources": [{**self.source, "user_id": "u2"}]},
                       {"occurred_at": "2026-10-25T10:00:00"}, {"occurred_at": True},
                       {"payload": {"x": float("nan")}}, {"payload": []},
                       {"text": " ", "payload": {}}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.event("bad", **fields)
        with self.assertRaises(ValueError):
            self.events.list_events(user_id="u1", start_at="2026-10-25T02:00:00Z", end_at="2026-10-25T02:00:00Z")
        self.assertEqual(self.events.list_events(user_id="u1"), [])

    def test_event_late_source_failure_or_ignore_rolls_back_event_and_prior_sources(self):
        sources = [self.source, {**self.source, "message_id": "m2"}]
        for mode in ("FAIL", "IGNORE"):
            abort = "RAISE(FAIL, 'event failed')" if mode == "FAIL" else "RAISE(IGNORE)"
            with self.messages.conn:
                self.messages.conn.execute(f"""
                    CREATE TRIGGER fail_event BEFORE INSERT ON event_sources
                    WHEN NEW.source_message_id = 'm2' BEGIN
                        UPDATE messages SET occurred_at = 'side-effect' WHERE user_id = 'u2';
                        SELECT {abort}; END
                """)
            original = self.messages.list_user_messages(user_id="u2")
            try:
                with self.assertRaises(sqlite3.IntegrityError if mode == "FAIL" else ValueError):
                    self.event(sources=sources)
                self.assertEqual(self.events.list_events(user_id="u1"), [])
                self.assertEqual(self.messages.conn.execute("SELECT COUNT(*) FROM event_sources").fetchone()[0], 0)
                self.assertEqual(self.messages.list_user_messages(user_id="u2"), original)
            finally:
                with self.messages.conn:
                    self.messages.conn.execute("DROP TRIGGER fail_event")

    def test_all_three_stores_survive_reopen_and_summary_can_rebuild_a_new_queue(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "stage7.sqlite"
            messages = MessageStore(path)
            try:
                for index in range(1, 5):
                    messages.append_message(user_id="u1", session_id="chat", message_id=f"m{index}",
                                            occurred_at="2026-10-25T02:00:00Z",
                                            payload={"role": "user", "content": f"原文{index}"})
                summary = SummaryStore(messages).save_summary(user_id="u1", session_id="chat", summary_id="s1",
                                                              content="前两条摘要", start_seq=1, end_seq=2)
                tasks = TaskStore(messages)
                task = tasks.create_task(user_id="u1", task_id="t1", title="重启后继续")
                tasks.create_task(user_id="u1", task_id="t2", title="已完成")
                tasks.set_status(user_id="u1", task_id="t2", status="done")
                event = EventStore(messages).record_event(user_id="u1", event_id="e1", occurred_at="2026-10-25T02:00:00Z",
                                                           text="保存了进度", sources=[self.source])
            finally:
                messages.close()
            messages = MessageStore(path)
            try:
                summaries = SummaryStore(messages)
                self.assertEqual(summaries.get_summary(user_id="u1", session_id="chat", summary_id="s1"), summary)
                self.assertEqual(TaskStore(messages).list_tasks(user_id="u1", status="pending"), [task])
                self.assertEqual(EventStore(messages).get_event(user_id="u1", event_id="e1"), event)
                queue = ContextQueue(messages, user_id="u1", session_id="chat")
                queue.refresh()
                context = ContextCompactor(queue, summaries).build_context(max_tokens=3, token_counter=len)
                self.assertEqual(json.loads(context[0]["content"])["summary_id"], "s1")
                self.assertEqual(len(messages.list_messages(user_id="u1", session_id="chat")), 4)
            finally:
                messages.close()


if __name__ == "__main__":
    unittest.main()
