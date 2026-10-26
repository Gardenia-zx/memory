import json
import sqlite3
import unittest

from src.candidate_extractor import CandidateExtractor
from src.conflict_store import ConflictStore
from src.fact_store import FactStore
from src.forgetter import MemoryForgetter
from src.memory_pipeline import MemoryPipeline
from src.message_store import MessageStore
from src.model_adapter import ModelNotConfiguredError
from src.job_store import BackgroundWorker, JobStore


class FakeAdapter:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, *, task, payload):
        self.calls.append((task, payload))
        if not self.responses:
            raise AssertionError("模型被额外调用")
        return self.responses.pop(0)


def stable_id(kind, source, index):
    return f"{kind}-{source['message_id']}-{index}"


class MemoryPipelineTests(unittest.TestCase):
    def setUp(self):
        self.messages = MessageStore(":memory:")
        self.facts = FactStore(self.messages)
        self.conflicts = ConflictStore(self.facts)

    def tearDown(self):
        self.messages.close()

    def message(self, message_id, content, *, user_id="u1", session_id="s1"):
        return self.messages.append_message(
            user_id=user_id, session_id=session_id, message_id=message_id,
            occurred_at="2026-10-25T10:00:00+08:00",
            payload={"role": "user", "content": content},
        )

    def test_extractor_binds_trusted_identity_source_and_session_scope(self):
        message = self.message("m1", "我喜欢茶")
        adapter = FakeAdapter(json.dumps({"candidates": [{
            "subject_id": "user:u1", "predicate": "likes",
            "object": {"type": "literal", "value": "茶"}, "scope": "session",
        }]}, ensure_ascii=False))
        candidate = CandidateExtractor(adapter, id_factory=stable_id).extract(
            user_id="u1", message=message, existing_facts=[],
        )[0]
        self.assertEqual("fact-m1-0", candidate["fact_id"])
        self.assertEqual("s1", candidate["session_id"])
        self.assertEqual([{"user_id": "u1", "session_id": "s1", "message_id": "m1"}],
                         candidate["sources"])
        self.assertEqual("extract_memory_candidates", adapter.calls[0][0])

    def test_model_cannot_supply_trusted_fields_or_non_strict_json(self):
        message = self.message("m1", "内容")
        bad = [
            '{"candidates":[{"user_id":"other"}]}',
            '{"candidates":[],"extra":true}',
            '{"candidates":[],"candidates":[]}',
            '{"candidates":[{"subject_id":"x","predicate":"p","object":{"type":"literal","value":NaN}}]}',
        ]
        for response in bad:
            with self.assertRaises(ValueError):
                CandidateExtractor(FakeAdapter(response), id_factory=stable_id).extract(
                    user_id="u1", message=message, existing_facts=[],
                )

    def test_add_pipeline_commits_then_retry_returns_frozen_result_without_model(self):
        self.message("m1", "我喜欢茶")
        adapter = FakeAdapter('{"candidates":[{'
                              '"subject_id":"user:u1","predicate":"likes",'
                              '"object":{"type":"literal","value":"茶"}}]}')
        pipeline = MemoryPipeline(self.facts, adapter, id_factory=stable_id)
        first = pipeline.process_message(user_id="u1", session_id="s1", message_id="m1")
        retried = pipeline.process_message(user_id="u1", session_id="s1", message_id="m1")
        self.assertEqual(first, retried)
        self.assertEqual("applied", first["items"][0]["status"])
        self.assertEqual("茶", first["items"][0]["result"]["fact"]["object"]["value"])
        self.assertEqual(1, len(adapter.calls))

    def test_unresolved_new_value_is_persisted_as_conflict(self):
        source = self.message("m-old", "我喜欢咖啡")
        self.facts.add_fact(
            user_id="u1", fact_id="f-old", subject_id="user:u1", predicate="likes",
            object={"type": "literal", "value": "咖啡"}, sources=[source],
        )
        self.message("m-new", "我喜欢茶")
        adapter = FakeAdapter('{"candidates":[{'
                              '"subject_id":"user:u1","predicate":"likes",'
                              '"object":{"type":"literal","value":"茶"}}]}')
        pipeline = MemoryPipeline(
            self.facts, adapter, id_factory=stable_id, conflict_store=self.conflicts,
        )
        result = pipeline.process_message(
            user_id="u1", session_id="s1", message_id="m-new",
        )
        item = result["items"][0]
        self.assertEqual("conflict", item["status"])
        self.assertEqual("conflict-m-new-0", item["conflict"]["conflict_id"])
        self.assertEqual([{"fact_id": "f-old", "version": 1}], item["related_facts"])
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="fact-m-new-0"))

    def test_committed_update_survives_progress_save_failure_and_retry_does_not_make_v3(self):
        old_source = self.message("m-old", "我住北京")
        self.facts.add_fact(
            user_id="u1", fact_id="f-home", subject_id="user:u1", predicate="lives_in",
            object={"type": "literal", "value": "北京"}, sources=[old_source],
        )
        self.message("m-fix", "纠正：一直住上海")
        adapter = FakeAdapter('{"candidates":[{'
                              '"intent":"correct","subject_id":"user:u1",'
                              '"predicate":"lives_in","object":{"type":"literal","value":"上海"},'
                              '"target":{"fact_id":"f-home","version":1}}]}')
        pipeline = MemoryPipeline(self.facts, adapter, id_factory=stable_id)
        self.messages.conn.execute("""
            CREATE TRIGGER fail_applied_progress BEFORE UPDATE ON memory_pipeline_runs
            WHEN NEW.state_json LIKE '%\"status\": \"applied\"%'
            BEGIN SELECT RAISE(FAIL, 'simulated progress failure'); END
        """)
        with self.assertRaises(sqlite3.IntegrityError):
            pipeline.process_message(user_id="u1", session_id="s1", message_id="m-fix")
        self.assertEqual("上海", self.facts.get_fact(
            user_id="u1", fact_id="f-home", version=2,
        )["object"]["value"])
        self.messages.conn.execute("DROP TRIGGER fail_applied_progress")
        result = pipeline.process_message(user_id="u1", session_id="s1", message_id="m-fix")
        self.assertEqual(2, result["items"][0]["result"]["fact"]["version"])
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="f-home", version=3))
        self.assertEqual(1, len(adapter.calls))

    def test_corrupt_saved_state_is_rejected_without_model_or_fact_write(self):
        self.message("m1", "内容")
        adapter = FakeAdapter('{"candidates":[]}')
        pipeline = MemoryPipeline(self.facts, adapter, id_factory=stable_id)
        pipeline.process_message(user_id="u1", session_id="s1", message_id="m1")
        self.messages.conn.execute("""
            UPDATE memory_pipeline_runs SET state_json = '{"result":{"user_id":"other"}}'
            WHERE user_id='u1' AND session_id='s1' AND message_id='m1'
        """)
        with self.assertRaises(ValueError):
            pipeline.process_message(user_id="u1", session_id="s1", message_id="m1")
        self.assertEqual(1, len(adapter.calls))
        self.assertEqual([], pipeline.load_fact_snapshot(user_id="u1"))

    def test_forget_removes_run_and_blocks_cached_or_new_processing(self):
        self.message("m1", "无事实")
        pipeline = MemoryPipeline(self.facts, FakeAdapter('{"candidates":[]}'))
        pipeline.process_message(user_id="u1", session_id="s1", message_id="m1")
        MemoryForgetter(self.messages).forget_user(user_id="u1", reason="user request")
        with self.assertRaises(ValueError):
            pipeline.process_message(user_id="u1", session_id="s1", message_id="m1")
        self.assertEqual(0, self.messages.conn.execute(
            "SELECT COUNT(*) FROM memory_pipeline_runs WHERE user_id='u1'",
        ).fetchone()[0])

    def test_unconfigured_default_never_sends_or_silently_skips(self):
        self.message("m1", "需要提取")
        with self.assertRaises(ModelNotConfiguredError):
            MemoryPipeline(self.facts).process_message(
                user_id="u1", session_id="s1", message_id="m1",
            )

    def test_background_worker_runs_real_pipeline_and_persists_retryable_result(self):
        self.message("m1", "我喜欢茶")
        adapter = FakeAdapter('{"candidates":[{'
                              '"subject_id":"user:u1","predicate":"likes",'
                              '"object":{"type":"literal","value":"茶"}}]}')
        pipeline = MemoryPipeline(self.facts, adapter, id_factory=stable_id)
        jobs = JobStore(self.messages)
        jobs.enqueue(
            user_id="u1", job_id="job-m1", kind="extract_message", subject_key="s1/m1",
            payload={"user_id": "u1", "session_id": "s1", "message_id": "m1"},
        )
        completed = BackgroundWorker(
            jobs, {"extract_message": pipeline.handle_job},
        ).run_once(user_id="u1")
        self.assertEqual("succeeded", completed["status"])
        self.assertEqual("completed", completed["result"]["status"])
        self.assertEqual("茶", self.facts.get_fact(
            user_id="u1", fact_id="fact-m1-0",
        )["object"]["value"])
        self.assertEqual(1, len(adapter.calls))

    def test_pipeline_job_rejects_payload_user_spoofing(self):
        pipeline = MemoryPipeline(self.facts, FakeAdapter('{"candidates":[]}'))
        with self.assertRaises(ValueError):
            pipeline.handle_job(
                {"user_id": "other", "session_id": "s1", "message_id": "m1"},
                {"user_id": "u1"},
            )


if __name__ == "__main__":
    unittest.main()
