import unittest

from src.entity_graph import EntityGraphStore
from src.event_store import EventStore
from src.external_document_store import ExternalDocumentStore
from src.fact_store import FactStore
from src.forgetter import MemoryForgetter
from src.job_store import JobStore
from src.memory_pipeline import MemoryPipeline
from src.message_store import MessageStore
from src.summary_generator import SummaryGenerator
from src.summary_store import SummaryStore
from src.task_store import TaskStore


class FakeAdapter:
    def __init__(self, response):
        self.response = response

    def complete_json(self, *, task, payload):
        return self.response


class FullMemoryLifecycleTests(unittest.TestCase):
    def test_all_current_user_artifacts_are_traceable_then_forgotten_atomically(self):
        store = MessageStore(":memory:")
        self.addCleanup(store.close)
        facts = FactStore(store)
        summaries = SummaryStore(store)
        tasks = TaskStore(store)
        events = EventStore(store)
        documents = ExternalDocumentStore(store)
        graph = EntityGraphStore(store)
        jobs = JobStore(store)

        source = {"user_id": "u1", "session_id": "s1", "message_id": "m1"}
        store.append_message(
            **source, occurred_at="2026-10-25T10:00:00+08:00",
            payload={"role": "user", "content": "我在杭州喜欢龙井茶"},
        )
        store.append_message(
            user_id="u1", session_id="s1", message_id="m2",
            occurred_at="2026-10-25T10:01:00+08:00",
            payload={"role": "assistant", "content": "已经记下"},
        )
        pipeline = MemoryPipeline(facts, FakeAdapter(
            '{"candidates":[{"subject_id":"user:u1","predicate":"likes_drink",'
            '"object":{"type":"literal","value":"龙井茶"}}]}'
        ))
        pipeline.process_message(**source)
        SummaryGenerator(summaries, FakeAdapter('{"summary":"用户在杭州喜欢龙井茶。"}')).generate(
            user_id="u1", session_id="s1", summary_id="sum1", start_seq=1, end_seq=2,
        )
        tasks.create_task(
            user_id="u1", task_id="task1", title="购买茶叶",
            payload={"constraints": ["本周完成"]},
        )
        events.record_event(
            user_id="u1", event_id="event1", occurred_at="2026-10-25T10:00:00+08:00",
            text="用户表达饮品偏好", phase="verified", participants=["user:u1"], sources=[source],
        )
        documents.add_document(
            user_id="u1", document_id="doc1", content="龙井茶冲泡指南", metadata={"uri": "docs/tea"},
        )
        graph.add_entity(
            user_id="u1", entity_id="person", entity_type="person", canonical_name="用户",
        )
        graph.add_entity(
            user_id="u1", entity_id="city", entity_type="city", canonical_name="杭州",
        )
        graph.add_relation(
            user_id="u1", relation_id="rel1", subject_entity_id="person",
            predicate="lives_in", object_entity_id="city", sources=[source],
        )
        jobs.enqueue(
            user_id="u1", job_id="job1", kind="refresh_index", subject_key="u1",
            payload={"document_id": "doc1"},
        )

        receipt = MemoryForgetter(store).forget_user(user_id="u1", reason="用户要求删除")
        expected = {
            "messages", "message_sequences", "facts", "fact_sources",
            "memory_pipeline_runs", "summaries", "tasks", "events", "event_sources",
            "external_documents", "external_document_chunks", "entities", "entity_aliases",
            "entity_relations", "entity_relation_sources", "memory_jobs",
        }
        self.assertTrue(expected <= set(receipt["deleted_rows"]))
        for table in receipt["deleted_rows"]:
            self.assertEqual(0, store.conn.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE user_id = ?', ("u1",),
            ).fetchone()[0], table)
        self.assertEqual([], store.conn.execute("PRAGMA foreign_key_check").fetchall())
        with self.assertRaises(ValueError):
            tasks.create_task(user_id="u1", task_id="after", title="不应复活")
        with self.assertRaises(ValueError):
            pipeline.process_message(**source)


if __name__ == "__main__":
    unittest.main()
