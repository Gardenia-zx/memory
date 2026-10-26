import json
from pathlib import Path
import tempfile
import threading
import unittest

from src.conflict_store import ConflictStore
from src.entity_graph import EntityGraphStore
from src.fact_store import FactStore
from src.forgetter import MemoryForgetter
from src.job_store import JobStore
from src.message_store import MessageStore
from src.task_store import TaskStore


class ForgetterTests(unittest.TestCase):
    def setUp(self):
        self.messages = MessageStore(":memory:")
        self.addCleanup(self.messages.close)
        self.facts = FactStore(self.messages)
        self.conflicts = ConflictStore(self.facts)
        self.jobs = JobStore(self.messages)
        self.graph = EntityGraphStore(self.messages)
        self.forgetter = MemoryForgetter(self.messages)

    def seed(self, user_id):
        source = {"user_id": user_id, "session_id": "s", "message_id": "m"}
        self.messages.append_message(
            **source, occurred_at="2026-01-01T00:00:00+00:00",
            payload={"role": "user", "content": "Alice knows Bob"},
        )
        fact = self.facts.add_fact(
            user_id=user_id, fact_id="f", subject_id=f"user:{user_id}", predicate="likes",
            object={"type": "literal", "value": "coffee"}, sources=[source],
        )
        self.conflicts.record_conflict(
            user_id=user_id, conflict_id="c",
            candidate={
                "subject_id": f"user:{user_id}", "predicate": "likes",
                "object": {"type": "literal", "value": "tea"}, "sources": [source],
            },
            related_facts=[{"fact_id": "f", "version": 1}], reason="ambiguous",
        )
        self.conflicts.resolve_conflict(
            user_id=user_id, conflict_id="c", outcome="KEEP_EXISTING", reason="keep",
        )
        self.jobs.enqueue(
            user_id=user_id, job_id="j", kind="extract", subject_key="s", payload={},
        )
        self.graph.add_entity(
            user_id=user_id, entity_id="a", entity_type="person",
            canonical_name="Alice", attributes={},
        )
        self.graph.add_entity(
            user_id=user_id, entity_id="b", entity_type="person",
            canonical_name="Bob", attributes={},
        )
        self.graph.add_relation(
            user_id=user_id, relation_id="r", subject_entity_id="a", predicate="knows",
            object_entity_id="b", sources=[source],
        )
        return source, fact

    def test_forget_user_removes_every_current_derived_table_and_keeps_other_users(self):
        self.seed("u1")
        self.seed("u2")
        receipt = self.forgetter.forget_user(user_id="u1", reason="用户要求删除")
        self.assertEqual(receipt["reason"], "用户要求删除")
        self.assertGreater(receipt["deleted_rows"]["messages"], 0)
        for table in receipt["deleted_rows"]:
            remaining = self.messages.conn.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE user_id = ?', ("u1",),
            ).fetchone()[0]
            self.assertEqual(remaining, 0, table)
        self.assertIsNotNone(self.messages.get_message(user_id="u2", session_id="s", message_id="m"))
        self.assertIsNotNone(self.facts.get_fact(user_id="u2", fact_id="f"))
        self.assertIsNotNone(self.conflicts.get_resolution(user_id="u2", conflict_id="c"))

    def test_receipt_is_idempotent_and_tombstone_blocks_old_or_direct_replay(self):
        source, _ = self.seed("u1")
        claimed = self.jobs.claim_next(user_id="u1", lease_token="old-lease")
        first = self.forgetter.forget_user(user_id="u1", reason="first")
        self.assertEqual(self.forgetter.forget_user(user_id="u1", reason=None), first)
        with self.assertRaises(ValueError):
            self.messages.append_message(
                **source, occurred_at="2026-01-01T00:00:00+00:00",
                payload={"role": "user", "content": "replay"},
            )
        with self.assertRaises(ValueError):
            self.jobs.complete(
                user_id="u1", job_id="j", lease_token=claimed["lease_token"], result={},
            )
        self.assertTrue(self.messages.is_forgotten(user_id="u1"))

    def test_failure_rolls_back_tombstone_and_all_deletions(self):
        self.seed("u1")
        with self.messages.conn:
            self.messages.conn.execute("""
                CREATE TRIGGER stop_forget BEFORE DELETE ON facts
                WHEN OLD.user_id = 'u1'
                BEGIN SELECT RAISE(ABORT, 'stop'); END
            """)
        with self.assertRaises(Exception):
            self.forgetter.forget_user(user_id="u1", reason="must rollback")
        self.assertFalse(self.messages.is_forgotten(user_id="u1"))
        self.assertIsNotNone(self.messages.get_message(user_id="u1", session_id="s", message_id="m"))
        self.assertIsNotNone(self.facts.get_fact(user_id="u1", fact_id="f"))
        raw = self.messages.conn.execute(
            "SELECT COUNT(*) FROM conflict_resolutions WHERE user_id = 'u1'"
        ).fetchone()[0]
        self.assertEqual(raw, 1)

    def test_ignored_leaf_deletion_or_tombstone_write_cannot_report_forgetting_success(self):
        self.seed("u1")
        tasks = TaskStore(self.messages)
        task = tasks.create_task(user_id="u1", task_id="t", title="must delete")
        for event in ("DELETE ON tasks", "INSERT ON memory_forget_tombstones", "UPDATE ON memory_forget_tombstones"):
            with self.messages.conn:
                self.messages.conn.execute(f"""
                    CREATE TRIGGER ignore_forget BEFORE {event}
                    BEGIN SELECT RAISE(IGNORE); END
                """)
            try:
                with self.assertRaises(ValueError):
                    self.forgetter.forget_user(user_id="u1", reason="must rollback")
                self.assertIsNone(self.forgetter.get_receipt(user_id="u1"))
                self.assertEqual(tasks.get_task(user_id="u1", task_id="t"), task)
                self.assertIsNotNone(self.messages.get_message(user_id="u1", session_id="s", message_id="m"))
                self.assertEqual(self.messages.conn.execute("PRAGMA foreign_key_check").fetchall(), [])
            finally:
                with self.messages.conn:
                    self.messages.conn.execute("DROP TRIGGER ignore_forget")

    def test_two_connections_forgetting_same_user_return_one_receipt(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "forget.sqlite"
            seed = MessageStore(path)
            seed.append_message(
                user_id="u1", session_id="s", message_id="m",
                occurred_at="2026-01-01T00:00:00+00:00",
                payload={"role": "user", "content": "delete me"},
            )
            seed.close()
            barrier = threading.Barrier(2)
            results, errors = [], []

            def forget(reason):
                store = MessageStore(path)
                try:
                    barrier.wait()
                    results.append(MemoryForgetter(store).forget_user(
                        user_id="u1", reason=reason,
                    ))
                except Exception as error:  # pragma: no cover - asserted below
                    errors.append(error)
                finally:
                    store.close()

            threads = [
                threading.Thread(target=forget, args=("first",)),
                threading.Thread(target=forget, args=("second",)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual([], errors)
            self.assertEqual(2, len(results))
            self.assertEqual(results[0], results[1])
            check = MessageStore(path)
            try:
                self.assertTrue(check.is_forgotten(user_id="u1"))
                self.assertEqual(0, check.conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE user_id='u1'",
                ).fetchone()[0])
            finally:
                check.close()


if __name__ == "__main__":
    unittest.main()
