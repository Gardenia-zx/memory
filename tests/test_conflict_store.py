"""争议记录的身份、精确来源、用户隔离及原子持久化测试。"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.conflict_store import ConflictStore
from src.fact_store import FactStore
from src.message_store import MessageStore


class ConflictStoreTests(unittest.TestCase):
    def setUp(self):
        self.messages = MessageStore(":memory:")
        self.addCleanup(self.messages.close)
        self.facts = FactStore(self.messages)
        self.conflicts = ConflictStore(self.facts)
        self.sources_by_user = self.seed(self.messages, self.facts)
        self.sources = self.sources_by_user["u1"]
        self.candidate = {
            "fact_id": "candidate-tea",
            "subject_id": "user:u1",
            "predicate": "prefers_drink",
            "object": {"type": "literal", "value": "茶"},
            "sources": [self.sources[1]],
        }
        self.request = {
            "user_id": "u1", "conflict_id": "c1",
            "candidate": self.candidate,
            "related_facts": [{"fact_id": "f1", "version": 1}],
            "reason": "旧值与候选不同，缺少纠错或多值依据",
        }

    @staticmethod
    def seed(messages, facts):
        sources_by_user = {}
        for user_id in ("u1", "u2"):
            sources = []
            for session_id, message_id in (("a", "m1"), ("b", "m1"), ("a", "m2")):
                source = {
                    "user_id": user_id, "session_id": session_id, "message_id": message_id,
                }
                messages.append_message(
                    **source, occurred_at="2026-10-21T10:00:00+08:00",
                    payload={"role": "user", "content": f"来源 {user_id}/{session_id}/{message_id}"},
                )
                sources.append(source)
            sources_by_user[user_id] = sources
            for fact_id, value in (("f1", "咖啡"), ("f2", "水")):
                facts.add_fact(
                    user_id=user_id, fact_id=fact_id, subject_id=f"user:{user_id}",
                    predicate="prefers_drink", object={"type": "literal", "value": value},
                    sources=[sources[0]],
                )
        return sources_by_user

    def base_snapshot(self):
        return tuple(
            tuple(map(tuple, self.messages.conn.execute(query).fetchall()))
            for query in (
                "SELECT * FROM messages ORDER BY user_id, session_id, message_id",
                "SELECT * FROM facts ORDER BY user_id, fact_id, version",
                """SELECT * FROM fact_sources
                   ORDER BY user_id, fact_id, version, source_session_id, source_message_id""",
                "SELECT * FROM fact_transitions ORDER BY user_id, new_fact_id",
            )
        )

    def snapshot(self):
        return self.base_snapshot() + tuple(
            tuple(map(tuple, self.messages.conn.execute(query).fetchall()))
            for query in (
                "SELECT * FROM conflicts ORDER BY user_id, conflict_id",
                """SELECT * FROM conflict_facts
                   ORDER BY user_id, conflict_id, fact_id, version""",
                """SELECT * FROM conflict_sources
                   ORDER BY user_id, conflict_id, source_session_id, source_message_id""",
            )
        )

    def test_record_returns_open_record_without_changing_facts_or_messages(self):
        self.assertIs(self.conflicts.conn, self.messages.conn)
        before = self.base_snapshot()
        before_request = deepcopy(self.request)
        earliest = datetime.now(timezone.utc)
        record = self.conflicts.record_conflict(**self.request)
        latest = datetime.now(timezone.utc)

        self.assertEqual(set(record), {
            "user_id", "conflict_id", "candidate", "related_facts",
            "reason", "status", "recorded_at",
        })
        self.assertEqual(record["user_id"], "u1")
        self.assertEqual(record["conflict_id"], "c1")
        self.assertEqual(record["candidate"], self.candidate)
        self.assertEqual(record["related_facts"], self.request["related_facts"])
        self.assertEqual(record["reason"], self.request["reason"])
        self.assertEqual(record["status"], "open")
        recorded_at = datetime.fromisoformat(record["recorded_at"])
        self.assertEqual(recorded_at.utcoffset(), timedelta(0))
        self.assertLessEqual(earliest, recorded_at)
        self.assertLessEqual(recorded_at, latest)
        self.assertEqual(self.conflicts.get_conflict(user_id="u1", conflict_id="c1"), record)
        self.assertEqual(self.request, before_request)
        self.assertEqual(self.base_snapshot(), before)

    def test_id_retry_returns_first_record_before_validating_any_business_fields(self):
        record = self.conflicts.record_conflict(**self.request)
        before = self.snapshot()
        changes = self.messages.conn.total_changes
        for request in (
            {"user_id": "u1", "conflict_id": "c1"},
            {**self.request, "candidate": None, "related_facts": [], "reason": None},
            {
                **self.request,
                "candidate": {"user_id": "u2", "value": float("nan")},
                "related_facts": "invalid", "reason": "与首次不同",
            },
        ):
            with self.subTest(request=request):
                self.assertEqual(self.conflicts.record_conflict(**request), record)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(self.messages.conn.total_changes, changes)

    def test_same_body_with_a_new_id_creates_an_independent_record(self):
        first = self.conflicts.record_conflict(**self.request)
        second = self.conflicts.record_conflict(**{**self.request, "conflict_id": "c2"})
        self.assertEqual(second["candidate"], first["candidate"])
        self.assertEqual(second["related_facts"], first["related_facts"])
        self.assertEqual(second["conflict_id"], "c2")
        self.assertEqual(len(self.conflicts.list_conflicts(user_id="u1")), 2)

    def test_sources_and_fact_references_deduplicate_and_sort_by_complete_identity(self):
        candidate = {
            **self.candidate,
            "sources": [self.sources[1], self.sources[2], self.sources[0], self.sources[1],
                        {**self.sources[0], "extra": "忽略来源元字段"}],
        }
        record = self.conflicts.record_conflict(**{
            **self.request, "candidate": candidate,
            "related_facts": [
                {"fact_id": "f2", "version": 1},
                {"fact_id": "f1", "version": 1},
                {"fact_id": "f2", "version": 1},
            ],
        })
        self.assertEqual(record["candidate"]["sources"], [
            self.sources[0], self.sources[2], self.sources[1],
        ])
        self.assertEqual(record["related_facts"], [
            {"fact_id": "f1", "version": 1}, {"fact_id": "f2", "version": 1},
        ])
        self.assertEqual(self.messages.conn.execute("SELECT COUNT(*) FROM conflict_sources").fetchone()[0], 3)
        self.assertEqual(self.messages.conn.execute("SELECT COUNT(*) FROM conflict_facts").fetchone()[0], 2)

    def test_related_facts_preserve_exact_historical_versions_without_following_latest(self):
        self.facts.update_fact(
            user_id="u1", fact_id="f1", new_version=2,
            object={"type": "literal", "value": "果汁"}, sources=[self.sources[1]],
        )
        self.facts.retract_fact(user_id="u1", fact_id="f1", expected_version=2)
        before = self.base_snapshot()
        record = self.conflicts.record_conflict(**{
            **self.request,
            "related_facts": [{"fact_id": "f1", "version": 2}, {"fact_id": "f1", "version": 1}],
        })
        self.assertEqual(record["related_facts"], [
            {"fact_id": "f1", "version": 1}, {"fact_id": "f1", "version": 2},
        ])
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="f1", version=1)["status"], "superseded")
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="f1", version=2)["status"], "retracted")
        self.assertEqual(self.base_snapshot(), before)

    def test_users_can_share_conflict_ids_without_reading_or_overwriting_each_other(self):
        first = self.conflicts.record_conflict(**self.request)
        self.assertIsNone(self.conflicts.get_conflict(user_id="u2", conflict_id="c1"))
        self.assertEqual(self.conflicts.list_conflicts(user_id="u2"), [])
        candidate = {
            **self.candidate, "subject_id": "user:u2",
            "object": {"type": "literal", "value": "牛奶"},
            "sources": [self.sources_by_user["u2"][1]],
        }
        second = self.conflicts.record_conflict(**{
            **self.request, "user_id": "u2", "candidate": candidate,
        })
        self.assertEqual(second["user_id"], "u2")
        self.assertEqual(second["candidate"], candidate)
        self.assertEqual(self.conflicts.list_conflicts(user_id="u1"), [first])
        self.assertEqual(self.conflicts.list_conflicts(user_id="u2"), [second])
        self.assertEqual(self.conflicts.get_conflict(user_id="u1", conflict_id="c1"), first)

    def test_identity_validation_applies_to_reads_writes_and_retries_without_stripping_ids(self):
        self.conflicts.record_conflict(**self.request)
        before = self.snapshot()
        for field in ("user_id", "conflict_id"):
            for value in (None, True, 1, "", " \t"):
                ids = {"user_id": "u1", "conflict_id": "c1", field: value}
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.conflicts.record_conflict(**ids)
                    with self.assertRaises(ValueError):
                        self.conflicts.get_conflict(**ids)
        for value in (None, True, "", " \t"):
            with self.subTest(user_id=value), self.assertRaises(ValueError):
                self.conflicts.list_conflicts(user_id=value)
        self.assertEqual(self.snapshot(), before)
        record = self.conflicts.record_conflict(**{
            **self.request, "conflict_id": " c2 ", "reason": "  保留原因原文  ",
        })
        self.assertEqual(record["conflict_id"], " c2 ")
        self.assertEqual(record["reason"], "  保留原因原文  ")
        self.assertIsNone(self.conflicts.get_conflict(user_id="u1", conflict_id="c2"))

    def test_candidate_is_an_opaque_strict_json_snapshot_not_a_relation_judgment(self):
        candidate = {
            "intent": "尚未识别的意图", "payload": {"中文": (1, True, None)},
            "sources": [self.sources[1]],
        }
        record = self.conflicts.record_conflict(**{**self.request, "candidate": candidate})
        self.assertEqual(record["candidate"]["intent"], candidate["intent"])
        self.assertEqual(record["candidate"]["payload"], {"中文": [1, True, None]})
        before = self.snapshot()
        for candidate in (
            None, [], "candidate", {},
            {"sources": [self.sources[0]], "user_id": "u1"},
            {"sources": [self.sources[0]], "data": float("nan")},
            {"sources": [self.sources[0]], "data": float("inf")},
            {"sources": [self.sources[0]], "data": {"not-json"}},
        ):
            with self.subTest(candidate=candidate):
                with self.assertRaises((TypeError, ValueError)):
                    self.conflicts.record_conflict(**{
                        **self.request, "conflict_id": "invalid", "candidate": candidate,
                    })
                self.assertEqual(self.snapshot(), before)

    def test_sources_must_be_real_messages_of_the_current_user(self):
        other_only = {"user_id": "u2", "session_id": "private", "message_id": "m9"}
        self.messages.append_message(
            **other_only, occurred_at="2026-10-21T10:00:00+08:00",
            payload={"role": "user", "content": "只存在于另一用户"},
        )
        before = self.snapshot()
        for sources in (
            None, [], "invalid", [None],
            [{"user_id": "u1", "message_id": "m1"}],
            [{**self.sources[0], "message_id": "missing"}],
            [self.sources_by_user["u2"][0]],
            [{**other_only, "user_id": "u1"}],
        ):
            with self.subTest(sources=sources):
                with self.assertRaises(ValueError):
                    self.conflicts.record_conflict(**{
                        **self.request, "candidate": {**self.candidate, "sources": sources},
                    })
                self.assertEqual(self.snapshot(), before)

    def test_related_fact_references_require_exact_existing_versions_in_this_user(self):
        self.facts.add_fact(
            user_id="u2", fact_id="private-fact", subject_id="user:u2", predicate="private",
            object={"type": "literal", "value": "值"}, sources=[self.sources_by_user["u2"][0]],
        )
        before = self.snapshot()
        for related in (
            None, [], "f1", [None], [{}],
            [{"fact_id": "f1", "version": True}],
            [{"fact_id": "f1", "version": 0}],
            [{"fact_id": "f1", "version": 1.0}],
            [{"fact_id": "f1", "version": 2}],
            [{"fact_id": "private-fact", "version": 1}],
            [{"fact_id": "f1", "version": 1, "user_id": "u2"}],
        ):
            with self.subTest(related=related):
                with self.assertRaises(ValueError):
                    self.conflicts.record_conflict(**{**self.request, "related_facts": related})
                self.assertEqual(self.snapshot(), before)

    def test_reason_is_required_and_cannot_create_a_partial_record(self):
        before = self.snapshot()
        for reason in (None, True, 1, "", " \t"):
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                self.conflicts.record_conflict(**{**self.request, "reason": reason})
            self.assertEqual(self.snapshot(), before)

    def test_mutating_inputs_or_returned_records_does_not_change_persisted_data(self):
        request = deepcopy(self.request)
        request["candidate"]["object"]["value"] = {"nested": ["茶"]}
        result = self.conflicts.record_conflict(**request)
        expected = deepcopy(result)
        request["candidate"]["object"]["value"]["nested"].append("新值")
        request["candidate"]["sources"][0]["message_id"] = "changed"
        request["related_facts"][0]["version"] = 999
        result["candidate"]["object"]["value"]["nested"].append("别的值")
        result["candidate"]["sources"][0]["session_id"] = "changed"
        result["related_facts"].clear()
        self.assertEqual(self.conflicts.get_conflict(user_id="u1", conflict_id="c1"), expected)
        exported = self.conflicts.list_conflicts(user_id="u1")
        exported[0]["candidate"]["object"]["value"]["nested"].clear()
        self.assertEqual(self.conflicts.get_conflict(user_id="u1", conflict_id="c1"), expected)

    def test_list_orders_by_recorded_time_then_conflict_id_and_unknown_reads_are_empty(self):
        self.assertIsNone(self.conflicts.get_conflict(user_id="u1", conflict_id="missing"))
        self.assertEqual(self.conflicts.list_conflicts(user_id="u1"), [])
        for conflict_id in ("c3", "c1", "c2"):
            self.conflicts.record_conflict(**{**self.request, "conflict_id": conflict_id})
        with self.messages.conn:
            self.messages.conn.execute(
                "UPDATE conflicts SET recorded_at = '2026-10-21T00:00:00+00:00' WHERE user_id = 'u1'"
            )
            self.messages.conn.execute(
                """UPDATE conflicts SET recorded_at = '2026-10-20T00:00:00+00:00'
                   WHERE user_id = 'u1' AND conflict_id = 'c3'"""
            )
        self.assertEqual(
            [record["conflict_id"] for record in self.conflicts.list_conflicts(user_id="u1")],
            ["c3", "c1", "c2"],
        )

    def test_sources_are_stored_only_in_the_relation_table_and_reconstructed_on_read(self):
        record = self.conflicts.record_conflict(**self.request)
        row = self.messages.conn.execute("SELECT candidate_json FROM conflicts").fetchone()
        stored_candidate = json.loads(row["candidate_json"])
        self.assertNotIn("sources", stored_candidate)
        expected_candidate = {key: value for key, value in self.candidate.items() if key != "sources"}
        self.assertEqual(stored_candidate, expected_candidate)
        source_rows = self.messages.conn.execute(
            "SELECT * FROM conflict_sources WHERE user_id = 'u1' AND conflict_id = 'c1'"
        ).fetchall()
        self.assertEqual(len(source_rows), 1)
        self.assertEqual(source_rows[0]["source_session_id"], self.sources[1]["session_id"])
        self.assertEqual(source_rows[0]["source_message_id"], self.sources[1]["message_id"])
        self.assertEqual(self.conflicts.get_conflict(user_id="u1", conflict_id="c1"), record)

    def test_failures_in_each_table_roll_back_parent_children_and_other_side_effects(self):
        request = {
            **self.request, "candidate": {**self.candidate, "sources": self.sources},
            "related_facts": [{"fact_id": "f1", "version": 1}, {"fact_id": "f2", "version": 1}],
        }
        for event in (
            "AFTER INSERT ON conflicts",
            "AFTER INSERT ON conflict_facts WHEN NEW.fact_id = 'f2'",
            "AFTER INSERT ON conflict_sources WHEN NEW.source_message_id = 'm2'",
        ):
            with self.subTest(event=event):
                with self.messages.conn:
                    self.messages.conn.execute(f"""
                        CREATE TRIGGER fail_conflict_record {event}
                        BEGIN
                            UPDATE facts SET predicate = 'trigger-side-effect'
                            WHERE user_id = 'u1' AND fact_id = 'f1' AND version = 1;
                            SELECT RAISE(FAIL, 'conflict write failed');
                        END
                    """)
                before = self.snapshot()
                try:
                    with self.assertRaisesRegex(sqlite3.IntegrityError, "conflict write failed"):
                        self.conflicts.record_conflict(**request)
                    self.assertEqual(self.snapshot(), before)
                    self.assertIsNone(self.conflicts.get_conflict(user_id="u1", conflict_id="c1"))
                finally:
                    with self.messages.conn:
                        self.messages.conn.execute("DROP TRIGGER fail_conflict_record")

    def test_ignored_inserts_in_any_table_are_errors_and_leave_no_partial_record(self):
        request = {
            **self.request, "candidate": {**self.candidate, "sources": self.sources},
            "related_facts": [{"fact_id": "f1", "version": 1}, {"fact_id": "f2", "version": 1}],
        }
        for event in (
            "BEFORE INSERT ON conflicts",
            "BEFORE INSERT ON conflict_facts WHEN NEW.fact_id = 'f2'",
            "BEFORE INSERT ON conflict_sources WHEN NEW.source_message_id = 'm2'",
        ):
            with self.subTest(event=event):
                with self.messages.conn:
                    self.messages.conn.execute(f"""
                        CREATE TRIGGER ignore_conflict_record {event}
                        BEGIN
                            SELECT RAISE(IGNORE);
                        END
                    """)
                before = self.snapshot()
                try:
                    with self.assertRaises(ValueError):
                        self.conflicts.record_conflict(**request)
                    self.assertEqual(self.snapshot(), before)
                finally:
                    with self.messages.conn:
                        self.messages.conn.execute("DROP TRIGGER ignore_conflict_record")

    def test_foreign_keys_prevent_orphan_fact_and_message_links(self):
        self.conflicts.record_conflict(**self.request)
        statements = (
            """INSERT INTO conflict_facts (user_id, conflict_id, fact_id, version)
               VALUES ('u1', 'missing-parent', 'f1', 1)""",
            """INSERT INTO conflict_facts (user_id, conflict_id, fact_id, version)
               VALUES ('u1', 'c1', 'missing-fact', 1)""",
            """INSERT INTO conflict_sources
               (user_id, conflict_id, source_session_id, source_message_id)
               VALUES ('u1', 'missing-parent', 'a', 'm1')""",
            """INSERT INTO conflict_sources
               (user_id, conflict_id, source_session_id, source_message_id)
               VALUES ('u1', 'c1', 'a', 'missing-message')""",
        )
        for statement in statements:
            before = self.snapshot()
            with self.subTest(statement=statement), self.assertRaises(sqlite3.IntegrityError):
                with self.messages.conn:
                    self.messages.conn.execute(statement)
            self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.messages.conn.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_existing_database_gains_tables_and_disk_reopen_retains_idempotency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            messages = MessageStore(path)
            try:
                facts = FactStore(messages)
                self.seed(messages, facts)
                original = facts.get_fact(user_id="u1", fact_id="f1", version=1)
                # 此时只有旧项目的表，ConflictStore 尚未初始化。
            finally:
                messages.close()

            messages = MessageStore(path)
            try:
                facts = FactStore(messages)
                conflicts = ConflictStore(facts)
                self.assertEqual(facts.get_fact(user_id="u1", fact_id="f1", version=1), original)
                record = conflicts.record_conflict(**self.request)
                self.assertEqual(facts.get_fact(user_id="u1", fact_id="f1", version=1), original)
            finally:
                messages.close()

            messages = MessageStore(path)
            try:
                facts = FactStore(messages)
                conflicts = ConflictStore(facts)
                changes = messages.conn.total_changes
                self.assertEqual(conflicts.get_conflict(user_id="u1", conflict_id="c1"), record)
                self.assertEqual(conflicts.list_conflicts(user_id="u1"), [record])
                self.assertEqual(conflicts.record_conflict(user_id="u1", conflict_id="c1"), record)
                self.assertEqual(messages.conn.total_changes, changes)
                self.assertEqual(facts.get_fact(user_id="u1", fact_id="f1", version=1), original)
            finally:
                messages.close()


if __name__ == "__main__":
    unittest.main()
