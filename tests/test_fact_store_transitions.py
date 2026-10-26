"""有效期迁移的区间、来源、持久幂等性及整事务回滚测试。"""

from copy import deepcopy
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.fact_store import FactStore
from src.message_store import MessageStore


class FactStoreTransitionTests(unittest.TestCase):
    def setUp(self):
        self.messages = MessageStore(":memory:")
        self.addCleanup(self.messages.close)
        self.facts = FactStore(self.messages)
        self.sources_by_user = {}
        for user_id in ("u1", "u2"):
            sources = []
            for message_id in ("m1", "m2", "m3"):
                source = {
                    "user_id": user_id, "session_id": "history", "message_id": message_id,
                }
                self.append_source(self.messages, source)
                sources.append(source)
            self.sources_by_user[user_id] = sources
        self.sources = self.sources_by_user["u1"]
        self.request = {
            "user_id": "u1",
            "fact_id": "home",
            "subject_id": "user:u1",
            "predicate": "lives_in",
            "object": {"type": "literal", "value": "北京"},
            "sources": [self.sources[0]],
            "valid_from": "2026-01-01T08:00:00+08:00",
            "valid_to": "2027-01-01T08:00:00+08:00",
        }
        self.transition_request = {
            "user_id": "u1",
            "fact_id": "home",
            "new_version": 2,
            "new_fact_id": "after",
            "effective_at": "2026-06-01T08:00:00+08:00",
            "object": {"type": "literal", "value": "杭州"},
            "sources": [self.sources[1], self.sources[2], self.sources[1]],
        }

    @staticmethod
    def append_source(messages, source):
        messages.append_message(
            **source,
            occurred_at="2026-10-21T10:00:00+08:00",
            payload={"role": "user", "content": f"来源 {source['user_id']}/{source['message_id']}"},
        )

    def snapshot(self):
        result = []
        for query in (
            "SELECT * FROM facts ORDER BY user_id, fact_id, version",
            """SELECT * FROM fact_sources
               ORDER BY user_id, fact_id, version, source_session_id, source_message_id""",
            "SELECT * FROM fact_transitions ORDER BY user_id, new_fact_id",
            "SELECT * FROM messages ORDER BY user_id, session_id, message_id",
        ):
            result.append(tuple(map(tuple, self.messages.conn.execute(query).fetchall())))
        return tuple(result)

    def retry_request(self, **changes):
        return {
            **self.transition_request,
            "fact_id": None,
            "new_version": None,
            "effective_at": None,
            "object": None,
            "sources": None,
            "valid_to": "invalid",
            **changes,
        }

    def test_transition_preserves_original_and_attributes_sources_to_each_version(self):
        first = self.facts.add_fact(**self.request)
        request_before = deepcopy(self.transition_request)
        result = self.facts.transition_fact(**self.transition_request)

        self.assertEqual(set(result), {"previous_fact", "fact"})
        closed, current = result["previous_fact"], result["fact"]
        self.assertEqual(self.transition_request, request_before)
        self.assertEqual(
            self.facts.get_fact(user_id="u1", fact_id="home", version=1),
            {**first, "status": "superseded"},
        )
        self.assertEqual(closed["fact_id"], "home")
        self.assertEqual(closed["version"], 2)
        self.assertEqual(closed["status"], "active")
        self.assertEqual(closed["object"], first["object"])
        self.assertEqual(closed["valid_from"], first["valid_from"])
        self.assertEqual(closed["valid_to"], "2026-06-01T00:00:00+00:00")
        self.assertEqual(closed["sources"], self.sources)
        self.assertEqual(current["fact_id"], "after")
        self.assertEqual(current["version"], 1)
        self.assertEqual(current["status"], "active")
        self.assertEqual(current["object"], self.transition_request["object"])
        self.assertEqual(current["valid_from"], closed["valid_to"])
        self.assertIsNone(current["valid_to"])
        self.assertEqual(current["sources"], self.sources[1:])
        for field in ("user_id", "subject_id", "predicate", "scope", "session_id"):
            with self.subTest(field=field):
                self.assertEqual(closed[field], first[field])
                self.assertEqual(current[field], first[field])
        self.assertEqual(self.facts.get_latest_fact(user_id="u1", fact_id="home"), closed)
        self.assertEqual(self.facts.get_latest_fact(user_id="u1", fact_id="after"), current)
        record = self.messages.conn.execute("SELECT * FROM fact_transitions").fetchone()
        self.assertEqual(record["user_id"], "u1")
        self.assertEqual(record["new_fact_id"], "after")
        self.assertEqual(record["previous_fact_id"], "home")
        self.assertEqual(record["previous_version"], 2)

    def test_queries_switch_exactly_at_boundary_and_default_new_end_is_open(self):
        self.facts.add_fact(**self.request)
        result = self.facts.transition_fact(**self.transition_request)
        for as_of, expected in (
            ("2025-12-31T23:59:59+00:00", []),
            ("2026-01-01T00:00:00+00:00", [result["previous_fact"]]),
            ("2026-05-31T23:59:59.999999+00:00", [result["previous_fact"]]),
            ("2026-06-01T00:00:00+00:00", [result["fact"]]),
            ("2026-06-01T00:00:00.000001+00:00", [result["fact"]]),
            ("2028-01-01T00:00:00+00:00", [result["fact"]]),
        ):
            with self.subTest(as_of=as_of):
                self.assertEqual(self.facts.list_facts(user_id="u1", as_of=as_of), expected)

    def test_session_scope_and_explicit_new_end_are_inherited_and_respected(self):
        self.facts.add_fact(**self.request, scope="session", session_id="current-chat")
        result = self.facts.transition_fact(
            **self.transition_request, valid_to="2028-01-01T08:00:00+08:00",
        )
        for fact in result.values():
            self.assertEqual(fact["scope"], "session")
            self.assertEqual(fact["session_id"], "current-chat")
        current = result["fact"]
        self.assertEqual(current["valid_to"], "2028-01-01T00:00:00+00:00")
        self.assertEqual(self.facts.list_facts(user_id="u1", as_of="2026-07-01T00:00:00+00:00"), [])
        self.assertEqual(
            self.facts.list_facts(user_id="u1", session_id="other-chat", as_of="2026-07-01T00:00:00+00:00"),
            [],
        )
        for as_of, expected in (
            ("2026-05-31T23:59:59+00:00", [result["previous_fact"]]),
            ("2026-06-01T00:00:00+00:00", [current]),
            ("2027-06-01T00:00:00+00:00", [current]),
            ("2028-01-01T00:00:00+00:00", []),
        ):
            with self.subTest(as_of=as_of):
                self.assertEqual(
                    self.facts.list_facts(user_id="u1", session_id="current-chat", as_of=as_of),
                    expected,
                )

    def test_transition_allows_unknown_old_time_boundaries(self):
        self.facts.add_fact(**{**self.request, "valid_from": None, "valid_to": None})
        result = self.facts.transition_fact(**self.transition_request)
        self.assertIsNone(result["previous_fact"]["valid_from"])
        self.assertEqual(result["previous_fact"]["valid_to"], "2026-06-01T00:00:00+00:00")
        self.assertIsNone(result["fact"]["valid_to"])

    def test_transition_uses_latest_version_content_and_sources(self):
        original = self.facts.add_fact(**self.request)
        latest = self.facts.update_fact(
            user_id="u1", fact_id="home", new_version=2,
            object={"type": "literal", "value": "天津"}, sources=[self.sources[1]],
        )
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.facts.transition_fact(**self.transition_request)
        self.assertEqual(self.snapshot(), before)
        result = self.facts.transition_fact(**{
            **self.transition_request, "new_version": 3, "sources": [self.sources[2]],
        })
        closed = result["previous_fact"]
        self.assertEqual(closed["version"], 3)
        self.assertEqual(closed["object"], latest["object"])
        self.assertEqual(closed["sources"], self.sources[1:])
        self.assertEqual(result["fact"]["sources"], [self.sources[2]])
        self.assertEqual(
            self.facts.get_fact(user_id="u1", fact_id="home", version=1),
            {**original, "status": "superseded"},
        )
        self.assertEqual(
            self.facts.get_fact(user_id="u1", fact_id="home", version=2),
            {**latest, "status": "superseded"},
        )

    def test_retry_uses_only_user_and_new_id_and_ignores_changed_or_missing_body(self):
        self.facts.add_fact(**self.request)
        result = self.facts.transition_fact(**self.transition_request)
        before = self.snapshot()
        changes = self.messages.conn.total_changes
        for request in (
            self.transition_request,
            self.retry_request(),
            self.retry_request(
                fact_id="somewhere-else", new_version=True, effective_at="invalid",
                object={"type": "literal", "value": "与首次不同"}, sources=[],
            ),
        ):
            with self.subTest(request=request):
                self.assertEqual(self.facts.transition_fact(**request), result)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(self.messages.conn.total_changes, changes)

    def test_retry_returns_fixed_retracted_versions_without_restoring_them(self):
        self.facts.add_fact(**self.request)
        self.facts.transition_fact(**self.transition_request)
        closed = self.facts.retract_fact(user_id="u1", fact_id="home", expected_version=2)
        current = self.facts.retract_fact(user_id="u1", fact_id="after", expected_version=1)
        before = self.snapshot()
        changes = self.messages.conn.total_changes
        self.assertEqual(
            self.facts.transition_fact(**self.retry_request()),
            {"previous_fact": closed, "fact": current},
        )
        self.assertEqual(closed["status"], "retracted")
        self.assertEqual(current["status"], "retracted")
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.messages.conn.total_changes, changes)

    def test_retry_reads_fixed_versions_after_later_corrections_and_retractions(self):
        self.facts.add_fact(**self.request)
        self.facts.transition_fact(**self.transition_request)
        for fact_id, new_version in (("home", 3), ("after", 2)):
            self.facts.update_fact(
                user_id="u1", fact_id=fact_id, new_version=new_version,
                object={"type": "literal", "value": "后来纠正的值"}, sources=[self.sources[2]],
            )
            self.facts.retract_fact(user_id="u1", fact_id=fact_id, expected_version=new_version)
        fixed_closed = self.facts.get_fact(user_id="u1", fact_id="home", version=2)
        fixed_current = self.facts.get_fact(user_id="u1", fact_id="after", version=1)
        before = self.snapshot()
        changes = self.messages.conn.total_changes
        result = self.facts.transition_fact(**self.retry_request())
        self.assertEqual(result, {"previous_fact": fixed_closed, "fact": fixed_current})
        self.assertEqual(result["previous_fact"]["status"], "superseded")
        self.assertEqual(result["fact"]["status"], "superseded")
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.messages.conn.total_changes, changes)
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="home", version=4))
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="after", version=3))

    def test_same_ids_are_isolated_by_user_and_sources_cannot_cross_users(self):
        self.facts.add_fact(**self.request)
        self.facts.add_fact(**{
            **self.request, "user_id": "u2", "subject_id": "user:u2",
            "object": {"type": "literal", "value": "东京"},
            "sources": [self.sources_by_user["u2"][0]],
        })
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.facts.transition_fact(**{
                **self.transition_request, "sources": [self.sources_by_user["u2"][1]],
            })
        self.assertEqual(self.snapshot(), before)
        first_user = self.facts.transition_fact(**self.transition_request)
        second_user = self.facts.transition_fact(**{
            **self.transition_request, "user_id": "u2",
            "object": {"type": "literal", "value": "大阪"},
            "sources": [self.sources_by_user["u2"][1]],
        })
        self.assertEqual(first_user["fact"]["object"]["value"], "杭州")
        self.assertEqual(second_user["fact"]["object"]["value"], "大阪")
        for result, user_id in ((first_user, "u1"), (second_user, "u2")):
            for fact in result.values():
                self.assertEqual(fact["user_id"], user_id)
        self.assertEqual(
            self.facts.transition_fact(**self.retry_request(user_id="u2")), second_user,
        )
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.facts.transition_fact(**{**self.transition_request, "user_id": "u3"})
        self.assertEqual(self.snapshot(), before)

    def test_existing_new_id_without_transition_record_is_rejected_at_any_version(self):
        self.facts.add_fact(**self.request)
        for version, status in ((1, "active"), (3, "retracted")):
            new_fact_id = f"occupied-{version}"
            self.facts.add_fact(**{
                **self.request, "fact_id": new_fact_id, "version": version, "status": status,
            })
            before = self.snapshot()
            with self.subTest(version=version, status=status), self.assertRaises(ValueError):
                self.facts.transition_fact(**{
                    **self.transition_request, "new_fact_id": new_fact_id,
                })
            self.assertEqual(self.snapshot(), before)
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.facts.transition_fact(**{**self.transition_request, "new_fact_id": "home"})
        self.assertEqual(self.snapshot(), before)

    def test_invalid_times_versions_and_missing_messages_never_leave_half_a_transition(self):
        self.facts.add_fact(**self.request)
        before = self.snapshot()
        invalid_changes = [
            {"effective_at": value} for value in (
                None, "invalid", "2026-06-01T00:00:00",
                "2025-12-31T00:00:00+00:00", "2026-01-01T00:00:00+00:00",
                "2027-01-01T00:00:00+00:00", "2028-01-01T00:00:00+00:00",
            )
        ]
        invalid_changes += [
            {"new_version": value} for value in (None, True, 2.0, 0, 1, 3)
        ]
        invalid_changes += [
            {"valid_to": "2026-06-01T00:00:00+00:00"},
            {"valid_to": "2026-05-31T00:00:00+00:00"},
            {"valid_to": "2026-07-01T00:00:00"},
            {"fact_id": "missing"},
            {"sources": [{**self.sources[1], "message_id": "missing"}]},
            {"sources": []},
            {"object": {"type": "literal", "value": float("nan")}},
        ]
        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.facts.transition_fact(**{**self.transition_request, **changes})
                self.assertEqual(self.snapshot(), before)
                self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="home", version=2))
                self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="after", version=1))

    def test_old_latest_version_must_be_active(self):
        for status in ("retracted", "superseded"):
            fact_id = f"old-{status}"
            self.facts.add_fact(**{**self.request, "fact_id": fact_id, "status": status})
            before = self.snapshot()
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.facts.transition_fact(**{
                    **self.transition_request, "fact_id": fact_id,
                    "new_fact_id": f"new-{status}",
                })
            self.assertEqual(self.snapshot(), before)

    def test_retry_identity_fields_still_require_valid_ids(self):
        self.facts.add_fact(**self.request)
        self.facts.transition_fact(**self.transition_request)
        before = self.snapshot()
        for field in ("user_id", "new_fact_id"):
            for value in (None, True, "", " \t"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.facts.transition_fact(**self.retry_request(**{field: value}))
                self.assertEqual(self.snapshot(), before)

    def test_failures_at_all_write_stages_roll_back_facts_sources_and_transition_record(self):
        self.facts.add_fact(**self.request)
        self.facts.add_fact(**{**self.request, "fact_id": "unrelated"})
        stages = (
            ("old_status", """AFTER UPDATE OF status ON facts
                WHEN OLD.user_id = 'u1' AND OLD.fact_id = 'home' AND OLD.version = 1
                     AND NEW.status = 'superseded'"""),
            ("closed_version", """AFTER INSERT ON facts
                WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'home' AND NEW.version = 2"""),
            ("new_fact", """AFTER INSERT ON facts
                WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'after' AND NEW.version = 1"""),
            ("second_new_source", """AFTER INSERT ON fact_sources
                WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'after'
                     AND NEW.source_message_id = 'm3'"""),
            ("transition_record", "AFTER INSERT ON fact_transitions"),
        )
        for stage, event in stages:
            with self.subTest(stage=stage):
                with self.messages.conn:
                    self.messages.conn.execute(f"""
                        CREATE TRIGGER fail_transition {event}
                        BEGIN
                            UPDATE facts SET predicate = 'changed-by-trigger'
                            WHERE user_id = 'u1' AND fact_id = 'unrelated' AND version = 1;
                            SELECT RAISE(FAIL, 'transition stage failed');
                        END
                    """)
                before = self.snapshot()
                try:
                    with self.assertRaisesRegex(sqlite3.IntegrityError, "transition stage failed"):
                        self.facts.transition_fact(**self.transition_request)
                    self.assertEqual(self.snapshot(), before)
                finally:
                    with self.messages.conn:
                        self.messages.conn.execute("DROP TRIGGER fail_transition")

    def test_ignored_old_status_update_is_rejected_and_rolls_back(self):
        self.facts.add_fact(**self.request)
        with self.messages.conn:
            self.messages.conn.execute("""
                CREATE TRIGGER ignore_transition
                BEFORE UPDATE OF status ON facts
                WHEN OLD.user_id = 'u1' AND OLD.fact_id = 'home' AND OLD.version = 1
                     AND NEW.status = 'superseded'
                BEGIN
                    SELECT RAISE(IGNORE);
                END
            """)
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.facts.transition_fact(**self.transition_request)
        self.assertEqual(self.snapshot(), before)

    def test_ignored_transition_record_insert_rolls_back_both_fact_writes(self):
        first = self.facts.add_fact(**self.request)
        with self.messages.conn:
            self.messages.conn.execute("""
                CREATE TRIGGER ignore_transition_record
                BEFORE INSERT ON fact_transitions
                BEGIN
                    SELECT RAISE(IGNORE);
                END
            """)
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.facts.transition_fact(**self.transition_request)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.facts.get_latest_fact(user_id="u1", fact_id="home"), first)
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="home", version=2))
        self.assertIsNone(self.facts.get_latest_fact(user_id="u1", fact_id="after"))

    def test_transition_record_has_foreign_keys_to_both_fixed_fact_versions(self):
        self.facts.add_fact(**self.request)
        self.facts.transition_fact(**self.transition_request)
        references = self.messages.conn.execute("PRAGMA foreign_key_list(fact_transitions)").fetchall()
        fact_links = [row for row in references if row["table"] == "facts"]
        self.assertEqual(len({row["id"] for row in fact_links}), 2)
        columns = {(row["from"], row["to"]) for row in fact_links}
        self.assertIn(("previous_fact_id", "fact_id"), columns)
        self.assertIn(("previous_version", "version"), columns)
        self.assertIn(("new_fact_id", "fact_id"), columns)
        self.assertEqual(self.messages.conn.execute("PRAGMA foreign_key_check").fetchall(), [])
        before = self.snapshot()
        with self.assertRaises(sqlite3.IntegrityError):
            with self.messages.conn:
                self.messages.conn.execute("""
                    INSERT INTO fact_transitions
                    (user_id, new_fact_id, previous_fact_id, previous_version)
                    VALUES ('u1', 'after', 'home', 2)
                """)
        self.assertEqual(self.snapshot(), before)

    def test_existing_schema_upgrades_and_reopened_disk_store_remembers_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            messages = MessageStore(path)
            try:
                facts = FactStore(messages)
                for source in self.sources:
                    self.append_source(messages, source)
                first = facts.add_fact(**self.request)
                # 模拟迁移功能加入前的持久库，保留原 messages/facts/fact_sources。
                with messages.conn:
                    messages.conn.execute("DROP TABLE fact_transitions")
            finally:
                messages.close()

            messages = MessageStore(path)
            try:
                facts = FactStore(messages)
                self.assertEqual(facts.get_fact(user_id="u1", fact_id="home"), first)
                result = facts.transition_fact(**self.transition_request)
            finally:
                messages.close()

            messages = MessageStore(path)
            try:
                facts = FactStore(messages)
                changes = messages.conn.total_changes
                self.assertEqual(facts.transition_fact(**self.retry_request()), result)
                self.assertEqual(messages.conn.total_changes, changes)
                count = messages.conn.execute("SELECT COUNT(*) FROM fact_transitions").fetchone()[0]
                self.assertEqual(count, 1)
                self.assertEqual(
                    facts.get_fact(user_id="u1", fact_id="home", version=1),
                    {**first, "status": "superseded"},
                )
                self.assertIsNone(facts.get_fact(user_id="u1", fact_id="home", version=3))
                self.assertIsNone(facts.get_fact(user_id="u1", fact_id="after", version=2))
            finally:
                messages.close()


if __name__ == "__main__":
    unittest.main()
