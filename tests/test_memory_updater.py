"""确定性记忆决策的分发、幂等性和用户边界回归测试。"""

import sqlite3
import unittest

from src.fact_store import FactStore
from src.memory_updater import MemoryUpdater
from src.message_store import MessageStore


class MemoryUpdaterTests(unittest.TestCase):
    def setUp(self):
        self.messages = MessageStore(":memory:")
        self.addCleanup(self.messages.close)
        self.facts = FactStore(self.messages)
        self.updater = MemoryUpdater(self.facts)
        self.sources = []
        for user_id, message_id in (
            ("u1", "m1"), ("u1", "m2"), ("u1", "m3"), ("u2", "m1"),
        ):
            source = {
                "user_id": user_id,
                "session_id": "history",
                "message_id": message_id,
            }
            self.messages.append_message(
                **source,
                occurred_at="2026-10-21T10:00:00+08:00",
                payload={"role": "user", "content": f"来源 {user_id}/{message_id}"},
            )
            self.sources.append(source)
        self.add_decision = {
            "action": "ADD",
            "fact_id": "f1",
            "subject_id": "user:u1",
            "predicate": "prefers_drink",
            "object": {"type": "literal", "value": "咖啡"},
            "sources": [self.sources[0]],
        }
        self.update_decision = {
            "action": "UPDATE",
            "fact_id": "f1",
            "new_version": 2,
            "object": {"type": "literal", "value": "茶"},
            "sources": [self.sources[1]],
        }

    def test_add_defaults_scope_times_sources_and_owner_isolation(self):
        result = self.updater.apply(user_id="u1", decision=self.add_decision)
        first = self.facts.get_fact(user_id="u1", fact_id="f1")
        self.assertEqual(result, {"action": "ADD", "fact": first})
        self.assertEqual(first["version"], 1)
        self.assertEqual(first["status"], "active")
        self.assertEqual(first["scope"], "user")
        self.assertIsNone(first["session_id"])
        self.assertIsNone(first["valid_from"])
        self.assertIsNone(first["valid_to"])
        self.assertEqual(first["object"], self.add_decision["object"])
        self.assertEqual(first["sources"], [self.sources[0]])

        scoped_result = self.updater.apply(
            user_id="u1",
            decision={
                **self.add_decision,
                "fact_id": "scoped",
                "scope": "session",
                "session_id": "current-chat",
                "valid_from": "2000-01-01T08:00:00+08:00",
                "valid_to": "2100-01-01T08:00:00+08:00",
                "sources": [self.sources[2], self.sources[0], self.sources[2]],
            },
        )
        scoped = self.facts.get_fact(user_id="u1", fact_id="scoped")
        self.assertEqual(scoped_result, {"action": "ADD", "fact": scoped})
        self.assertEqual(scoped["scope"], "session")
        self.assertEqual(scoped["session_id"], "current-chat")
        self.assertEqual(scoped["valid_from"], "2000-01-01T00:00:00+00:00")
        self.assertEqual(scoped["valid_to"], "2100-01-01T00:00:00+00:00")
        self.assertEqual(scoped["sources"], [self.sources[0], self.sources[2]])
        self.assertEqual(self.facts.list_facts(user_id="u1"), [first])
        self.assertEqual(
            self.facts.list_facts(
                user_id="u1", session_id="current-chat",
                as_of="2030-01-01T00:00:00+00:00",
            ),
            [first, scoped],
        )

        other = self.updater.apply(
            user_id="u2",
            decision={
                **self.add_decision,
                "subject_id": "user:u2",
                "object": {"type": "literal", "value": "水"},
                "sources": [self.sources[3]],
            },
        )["fact"]
        self.assertEqual(other["user_id"], "u2")
        self.assertEqual(other, self.facts.get_fact(user_id="u2", fact_id="f1"))
        self.assertEqual(self.facts.list_facts(user_id="u2"), [other])
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="f1"), first)

    def test_update_and_delete_preserve_history_and_change_active_listing(self):
        first = self.updater.apply(user_id="u1", decision=self.add_decision)["fact"]
        updated = self.updater.apply(user_id="u1", decision=self.update_decision)
        second = self.facts.get_fact(user_id="u1", fact_id="f1", version=2)
        self.assertEqual(updated, {"action": "UPDATE", "fact": second})
        self.assertEqual(second["version"], 2)
        self.assertEqual(second["status"], "active")
        self.assertEqual(second["object"], self.update_decision["object"])
        self.assertEqual(second["sources"], self.update_decision["sources"])
        self.assertEqual(
            self.facts.get_fact(user_id="u1", fact_id="f1"),
            {**first, "status": "superseded"},
        )
        self.assertEqual(self.facts.list_facts(user_id="u1"), [second])

        decision = {"action": "DELETE", "fact_id": "f1", "expected_version": 2}
        deleted = self.updater.apply(user_id="u1", decision=decision)
        retracted = {**second, "status": "retracted"}
        self.assertEqual(deleted, {"action": "DELETE", "fact": retracted})
        self.assertEqual(self.facts.get_latest_fact(user_id="u1", fact_id="f1"), retracted)
        self.assertEqual(self.facts.list_facts(user_id="u1"), [])
        self.assertEqual(
            self.facts.get_fact(user_id="u1", fact_id="f1"),
            {**first, "status": "superseded"},
        )
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="f1", version=3))
        self.assertEqual(self.updater.apply(user_id="u1", decision=decision), deleted)

    def test_add_retry_returns_existing_version_with_missing_or_invalid_body(self):
        first = self.updater.apply(user_id="u1", decision=self.add_decision)["fact"]
        before = self.messages.conn.total_changes
        for decision in (
            {"action": "ADD", "fact_id": "f1"},
            {
                **self.add_decision, "subject_id": None, "predicate": None,
                "object": None, "sources": [], "scope": "invalid",
                "valid_from": "invalid",
            },
            {**self.add_decision, "object": self.update_decision["object"]},
        ):
            with self.subTest(decision=decision):
                self.assertEqual(
                    self.updater.apply(user_id="u1", decision=decision),
                    {"action": "ADD", "fact": first},
                )
                self.assertEqual(self.messages.conn.total_changes, before)
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="f1", version=2))

    def test_update_retry_returns_target_version_even_after_retraction(self):
        self.updater.apply(user_id="u1", decision=self.add_decision)
        second = self.updater.apply(user_id="u1", decision=self.update_decision)["fact"]
        before = self.messages.conn.total_changes
        for decision in (
            {"action": "UPDATE", "fact_id": "f1", "new_version": 2},
            {**self.update_decision, "object": None, "sources": []},
            {**self.update_decision, "object": self.add_decision["object"]},
        ):
            with self.subTest(decision=decision):
                self.assertEqual(
                    self.updater.apply(user_id="u1", decision=decision),
                    {"action": "UPDATE", "fact": second},
                )
                self.assertEqual(self.messages.conn.total_changes, before)

        retracted = self.updater.apply(
            user_id="u1",
            decision={"action": "DELETE", "fact_id": "f1", "expected_version": 2},
        )["fact"]
        before = self.messages.conn.total_changes
        self.assertEqual(
            self.updater.apply(
                user_id="u1",
                decision={"action": "UPDATE", "fact_id": "f1", "new_version": 2},
            ),
            {"action": "UPDATE", "fact": retracted},
        )
        self.assertEqual(self.messages.conn.total_changes, before)
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="f1", version=3))

    def test_noop_returns_none_without_writing(self):
        first = self.updater.apply(user_id="u1", decision=self.add_decision)["fact"]
        before = self.messages.conn.total_changes
        self.assertEqual(
            self.updater.apply(user_id="u1", decision={"action": "NOOP"}),
            {"action": "NOOP", "fact": None},
        )
        self.assertEqual(self.messages.conn.total_changes, before)
        self.assertEqual(self.facts.list_facts(user_id="u1"), [first])

    def test_invalid_owner_envelope_action_and_extra_fields_are_rejected(self):
        first = self.updater.apply(user_id="u1", decision=self.add_decision)["fact"]
        before = self.messages.conn.total_changes
        for owner in (None, True, 1, "", " \t"):
            with self.subTest(owner=owner), self.assertRaises(ValueError):
                self.updater.apply(user_id=owner, decision={"action": "NOOP"})
        for decision in (None, [], "NOOP", 1, {}, {"decision": {"action": "NOOP"}}):
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                self.updater.apply(user_id="u1", decision=decision)
        for action in (None, True, 1, [], {}, "", "add", "UPSERT"):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.updater.apply(user_id="u1", decision={"action": action})

        for decision in (
            self.add_decision,
            self.update_decision,
            {"action": "DELETE", "fact_id": "f1", "expected_version": 1},
            {"action": "NOOP"},
        ):
            for field in ("user_id", "version", "status", "unexpected"):
                with self.subTest(action=decision["action"], field=field):
                    with self.assertRaises(ValueError):
                        self.updater.apply(
                            user_id="u1", decision={**decision, field: "u2"},
                        )
        for decision in (
            {**self.add_decision, "new_version": 2},
            {**self.update_decision, "subject_id": "user:u2"},
            {"action": "DELETE", "fact_id": "f1", "expected_version": 1, "object": {}},
            {"action": "NOOP", "fact_id": "f1"},
        ):
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                self.updater.apply(user_id="u1", decision=decision)
        self.assertEqual(self.messages.conn.total_changes, before)
        self.assertEqual(self.facts.list_facts(user_id="u1"), [first])

    def test_missing_fields_and_invalid_new_body_leave_existing_fact_unchanged(self):
        first = self.updater.apply(user_id="u1", decision=self.add_decision)["fact"]
        invalid_decisions = []
        for decision, required_fields in (
            ({**self.add_decision, "fact_id": "new"},
             ("fact_id", "subject_id", "predicate", "object", "sources")),
            (self.update_decision, ("fact_id", "new_version", "object", "sources")),
            ({"action": "DELETE", "fact_id": "f1", "expected_version": 1},
             ("fact_id", "expected_version")),
        ):
            for field in required_fields:
                invalid_decisions.append({key: value for key, value in decision.items() if key != field})
        invalid_decisions.extend((
            {**self.add_decision, "fact_id": "new", "object": {}},
            {**self.update_decision, "new_version": True},
            {"action": "DELETE", "fact_id": "f1", "expected_version": True},
        ))
        for decision in invalid_decisions:
            with self.subTest(decision=decision):
                with self.assertRaises(ValueError):
                    self.updater.apply(user_id="u1", decision=decision)
                self.assertEqual(self.facts.get_latest_fact(user_id="u1", fact_id="f1"), first)
                self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="f1", version=2))
                self.assertIsNone(self.facts.get_latest_fact(user_id="u1", fact_id="new"))
                self.assertEqual(self.facts.list_facts(user_id="u1"), [first])

    def test_cross_user_sources_and_mutations_are_rejected(self):
        first = self.updater.apply(user_id="u1", decision=self.add_decision)["fact"]
        for decision in (
            {**self.add_decision, "fact_id": "cross-user", "sources": [self.sources[3]]},
            {**self.update_decision, "sources": [self.sources[3]]},
        ):
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                self.updater.apply(user_id="u1", decision=decision)
        for decision in (
            {**self.update_decision, "sources": [self.sources[3]]},
            {"action": "DELETE", "fact_id": "f1", "expected_version": 1},
        ):
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                self.updater.apply(user_id="u2", decision=decision)
        self.assertEqual(self.facts.get_latest_fact(user_id="u1", fact_id="f1"), first)
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="f1", version=2))
        self.assertIsNone(self.facts.get_latest_fact(user_id="u1", fact_id="cross-user"))
        self.assertEqual(self.facts.list_facts(user_id="u2"), [])

    def test_sqlite_failure_propagates_and_rolls_back_entire_update(self):
        first = self.updater.apply(user_id="u1", decision=self.add_decision)["fact"]
        with self.messages.conn:
            self.messages.conn.execute(
                """CREATE TRIGGER fail_updater_source
                   BEFORE INSERT ON fact_sources
                   WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'f1'
                        AND NEW.version = 2 AND NEW.source_message_id = 'm3'
                   BEGIN
                       SELECT RAISE(ABORT, 'source insertion failed');
                   END"""
            )
        with self.assertRaisesRegex(sqlite3.IntegrityError, "source insertion failed"):
            self.updater.apply(
                user_id="u1",
                decision={
                    **self.update_decision,
                    "sources": [self.sources[1], self.sources[2]],
                },
            )
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="f1"), first)
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="f1", version=2))
        self.assertEqual(self.facts.get_latest_fact(user_id="u1", fact_id="f1"), first)
        self.assertEqual(self.facts.list_facts(user_id="u1"), [first])


if __name__ == "__main__":
    unittest.main()
