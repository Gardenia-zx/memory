"""事实撤回、版本检查和事务回滚的回归测试。"""

import sqlite3
import unittest

from src.fact_store import FactStore
from src.message_store import MessageStore


class FactStoreRetractionTests(unittest.TestCase):
    def setUp(self):
        self.messages = MessageStore(":memory:")
        self.addCleanup(self.messages.close)
        self.facts = FactStore(self.messages)
        self.sources = []
        for message_id in ("m1", "m2", "m3"):
            source = {
                "user_id": "u1",
                "session_id": "history",
                "message_id": message_id,
            }
            self.messages.append_message(
                **source,
                occurred_at="2026-10-21T10:00:00+08:00",
                payload={"role": "user", "content": f"事实来源 {message_id}"},
            )
            self.sources.append(source)
        self.request = {
            "user_id": "u1",
            "fact_id": "f1",
            "subject_id": "user:u1",
            "predicate": "prefers_drink",
            "object": {"type": "literal", "value": "咖啡"},
            "sources": [self.sources[0]],
        }
        self.retraction_request = {
            "user_id": "u1",
            "fact_id": "f1",
            "expected_version": 1,
        }

    def _snapshot(self):
        facts = self.messages.conn.execute(
            "SELECT * FROM facts ORDER BY user_id, fact_id, version"
        ).fetchall()
        sources = self.messages.conn.execute(
            """SELECT * FROM fact_sources
               ORDER BY user_id, fact_id, version,
                        source_session_id, source_message_id"""
        ).fetchall()
        messages = self.messages.conn.execute(
            "SELECT * FROM messages ORDER BY user_id, session_id, message_id"
        ).fetchall()
        return (
            tuple(map(tuple, facts)),
            tuple(map(tuple, sources)),
            tuple(map(tuple, messages)),
        )

    def test_retraction_preserves_content_history_sources_and_messages(self):
        first = self.facts.add_fact(
            **self.request,
            scope="session",
            session_id="current-chat",
            valid_from="2000-01-01T08:00:00+08:00",
            valid_to="2100-01-01T08:00:00+08:00",
        )
        second = self.facts.update_fact(
            user_id="u1",
            fact_id="f1",
            new_version=2,
            object={"type": "literal", "value": "茶"},
            sources=[self.sources[1], self.sources[2]],
        )
        other_source = {**self.sources[0], "user_id": "u2"}
        self.messages.append_message(
            **other_source,
            occurred_at="2026-10-21T10:00:00+08:00",
            payload={"role": "user", "content": "另一用户的来源"},
        )
        other_user_fact = self.facts.add_fact(
            **{
                **self.request,
                "user_id": "u2",
                "subject_id": "user:u2",
                "sources": [other_source],
            }
        )
        other_fact = self.facts.add_fact(**{**self.request, "fact_id": "f2"})
        before = self._snapshot()

        retracted = self.facts.retract_fact(
            **{**self.retraction_request, "expected_version": 2}
        )

        self.assertEqual(retracted, {**second, "status": "retracted"})
        self.assertEqual(
            self.facts.get_fact(user_id="u1", fact_id="f1", version=2),
            retracted,
        )
        self.assertEqual(
            self.facts.get_latest_fact(user_id="u1", fact_id="f1"), retracted,
        )
        self.assertEqual(
            self.facts.get_fact(user_id="u1", fact_id="f1", version=1),
            {**first, "status": "superseded"},
        )
        self.assertEqual(
            self.facts.list_facts(
                user_id="u1",
                session_id="current-chat",
                as_of="2030-01-01T00:00:00+00:00",
            ),
            [other_fact],
        )
        self.assertEqual(self.facts.list_facts(user_id="u2"), [other_user_fact])
        after = self._snapshot()
        self.assertEqual(len(after[0]), len(before[0]))
        self.assertEqual(after[1:], before[1:])

    def test_repeated_retraction_returns_same_fact_without_changes(self):
        self.facts.add_fact(**self.request)
        retracted = self.facts.retract_fact(**self.retraction_request)
        before = self._snapshot()

        retried = self.facts.retract_fact(**self.retraction_request)

        self.assertEqual(retried, retracted)
        self.assertEqual(self._snapshot(), before)

    def test_invalid_and_unknown_requests_have_no_side_effects(self):
        self.facts.add_fact(**self.request)
        before = self._snapshot()
        invalid_changes = [
            {field: value}
            for field in ("user_id", "fact_id")
            for value in (None, "", " \t", 1, True)
        ]
        invalid_changes += [
            {"expected_version": value}
            for value in (None, True, False, "1", 1.0, 0, -1)
        ]
        invalid_changes += [
            {"fact_id": "missing"},
            {"user_id": "u2"},
            {"expected_version": 2},
        ]
        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.facts.retract_fact(
                        **{**self.retraction_request, **changes}
                    )
                self.assertEqual(self._snapshot(), before)

    def test_stale_version_cannot_retract_newer_active_fact(self):
        for old_status in ("superseded", "retracted"):
            with self.subTest(old_status=old_status):
                fact_id = f"f-{old_status}"
                self.facts.add_fact(**{**self.request, "fact_id": fact_id})
                if old_status == "superseded":
                    latest = self.facts.update_fact(
                        user_id="u1",
                        fact_id=fact_id,
                        new_version=2,
                        object={"type": "literal", "value": "茶"},
                        sources=[self.sources[1]],
                    )
                else:
                    self.facts.retract_fact(
                        **{**self.retraction_request, "fact_id": fact_id}
                    )
                    latest = self.facts.add_fact(
                        **{**self.request, "fact_id": fact_id, "version": 2}
                    )
                before = self._snapshot()

                with self.assertRaises(ValueError):
                    self.facts.retract_fact(
                        **{**self.retraction_request, "fact_id": fact_id}
                    )

                self.assertEqual(self._snapshot(), before)
                self.assertEqual(
                    self.facts.get_latest_fact(user_id="u1", fact_id=fact_id),
                    latest,
                )

    def test_superseded_latest_version_cannot_be_retracted(self):
        self.facts.add_fact(**self.request, status="superseded")
        before = self._snapshot()

        with self.assertRaises(ValueError):
            self.facts.retract_fact(**self.retraction_request)

        self.assertEqual(self._snapshot(), before)

    def test_retracted_fact_cannot_be_updated_to_a_new_version(self):
        self.facts.add_fact(**self.request)
        self.facts.retract_fact(**self.retraction_request)
        before = self._snapshot()

        with self.assertRaises(ValueError):
            self.facts.update_fact(
                user_id="u1",
                fact_id="f1",
                new_version=2,
                object={"type": "literal", "value": "茶"},
                sources=[self.sources[1]],
            )

        self.assertEqual(self._snapshot(), before)
        self.assertIsNone(
            self.facts.get_fact(user_id="u1", fact_id="f1", version=2)
        )

    def test_trigger_failure_propagates_and_rolls_back_all_changes(self):
        self.facts.add_fact(**self.request)
        self.facts.add_fact(**{**self.request, "fact_id": "f2"})
        with self.messages.conn:
            self.messages.conn.execute(
                """CREATE TRIGGER fail_retraction
                   AFTER UPDATE OF status ON facts
                   WHEN OLD.user_id = 'u1' AND OLD.fact_id = 'f1'
                        AND OLD.version = 1 AND NEW.status = 'retracted'
                   BEGIN
                       UPDATE facts SET predicate = 'changed-by-trigger'
                       WHERE user_id = 'u1' AND fact_id = 'f2' AND version = 1;
                       SELECT RAISE(FAIL, 'retraction failed');
                   END"""
            )
        before = self._snapshot()

        with self.assertRaisesRegex(sqlite3.IntegrityError, "retraction failed"):
            self.facts.retract_fact(**self.retraction_request)

        self.assertEqual(self._snapshot(), before)

    def test_ignored_status_update_is_rejected_without_changes(self):
        self.facts.add_fact(**self.request)
        with self.messages.conn:
            self.messages.conn.execute(
                """CREATE TRIGGER ignore_retraction
                   BEFORE UPDATE OF status ON facts
                   WHEN OLD.user_id = 'u1' AND OLD.fact_id = 'f1'
                        AND OLD.version = 1 AND NEW.status = 'retracted'
                   BEGIN
                       SELECT RAISE(IGNORE);
                   END"""
            )
        before = self._snapshot()

        with self.assertRaises(ValueError):
            self.facts.retract_fact(**self.retraction_request)

        self.assertEqual(self._snapshot(), before)


if __name__ == "__main__":
    unittest.main()
