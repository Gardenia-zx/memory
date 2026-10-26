"""事实版本更新、幂等重试和事务回滚的回归测试。"""

import sqlite3
import unittest

from src.fact_store import FactStore
from src.message_store import MessageStore


class FactStoreUpdateTests(unittest.TestCase):
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
        self.update_request = {
            "user_id": "u1",
            "fact_id": "f1",
            "new_version": 2,
            "object": {"type": "literal", "value": "茶"},
            "sources": [self.sources[1]],
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
        return tuple(map(tuple, facts)), tuple(map(tuple, sources))

    def test_update_preserves_history_and_inherits_scope_and_validity(self):
        first = self.facts.add_fact(
            **self.request,
            scope="session",
            session_id="current-chat",
            valid_from="2000-01-01T08:00:00+08:00",
            valid_to="2100-01-01T08:00:00+08:00",
        )

        second = self.facts.update_fact(
            **{
                **self.update_request,
                "sources": [self.sources[1], self.sources[1]],
            }
        )

        self.assertEqual(second["version"], 2)
        self.assertEqual(second["status"], "active")
        self.assertEqual(second["object"], self.update_request["object"])
        self.assertEqual(second["sources"], [self.sources[1]])
        for field in (
            "user_id", "fact_id", "subject_id", "predicate",
            "scope", "session_id", "valid_from", "valid_to",
        ):
            with self.subTest(field=field):
                self.assertEqual(second[field], first[field])

        old = self.facts.get_fact(user_id="u1", fact_id="f1", version=1)
        self.assertEqual(old, {**first, "status": "superseded"})
        self.assertEqual(
            self.facts.get_latest_fact(user_id="u1", fact_id="f1"), second,
        )
        self.assertEqual(
            self.facts.list_facts(
                user_id="u1",
                session_id="current-chat",
                as_of="2030-01-01T00:00:00+00:00",
            ),
            [second],
        )
        self.assertEqual(
            self.facts.list_facts(
                user_id="u1", as_of="2030-01-01T00:00:00+00:00",
            ),
            [],
        )

    def test_retry_returns_existing_version_without_validating_new_body(self):
        self.facts.add_fact(**self.request)
        second = self.facts.update_fact(**self.update_request)
        before = self._snapshot()

        retried = self.facts.update_fact(
            **{**self.update_request, "object": None, "sources": []}
        )

        self.assertEqual(retried, second)
        self.assertEqual(self._snapshot(), before)

    def test_update_requires_a_strict_integer_version_at_least_two(self):
        self.facts.add_fact(**self.request)
        before = self._snapshot()
        for version in (None, True, False, "2", 2.0, 1, 0, -1):
            with self.subTest(version=version):
                with self.assertRaises(ValueError):
                    self.facts.update_fact(
                        **{**self.update_request, "new_version": version}
                    )
                self.assertEqual(self._snapshot(), before)

    def test_missing_or_skipped_version_is_rejected(self):
        self.facts.add_fact(**self.request)
        before = self._snapshot()
        for changes in (
            {"fact_id": "missing"},
            {"user_id": "u2"},
            {"new_version": 3},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.facts.update_fact(**{**self.update_request, **changes})
                self.assertEqual(self._snapshot(), before)

    def test_inactive_latest_version_is_rejected(self):
        for status in ("retracted", "superseded"):
            with self.subTest(status=status):
                fact_id = f"f-{status}"
                self.facts.add_fact(
                    **{**self.request, "fact_id": fact_id, "status": status}
                )
                before = self._snapshot()
                with self.assertRaises(ValueError):
                    self.facts.update_fact(
                        **{**self.update_request, "fact_id": fact_id}
                    )
                self.assertEqual(self._snapshot(), before)

    def test_invalid_new_body_rolls_back_old_status(self):
        self.facts.add_fact(**self.request)
        before = self._snapshot()
        for changes in (
            {"object": None},
            {"object": {"type": "literal", "value": float("nan")}},
            {"sources": []},
            {"sources": [{**self.sources[1], "message_id": "missing"}]},
            {"sources": [{**self.sources[1], "user_id": "u2"}]},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.facts.update_fact(**{**self.update_request, **changes})
                self.assertEqual(self._snapshot(), before)

    def test_second_source_insert_failure_rolls_back_entire_update(self):
        self.facts.add_fact(**self.request)
        with self.messages.conn:
            self.messages.conn.execute(
                """CREATE TRIGGER fail_second_update_source
                   BEFORE INSERT ON fact_sources
                   WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'f1'
                        AND NEW.version = 2 AND NEW.source_message_id = 'm3'
                   BEGIN
                       SELECT RAISE(ABORT, 'source insertion failed');
                   END"""
            )
        before = self._snapshot()

        with self.assertRaises(sqlite3.IntegrityError):
            self.facts.update_fact(
                **{
                    **self.update_request,
                    "sources": [self.sources[1], self.sources[2]],
                }
            )

        self.assertEqual(self._snapshot(), before)
        self.assertIsNone(
            self.facts.get_fact(user_id="u1", fact_id="f1", version=2)
        )

    def test_old_status_update_failure_leaves_no_committed_new_version(self):
        self.facts.add_fact(**self.request)
        with self.messages.conn:
            self.messages.conn.execute(
                """CREATE TRIGGER fail_old_status_update
                   BEFORE UPDATE OF status ON facts
                   WHEN OLD.user_id = 'u1' AND OLD.fact_id = 'f1'
                        AND OLD.version = 1 AND NEW.status = 'superseded'
                   BEGIN
                       SELECT RAISE(ABORT, 'status update failed');
                   END"""
            )
        before = self._snapshot()

        with self.assertRaises(sqlite3.IntegrityError):
            self.facts.update_fact(**self.update_request)

        self.assertEqual(self._snapshot(), before)
        self.assertIsNone(
            self.facts.get_fact(user_id="u1", fact_id="f1", version=2)
        )

    def test_add_returns_requested_version_and_retries_by_that_version(self):
        second = self.facts.add_fact(**self.request, version=2)

        self.assertEqual(second["version"], 2)
        self.assertEqual(second["status"], "active")
        self.assertIsNone(
            self.facts.get_fact(user_id="u1", fact_id="f1", version=1)
        )
        before = self._snapshot()
        retried = self.facts.add_fact(
            **{
                **self.request,
                "version": 2,
                "object": None,
                "sources": [],
                "status": "invalid",
            }
        )
        self.assertEqual(retried, second)
        self.assertEqual(self._snapshot(), before)


if __name__ == "__main__":
    unittest.main()
