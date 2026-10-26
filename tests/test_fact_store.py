"""第四关：事实写入、版本读取、来源约束和事务回滚。"""

import copy
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.fact_store import FactStore
from src.message_store import MessageStore


class FactStoreTests(unittest.TestCase):
    def setUp(self):
        project_root = Path(__file__).resolve().parents[1]
        temp_dir = tempfile.TemporaryDirectory(
            prefix=".fact-store-test-", dir=project_root,
        )
        self.assertTrue(Path(temp_dir.name).resolve().is_relative_to(project_root))
        self.addCleanup(temp_dir.cleanup)
        self.db_path = Path(temp_dir.name) / "memory.sqlite"
        self.messages = MessageStore(self.db_path)
        self.addCleanup(lambda: self.messages.close())
        self.facts = FactStore(self.messages)
        source = self._seed_source("m1")
        self.request = {
            "user_id": "u1",
            "fact_id": "f1",
            "subject_id": "user:u1",
            "predicate": "prefers_breakfast_drink",
            "object": {"type": "literal", "value": "咖啡"},
            "sources": [source],
        }

    @staticmethod
    def _source(message_id, *, user_id="u1", session_id="s1"):
        return {
            "user_id": user_id,
            "session_id": session_id,
            "message_id": message_id,
        }

    def _seed_source(self, message_id, *, user_id="u1", session_id="s1"):
        source = self._source(message_id, user_id=user_id, session_id=session_id)
        self.messages.append_message(
            **source,
            occurred_at="2026-10-21T10:00:00+08:00",
            payload={"role": "user", "content": "早餐喜欢喝咖啡"},
        )
        return source

    def _assert_no_facts(self):
        for table in ("facts", "fact_sources"):
            count = self.messages.conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            self.assertEqual(count, 0, table)

    def test_initialization_reuses_connection_and_can_repeat(self):
        self.assertIs(self.facts.conn, self.messages.conn)
        self.assertEqual(self.messages.conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        FactStore(self.messages)
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="missing"))
        self._assert_no_facts()

    def test_add_get_roundtrip_sorts_unique_sources_and_normalizes_times(self):
        second = self._seed_source("m2", session_id="s2")
        earlier = self._seed_source("a", session_id="s1")
        sources = [second, self.request["sources"][0], earlier, second]
        before = datetime.now(timezone.utc)

        saved = self.facts.add_fact(
            **{
                **self.request,
                "sources": sources,
                "valid_from": "2026-10-21T10:00:00+08:00",
                "valid_to": "2026-10-21T03:00:00+00:00",
            }
        )

        self.assertEqual(saved["fact_id"], "f1")
        self.assertEqual(saved["user_id"], "u1")
        self.assertEqual(saved["subject_id"], self.request["subject_id"])
        self.assertEqual(saved["predicate"], self.request["predicate"])
        self.assertEqual(saved["version"], 1)
        self.assertEqual(saved["status"], "active")
        self.assertEqual(saved["object"], self.request["object"])
        self.assertNotIn("object_json", saved)
        self.assertEqual(saved["scope"], "user")
        self.assertIsNone(saved["session_id"])
        self.assertEqual(saved["valid_from"], "2026-10-21T02:00:00+00:00")
        self.assertEqual(saved["valid_to"], "2026-10-21T03:00:00+00:00")
        self.assertEqual(saved["sources"], [earlier, self.request["sources"][0], second])
        recorded = datetime.fromisoformat(saved["recorded_at"])
        self.assertEqual(recorded.utcoffset().total_seconds(), 0)
        self.assertGreaterEqual(recorded, before)
        self.assertLessEqual(recorded, datetime.now(timezone.utc))
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="f1"), saved)

    def test_get_uses_user_fact_and_version(self):
        first = self.facts.add_fact(**self.request)
        second_source = self._seed_source("m2", session_id="s2")
        with self.messages.conn:
            self.messages.conn.execute(
                """INSERT INTO facts (
                       user_id, fact_id, version, subject_id, predicate, object_json,
                       scope, session_id, valid_from, valid_to, status, recorded_at
                   ) SELECT user_id, fact_id, 2, subject_id, predicate, ?,
                            scope, session_id, valid_from, valid_to, status, recorded_at
                     FROM facts WHERE user_id = ? AND fact_id = ? AND version = 1""",
                (json.dumps({"type": "literal", "value": "茶"}), "u1", "f1"),
            )
            self.messages.conn.execute(
                """INSERT INTO fact_sources (
                       user_id, fact_id, version, source_session_id, source_message_id
                   ) VALUES (?, ?, 2, ?, ?)""",
                ("u1", "f1", second_source["session_id"], second_source["message_id"]),
            )

        second = self.facts.get_fact(user_id="u1", fact_id="f1", version=2)
        self.assertEqual(second["version"], 2)
        self.assertEqual(second["object"], {"type": "literal", "value": "茶"})
        self.assertEqual(second["sources"], [second_source])
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="f1"), first)
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="f1", version=3))
        self.assertIsNone(self.facts.get_fact(user_id="u2", fact_id="f1"))
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="unknown"))

    def test_reopen_preserves_facts_and_idempotency(self):
        first = self.facts.add_fact(**self.request)
        self.messages.close()
        self.messages = MessageStore(self.db_path)
        self.facts = FactStore(self.messages)

        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="f1"), first)
        self.assertEqual(self.facts.add_fact(**self.request), first)
        self.assertEqual(self.messages.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 1)
        self.assertEqual(self.messages.conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)

    def test_retry_returns_original_before_validating_changed_fields(self):
        first = self.facts.add_fact(**self.request)
        changes = {
            "subject_id": None,
            "predicate": "",
            "object": None,
            "sources": [],
            "scope": "invalid",
            "session_id": 123,
            "valid_from": "bad-time",
            "valid_to": False,
        }

        self.assertEqual(self.facts.add_fact(**{**self.request, **changes}), first)
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="f1"), first)
        self.assertEqual(self.messages.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 1)
        self.assertEqual(self.messages.conn.execute("SELECT COUNT(*) FROM fact_sources").fetchone()[0], 1)

    def test_distinct_fact_ids_and_users_remain_independent(self):
        first = self.facts.add_fact(**self.request)
        second = self.facts.add_fact(**{**self.request, "fact_id": "f2"})
        other_source = self._seed_source("m1", user_id="u2")
        other = self.facts.add_fact(**{
            **self.request,
            "user_id": "u2",
            "subject_id": "user:u2",
            "sources": [other_source],
        })

        self.assertEqual(first["object"], second["object"])
        self.assertEqual(second["fact_id"], "f2")
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="f1"), first)
        self.assertEqual(self.facts.get_fact(user_id="u2", fact_id="f1"), other)
        self.assertEqual(other["sources"], [other_source])

    def test_invalid_ids_raise_without_creating_records(self):
        for field in ("user_id", "fact_id", "subject_id", "predicate"):
            for value in (None, "", " \t", 123, True):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.facts.add_fact(**{**self.request, field: value})
                    self._assert_no_facts()
        for field in ("user_id", "fact_id"):
            for value in (None, "", " \t", 123, True):
                with self.subTest(get_field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.facts.get_fact(**{"user_id": "u1", "fact_id": "f1", field: value})

    def test_get_rejects_invalid_versions(self):
        for version in (0, -1, True, False, None, 1.5, "1"):
            with self.subTest(version=version):
                with self.assertRaises(ValueError):
                    self.facts.get_fact(user_id="u1", fact_id="f1", version=version)

    def test_invalid_literal_object_structure_is_rejected(self):
        for value in (None, "咖啡", [], {}, {"type": "literal"}, {"value": "咖啡"},
                      {"type": "entity", "value": "咖啡"}):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.facts.add_fact(**{**self.request, "object": value})
                self._assert_no_facts()

    def test_unserializable_and_nonfinite_values_leave_no_records(self):
        for value in ({"咖啡"}, b"coffee", float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    self.facts.add_fact(**{
                        **self.request, "object": {"type": "literal", "value": value},
                    })
                self._assert_no_facts()

    def test_invalid_scope_and_session_combinations_are_rejected(self):
        cases = [
            {"scope": "user", "session_id": value}
            for value in ("s1", "", 123, False)
        ]
        cases += [
            {"scope": "session", "session_id": value}
            for value in (None, "", " \t", 123, False)
        ]
        cases += [{"scope": scope} for scope in (None, "global", "", True)]
        for change in cases:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.facts.add_fact(**{**self.request, **change})
                self._assert_no_facts()

    def test_session_scope_is_distinct_from_source_session(self):
        saved = self.facts.add_fact(**{
            **self.request, "scope": "session", "session_id": "target-session",
        })

        self.assertEqual(saved["scope"], "session")
        self.assertEqual(saved["session_id"], "target-session")
        self.assertEqual(saved["sources"], self.request["sources"])
        self.assertIsNone(saved["valid_from"])
        self.assertIsNone(saved["valid_to"])

    def test_invalid_missing_or_foreign_sources_are_rejected(self):
        other = self._seed_source("other", user_id="u2")
        invalid = [None, {}, (), [], [None], [{}], [other],
                   [self._source("missing")],
                   [self._source("m1", session_id="unknown")]]
        for field in ("user_id", "session_id", "message_id"):
            for value in (None, "", " \t", 123, True):
                invalid.append([{**self.request["sources"][0], field: value}])
        for sources in invalid:
            with self.subTest(sources=sources):
                with self.assertRaises(ValueError):
                    self.facts.add_fact(**{**self.request, "sources": sources})
                self._assert_no_facts()

    def test_invalid_times_and_nonpositive_intervals_are_rejected(self):
        invalid_times = ("", "bad-time", "2026-02-30T10:00:00+08:00",
                         "2026-10-21T10:00:00", 123, True)
        for field in ("valid_from", "valid_to"):
            for value in invalid_times:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.facts.add_fact(**{**self.request, field: value})
                    self._assert_no_facts()
        for end in ("2026-10-21T02:00:00+00:00", "2026-10-21T01:59:59+00:00"):
            with self.subTest(end=end):
                with self.assertRaises(ValueError):
                    self.facts.add_fact(**{
                        **self.request,
                        "valid_from": "2026-10-21T10:00:00+08:00",
                        "valid_to": end,
                    })
                self._assert_no_facts()

    def test_unknown_or_one_sided_validity_is_preserved(self):
        cases = [
            (None, None),
            ("2026-10-21T10:00:00+08:00", None),
            (None, "2026-10-21T10:00:00+08:00"),
        ]
        for index, (start, end) in enumerate(cases):
            with self.subTest(start=start, end=end):
                saved = self.facts.add_fact(**{
                    **self.request, "fact_id": f"f{index}",
                    "valid_from": start, "valid_to": end,
                })
                expected = "2026-10-21T02:00:00+00:00"
                self.assertEqual(saved["valid_from"], expected if start else None)
                self.assertEqual(saved["valid_to"], expected if end else None)

    def test_mutating_inputs_or_returned_objects_does_not_change_storage(self):
        request = copy.deepcopy(self.request)
        request["object"]["value"] = {"drinks": ["咖啡", "茶"]}
        saved = self.facts.add_fact(**request)
        expected = copy.deepcopy(saved)

        request["object"]["value"]["drinks"].append("水")
        request["sources"][0]["message_id"] = "changed-input"
        saved["object"]["value"]["drinks"].clear()
        saved["sources"].clear()
        fetched = self.facts.get_fact(user_id="u1", fact_id="f1")
        self.assertEqual(fetched, expected)
        fetched["sources"][0]["session_id"] = "changed-output"
        fetched["object"]["value"]["drinks"].append("修改")
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="f1"), expected)

    def test_foreign_keys_reject_missing_messages_and_fact_versions(self):
        saved = self.facts.add_fact(**self.request)
        for values in (
            ("u1", "f1", 1, "s1", "missing"),
            ("u1", "f1", 2, "s1", "m1"),
        ):
            with self.subTest(values=values):
                with self.assertRaises(sqlite3.IntegrityError):
                    with self.messages.conn:
                        self.messages.conn.execute(
                            """INSERT INTO fact_sources (
                                   user_id, fact_id, version, source_session_id, source_message_id
                               ) VALUES (?, ?, ?, ?, ?)""",
                            values,
                        )
                self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="f1"), saved)

    def test_failure_on_second_source_rolls_back_fact_and_first_source(self):
        second = self._seed_source("m2", session_id="s2")
        before = {
            session: self.messages.list_messages(user_id="u1", session_id=session)
            for session in ("s1", "s2")
        }
        with self.messages.conn:
            self.messages.conn.execute(
                """CREATE TRIGGER reject_second_source
                   BEFORE INSERT ON fact_sources
                   WHEN NEW.source_message_id = 'm2'
                   BEGIN
                       SELECT RAISE(ABORT, 'source insertion failed');
                   END"""
            )

        with self.assertRaises(sqlite3.IntegrityError):
            self.facts.add_fact(**{
                **self.request, "sources": [self.request["sources"][0], second],
            })

        self._assert_no_facts()
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="f1"))
        for session, expected in before.items():
            self.assertEqual(
                self.messages.list_messages(user_id="u1", session_id=session), expected,
            )
        with self.messages.conn:
            self.messages.conn.execute("DROP TRIGGER reject_second_source")
        saved = self.facts.add_fact(**{
            **self.request, "sources": [self.request["sources"][0], second],
        })
        self.assertEqual(len(saved["sources"]), 2)


if __name__ == "__main__":
    unittest.main()
