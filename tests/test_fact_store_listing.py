"""第四关：事实列表的用户隔离、会话作用域和有效期过滤。"""

import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.fact_store import FactStore
from src.message_store import MessageStore


class FactStoreListingTests(unittest.TestCase):
    def setUp(self):
        project_root = Path(__file__).resolve().parents[1]
        temp_dir = tempfile.TemporaryDirectory(
            prefix=".fact-listing-test-", dir=project_root,
        )
        self.assertTrue(Path(temp_dir.name).resolve().is_relative_to(project_root))
        self.addCleanup(temp_dir.cleanup)
        self.messages = MessageStore(Path(temp_dir.name) / "memory.sqlite")
        self.addCleanup(self.messages.close)
        self.facts = FactStore(self.messages)
        self.as_of = "2026-10-21T10:00:00+00:00"

    def _seed_source(self, *, user_id="u1", session_id="history", message_id="m1"):
        source = {
            "user_id": user_id,
            "session_id": session_id,
            "message_id": message_id,
        }
        self.messages.append_message(
            **source,
            occurred_at="2026-10-21T09:00:00+00:00",
            payload={"role": "user", "content": "早餐喜欢喝咖啡"},
        )
        return source

    def _add(self, fact_id, *, user_id="u1", source_session_id="history", **changes):
        source = self._seed_source(user_id=user_id, session_id=source_session_id)
        request = {
            "user_id": user_id,
            "fact_id": fact_id,
            "subject_id": f"user:{user_id}",
            "predicate": "likes",
            "object": {"type": "literal", "value": "咖啡"},
            "sources": [source],
        }
        request.update(changes)
        return self.facts.add_fact(**request)

    def test_empty_store_and_unknown_user_return_empty(self):
        self.assertEqual(
            self.facts.list_facts(user_id="u1", as_of=self.as_of), [],
        )
        self._add("f1")
        self.assertEqual(
            self.facts.list_facts(user_id="unknown", as_of=self.as_of), [],
        )

    def test_same_fact_id_belongs_only_to_requested_user(self):
        first = self._add("shared")
        second = self._add("shared", user_id="u2")
        self._add("shared", user_id="u10")

        self.assertEqual(
            self.facts.list_facts(user_id="u1", as_of=self.as_of), [first],
        )
        self.assertEqual(
            self.facts.list_facts(user_id="u2", as_of=self.as_of), [second],
        )

    def test_session_queries_include_user_scope_and_only_matching_session(self):
        second_session = self._add("c-session", scope="session", session_id="s2")
        user_scope = self._add("a-user")
        first_session = self._add("b-session", scope="session", session_id="s1")

        for session_id, expected in (
            (None, [user_scope]),
            ("s1", [user_scope, first_session]),
            ("s2", [user_scope, second_session]),
            ("unknown", [user_scope]),
        ):
            with self.subTest(session_id=session_id):
                self.assertEqual(
                    self.facts.list_facts(
                        user_id="u1", session_id=session_id, as_of=self.as_of,
                    ),
                    expected,
                )

    def test_applicable_session_is_independent_of_source_session(self):
        saved = self._add(
            "session-fact", scope="session", session_id="target",
            source_session_id="source-only",
        )

        self.assertEqual(
            self.facts.list_facts(user_id="u1", session_id="target", as_of=self.as_of),
            [saved],
        )
        self.assertEqual(
            self.facts.list_facts(
                user_id="u1", session_id="source-only", as_of=self.as_of,
            ),
            [],
        )

    def test_subject_and_predicate_filters_are_exact_and_can_be_combined(self):
        first = self._add("a")
        other_subject = self._add("b", subject_id="person:other")
        other_predicate = self._add("c", predicate="dislikes")
        self._add("d", subject_id="user:u10", predicate="likes_more")

        for filters, expected in (
            ({"subject_id": "user:u1"}, [first, other_predicate]),
            ({"predicate": "likes"}, [first, other_subject]),
            ({"subject_id": "user:u1", "predicate": "likes"}, [first]),
            ({"subject_id": "missing"}, []),
            ({"predicate": "like"}, []),
        ):
            with self.subTest(filters=filters):
                self.assertEqual(
                    self.facts.list_facts(user_id="u1", as_of=self.as_of, **filters),
                    expected,
                )

    def test_only_active_versions_are_returned_in_fact_and_version_order(self):
        self._add("z")
        self._add("a")
        self._add("old")
        self._add("withdrawn")
        with self.messages.conn:
            self.messages.conn.execute(
                "UPDATE facts SET status = 'superseded' WHERE fact_id = 'old'",
            )
            self.messages.conn.execute(
                "UPDATE facts SET status = 'retracted' WHERE fact_id = 'withdrawn'",
            )
            # 直接构造不同版本，验证列表按实际版本取来源；当前尚无更新接口。
            for version in (3, 2):
                self.messages.conn.execute(
                    """INSERT INTO facts (
                           user_id, fact_id, version, subject_id, predicate, object_json,
                           scope, session_id, valid_from, valid_to, status, recorded_at
                       ) SELECT user_id, fact_id, ?, subject_id, predicate, ?,
                                scope, session_id, valid_from, valid_to, status, recorded_at
                         FROM facts WHERE user_id = 'u1' AND fact_id = 'a' AND version = 1""",
                    (version, json.dumps({"type": "literal", "value": version})),
                )
                self.messages.conn.execute(
                    """INSERT INTO fact_sources (
                           user_id, fact_id, version, source_session_id, source_message_id
                       ) SELECT user_id, fact_id, ?, source_session_id, source_message_id
                         FROM fact_sources
                         WHERE user_id = 'u1' AND fact_id = 'a' AND version = 1""",
                    (version,),
                )

        listed = self.facts.list_facts(user_id="u1", as_of=self.as_of)

        self.assertEqual(
            [(fact["fact_id"], fact["version"]) for fact in listed],
            [("a", 1), ("a", 2), ("a", 3), ("z", 1)],
        )
        self.assertEqual(listed[1], self.facts.get_fact(user_id="u1", fact_id="a", version=2))
        self.assertEqual(listed[2], self.facts.get_fact(user_id="u1", fact_id="a", version=3))

    def test_validity_includes_start_but_excludes_end(self):
        saved = self._add(
            "bounded",
            valid_from="2026-10-21T10:00:00+00:00",
            valid_to="2026-10-21T11:00:00+00:00",
        )
        for as_of, expected in (
            ("2026-10-21T09:59:59.999999+00:00", []),
            ("2026-10-21T10:00:00+00:00", [saved]),
            ("2026-10-21T10:30:00+00:00", [saved]),
            ("2026-10-21T10:59:59.999999+00:00", [saved]),
            ("2026-10-21T11:00:00+00:00", []),
        ):
            with self.subTest(as_of=as_of):
                self.assertEqual(self.facts.list_facts(user_id="u1", as_of=as_of), expected)

    def test_unknown_bounds_and_timezone_conversion_do_not_admit_expired_or_future_facts(self):
        bounded = self._add(
            "bounded", valid_from="2026-10-21T18:00:00+08:00",
            valid_to="2026-10-21T19:00:00+08:00",
        )
        unbounded = self._add("none")
        start_only = self._add("start", valid_from=self.as_of)
        end_only = self._add("until", valid_to="2026-10-21T10:00:01+00:00")
        self._add("expired", valid_to=self.as_of)
        self._add("future", valid_from="2026-10-21T10:00:01+00:00")

        for as_of in (self.as_of, "2026-10-21T18:00:00+08:00"):
            with self.subTest(as_of=as_of):
                self.assertEqual(
                    self.facts.list_facts(user_id="u1", as_of=as_of),
                    [bounded, unbounded, start_only, end_only],
                )

    def test_default_as_of_uses_current_utc_time(self):
        current = self._add(
            "current", valid_from="2026-10-21T10:00:00+00:00",
            valid_to="2026-10-21T11:00:00+00:00",
        )
        self._add("expired", valid_to="2026-10-21T10:00:00+00:00")
        self._add("future", valid_from="2026-10-21T11:00:00+00:00")
        fixed_now = datetime(2026, 10, 21, 10, 30, tzinfo=timezone.utc)

        with patch("src.fact_store.datetime", wraps=datetime) as clock:
            clock.now.return_value = fixed_now
            listed = self.facts.list_facts(user_id="u1")

        self.assertEqual(listed, [current])
        clock.now.assert_called_once_with(timezone.utc)

    def test_invalid_filters_and_times_raise_even_when_database_is_empty(self):
        for field in ("user_id", "subject_id", "predicate", "session_id"):
            invalid_values = ("", " \t", 123, True, [], {})
            if field == "user_id":
                invalid_values += (None,)
            for value in invalid_values:
                with self.subTest(field=field, value=value):
                    request = {"user_id": "u1", "as_of": self.as_of, field: value}
                    with self.assertRaises(ValueError):
                        self.facts.list_facts(**request)
        for value in (
            "", " ", "not-a-time", "2026-02-30T10:00:00+00:00",
            "2026-10-21T10:00:00", 123, True,
        ):
            with self.subTest(as_of=value):
                with self.assertRaises(ValueError):
                    self.facts.list_facts(user_id="u1", as_of=value)

    def test_special_characters_are_exact_values_and_do_not_expand_user_scope(self):
        owner = " u' OR 1=1 -- "
        subject = " person:'quoted' "
        predicate = "likes%_'quoted'"
        session = " session' OR 1=1 -- "
        saved = self._add(
            "quoted", user_id=owner, subject_id=subject, predicate=predicate,
            scope="session", session_id=session,
        )
        self._add("other-user", scope="session", session_id=session)

        self.assertEqual(
            self.facts.list_facts(
                user_id=owner, subject_id=subject, predicate=predicate,
                session_id=session, as_of=self.as_of,
            ),
            [saved],
        )
        self.assertEqual(
            self.facts.list_facts(
                user_id="u1", subject_id=subject, predicate=predicate,
                session_id=session, as_of=self.as_of,
            ),
            [],
        )
        self.assertEqual(saved["user_id"], owner)
        self.assertEqual(saved["subject_id"], subject)
        self.assertEqual(saved["predicate"], predicate)
        self.assertEqual(saved["session_id"], session)

    def test_listing_preserves_complete_evidence_and_does_not_change_storage(self):
        second_source = self._seed_source(session_id="z-history", message_id="m2")
        first_source = self._seed_source()
        saved = self._add(
            "f1", object={"type": "literal", "value": {"drinks": ["咖啡", "茶"]}},
            sources=[second_source, first_source],
        )
        expected = copy.deepcopy(saved)
        before_changes = self.messages.conn.total_changes

        listed = self.facts.list_facts(user_id="u1", as_of=self.as_of)

        self.assertEqual(listed, [expected])
        self.assertNotIn("object_json", listed[0])
        self.assertEqual(listed[0]["sources"], [first_source, second_source])
        listed[0]["object"]["value"]["drinks"].clear()
        listed[0]["sources"][0]["message_id"] = "changed-output"
        listed.append({"fact_id": "invented"})
        self.assertEqual(
            self.facts.list_facts(user_id="u1", as_of=self.as_of), [expected],
        )
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="f1"), expected)
        self.assertEqual(self.messages.conn.total_changes, before_changes)


if __name__ == "__main__":
    unittest.main()
