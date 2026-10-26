"""人工解决争议的精确目标、结果快照、持久重试和组合事务测试。"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.conflict_store import ConflictStore
from src.fact_store import FactStore
from src.message_store import MessageStore


class ConflictResolutionTests(unittest.TestCase):
    def setUp(self):
        self.messages = MessageStore(":memory:")
        self.addCleanup(self.messages.close)
        self.facts = FactStore(self.messages)
        self.conflicts = ConflictStore(self.facts)
        self.sources_by_user = self.seed(self.messages, self.facts)
        self.sources = self.sources_by_user["u1"]
        self.candidate = {
            "fact_id": "candidate-original-id", "subject_id": "user:u1",
            "predicate": "prefers_drink", "object": {"type": "literal", "value": "茶"},
            "sources": [self.sources[1]],
        }
        self.original = self.record("c1")

    @staticmethod
    def seed(messages, facts):
        sources_by_user = {}
        for user_id in ("u1", "u2"):
            sources = []
            for message_id in ("m1", "m2", "m3"):
                source = {"user_id": user_id, "session_id": "history", "message_id": message_id}
                messages.append_message(
                    **source, occurred_at="2026-10-21T10:00:00+08:00",
                    payload={"role": "user", "content": f"来源 {user_id}/{message_id}"},
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

    def record(self, conflict_id, *, candidate=None, related_facts=None, user_id="u1"):
        return self.conflicts.record_conflict(
            user_id=user_id, conflict_id=conflict_id,
            candidate=deepcopy(self.candidate) if candidate is None else candidate,
            related_facts=[{"fact_id": "f1", "version": 1}] if related_facts is None else related_facts,
            reason="原记录：不同值尚无处理依据",
        )

    def resolve(self, outcome, *, conflict_id="c1", user_id="u1", **kwargs):
        return self.conflicts.resolve_conflict(
            user_id=user_id, conflict_id=conflict_id, outcome=outcome,
            reason=kwargs.pop("reason", "  人工核实后的处理理由  "), **kwargs,
        )

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
                "SELECT * FROM conflict_resolutions ORDER BY user_id, conflict_id",
            )
        )

    def assert_closed_original(self, conflict_id="c1", original=None):
        original = self.original if original is None else original
        expected = {**original, "status": "resolved"}
        self.assertEqual(self.conflicts.get_conflict(user_id="u1", conflict_id=conflict_id), expected)
        self.assertEqual(self.conflicts.record_conflict(user_id="u1", conflict_id=conflict_id), expected)

    def test_keep_existing_saves_noop_receipt_and_preserves_original_conflict_body(self):
        before = self.base_snapshot()
        earliest = datetime.now(timezone.utc)
        resolution = self.resolve("KEEP_EXISTING")
        latest = datetime.now(timezone.utc)
        self.assertEqual(set(resolution), {
            "user_id", "conflict_id", "outcome", "decision", "result", "reason", "resolved_at",
        })
        self.assertEqual(resolution["user_id"], "u1")
        self.assertEqual(resolution["conflict_id"], "c1")
        self.assertEqual(resolution["outcome"], "KEEP_EXISTING")
        self.assertEqual(resolution["decision"], {"action": "NOOP"})
        self.assertEqual(resolution["result"], {"action": "NOOP", "fact": None})
        self.assertEqual(resolution["reason"], "  人工核实后的处理理由  ")
        resolved_at = datetime.fromisoformat(resolution["resolved_at"])
        self.assertEqual(resolved_at.utcoffset(), timedelta(0))
        self.assertLessEqual(earliest, resolved_at)
        self.assertLessEqual(resolved_at, latest)
        self.assertEqual(self.base_snapshot(), before)
        self.assert_closed_original()
        self.assertEqual(self.conflicts.get_resolution(user_id="u1", conflict_id="c1"), resolution)

    def test_add_candidate_uses_explicit_new_id_and_stored_body_and_sources(self):
        first = self.facts.get_fact(user_id="u1", fact_id="f1")
        resolution = self.resolve("ADD_CANDIDATE", new_fact_id="accepted")
        decision, result = resolution["decision"], resolution["result"]
        self.assertEqual(resolution["outcome"], "ADD_CANDIDATE")
        self.assertEqual(decision["action"], "ADD")
        self.assertEqual(decision["fact_id"], "accepted")
        self.assertEqual(decision["object"], self.candidate["object"])
        self.assertEqual(decision["sources"], self.candidate["sources"])
        self.assertEqual(result["action"], "ADD")
        self.assertEqual(set(result), {"action", "fact"})
        current = self.facts.get_fact(user_id="u1", fact_id="accepted", version=1)
        self.assertEqual(result["fact"], current)
        self.assertEqual(current["version"], 1)
        self.assertEqual(current["status"], "active")
        self.assertEqual(current["scope"], "user")
        self.assertIsNone(current["session_id"])
        self.assertEqual(current["object"], self.candidate["object"])
        self.assertEqual(current["sources"], [self.sources[1]])
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="f1"), first)
        self.assertIsNone(self.facts.get_latest_fact(user_id="u1", fact_id="candidate-original-id"))
        self.assert_closed_original()

    def test_correct_updates_exact_related_fact_and_preserves_its_history(self):
        first = self.facts.get_fact(user_id="u1", fact_id="f1", version=1)
        target = {"fact_id": "f1", "version": 1}
        before_target = deepcopy(target)
        resolution = self.resolve("CORRECT", target=target)
        self.assertEqual(target, before_target)
        self.assertEqual(resolution["decision"], {
            "action": "UPDATE", "fact_id": "f1", "new_version": 2,
            "object": self.candidate["object"], "sources": self.candidate["sources"],
        })
        second = self.facts.get_fact(user_id="u1", fact_id="f1", version=2)
        self.assertEqual(resolution["result"], {"action": "UPDATE", "fact": second})
        self.assertEqual(second["object"], self.candidate["object"])
        self.assertEqual(second["sources"], [self.sources[1]])
        self.assertEqual(second["status"], "active")
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="f1", version=1),
                         {**first, "status": "superseded"})
        self.assert_closed_original()

    def test_correction_accepts_equal_utc_intervals_and_inherits_session_scope(self):
        self.facts.add_fact(
            user_id="u1", fact_id="dated", subject_id="user:u1", predicate="prefers_drink",
            object={"type": "literal", "value": "咖啡"}, sources=[self.sources[0]],
            scope="session", session_id="current-chat",
            valid_from="2026-01-01T08:00:00+08:00", valid_to="2027-01-01T08:00:00+08:00",
        )
        candidate = {
            **self.candidate, "scope": "session", "session_id": "current-chat",
            "valid_from": "2026-01-01T00:00:00+00:00", "valid_to": "2027-01-01T08:00:00+08:00",
        }
        original = self.record("dated-conflict", candidate=candidate,
                               related_facts=[{"fact_id": "dated", "version": 1}])
        result = self.resolve("CORRECT", conflict_id="dated-conflict",
                              target={"fact_id": "dated", "version": 1})["result"]["fact"]
        self.assertEqual(result["scope"], "session")
        self.assertEqual(result["session_id"], "current-chat")
        self.assertEqual(result["valid_from"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(result["valid_to"], "2027-01-01T00:00:00+00:00")
        self.assert_closed_original("dated-conflict", original)

    def test_correction_rejects_context_or_effective_interval_changes_without_writes(self):
        for index, changes in enumerate((
            {"subject_id": "user:another"},
            {"predicate": "likes_food"},
            {"scope": "session", "session_id": "current-chat"},
            {"valid_from": "2026-01-01T00:00:00+00:00"},
            {"valid_to": "2027-01-01T00:00:00+00:00"},
        )):
            conflict_id = f"mismatch-{index}"
            self.record(conflict_id, candidate={**self.candidate, **changes})
            before = self.snapshot()
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.resolve("CORRECT", conflict_id=conflict_id, target={"fact_id": "f1", "version": 1})
            self.assertEqual(self.snapshot(), before)

        self.facts.add_fact(
            user_id="u1", fact_id="scoped", subject_id="user:u1", predicate="prefers_drink",
            object={"type": "literal", "value": "咖啡"}, sources=[self.sources[0]],
            scope="session", session_id="chat-a",
        )
        self.record("other-session", candidate={**self.candidate, "scope": "session", "session_id": "chat-b"},
                    related_facts=[{"fact_id": "scoped", "version": 1}])
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.resolve("CORRECT", conflict_id="other-session", target={"fact_id": "scoped", "version": 1})
        self.assertEqual(self.snapshot(), before)

    def test_correction_requires_related_latest_active_target_and_never_falls_back(self):
        for target in (
            None, {}, "f1", {"fact_id": "f1", "version": True},
            {"fact_id": "f1", "version": 0}, {"fact_id": "f1", "version": 1.0},
            {"fact_id": "f1", "version": 1, "user_id": "u2"},
            {"fact_id": "f2", "version": 1}, {"fact_id": "missing", "version": 1},
        ):
            before = self.snapshot()
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.resolve("CORRECT", target=target)
            self.assertEqual(self.snapshot(), before)
        self.facts.update_fact(
            user_id="u1", fact_id="f1", new_version=2,
            object={"type": "literal", "value": "果汁"}, sources=[self.sources[2]],
        )
        for target in ({"fact_id": "f1", "version": 1}, {"fact_id": "f1", "version": 2}):
            before = self.snapshot()
            with self.subTest(stale_or_unrelated=target), self.assertRaises(ValueError):
                self.resolve("CORRECT", target=target)
            self.assertEqual(self.snapshot(), before)
        for status in ("superseded", "retracted"):
            fact_id = f"old-{status}"
            self.facts.add_fact(
                user_id="u1", fact_id=fact_id, subject_id="user:u1", predicate="prefers_drink",
                object={"type": "literal", "value": "咖啡"}, sources=[self.sources[0]], status=status,
            )
            conflict_id = f"conflict-{status}"
            target = {"fact_id": fact_id, "version": 1}
            self.record(conflict_id, related_facts=[target])
            before = self.snapshot()
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.resolve("CORRECT", conflict_id=conflict_id, target=target)
            self.assertEqual(self.snapshot(), before)

    def test_add_rejects_missing_or_occupied_new_identity_at_any_version(self):
        self.facts.add_fact(
            user_id="u1", fact_id="occupied-v3", version=3, status="retracted",
            subject_id="user:u1", predicate="prefers_drink",
            object=self.candidate["object"], sources=[self.sources[1]],
        )
        for new_fact_id in (None, True, "", " \t", "f1", "occupied-v3"):
            before = self.snapshot()
            with self.subTest(new_fact_id=new_fact_id), self.assertRaises(ValueError):
                self.resolve("ADD_CANDIDATE", new_fact_id=new_fact_id)
            self.assertEqual(self.snapshot(), before)
        self.assertIsNone(self.conflicts.get_resolution(user_id="u1", conflict_id="c1"))

    def test_new_resolution_validates_outcome_reason_and_outcome_specific_fields(self):
        for outcome in (None, True, [], "", "NOOP", "DELETE"):
            before = self.snapshot()
            with self.subTest(outcome=outcome), self.assertRaises(ValueError):
                self.resolve(outcome)
            self.assertEqual(self.snapshot(), before)
        for reason in (None, True, 1, "", " \t"):
            before = self.snapshot()
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                self.resolve("KEEP_EXISTING", reason=reason)
            self.assertEqual(self.snapshot(), before)
        for outcome, fields in (
            ("KEEP_EXISTING", {"target": {"fact_id": "f1", "version": 1}}),
            ("KEEP_EXISTING", {"new_fact_id": "unused"}),
            ("ADD_CANDIDATE", {"new_fact_id": "unused", "target": {"fact_id": "f1", "version": 1}}),
            ("CORRECT", {"target": {"fact_id": "f1", "version": 1}, "new_fact_id": "unused"}),
        ):
            before = self.snapshot()
            with self.subTest(outcome=outcome, fields=fields), self.assertRaises(ValueError):
                self.resolve(outcome, **fields)
            self.assertEqual(self.snapshot(), before)

    def test_identity_validation_and_unknown_or_nonopen_conflict_leave_no_receipt(self):
        for field in ("user_id", "conflict_id"):
            for value in (None, True, "", " \t"):
                ids = {"user_id": "u1", "conflict_id": "c1", field: value}
                before = self.snapshot()
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.conflicts.get_resolution(**ids)
                    with self.assertRaises(ValueError):
                        self.conflicts.resolve_conflict(**ids)
                    self.assertEqual(self.snapshot(), before)
        self.assertIsNone(self.conflicts.get_resolution(user_id="u1", conflict_id="missing"))
        self.assertIsNone(self.conflicts.get_resolution(user_id="u1", conflict_id="c1"))
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.resolve("KEEP_EXISTING", conflict_id="missing")
        self.assertEqual(self.snapshot(), before)
        with self.messages.conn:
            self.messages.conn.execute("UPDATE conflicts SET status = 'resolved' WHERE user_id = 'u1' AND conflict_id = 'c1'")
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.resolve("KEEP_EXISTING")
        self.assertEqual(self.snapshot(), before)

    def test_same_ids_and_new_fact_ids_are_isolated_between_users(self):
        self.assertIsNone(self.conflicts.get_resolution(user_id="u2", conflict_id="c1"))
        with self.assertRaises(ValueError):
            self.resolve("KEEP_EXISTING", user_id="u2")
        candidate = {
            **self.candidate, "subject_id": "user:u2", "sources": [self.sources_by_user["u2"][1]],
            "object": {"type": "literal", "value": "牛奶"},
        }
        self.record("c1", candidate=candidate, user_id="u2")
        first = self.resolve("ADD_CANDIDATE", new_fact_id="accepted")
        second = self.resolve("ADD_CANDIDATE", user_id="u2", new_fact_id="accepted")
        self.assertEqual(first["result"]["fact"]["user_id"], "u1")
        self.assertEqual(second["result"]["fact"]["user_id"], "u2")
        self.assertEqual(first["result"]["fact"]["object"], self.candidate["object"])
        self.assertEqual(second["result"]["fact"]["object"], candidate["object"])
        self.assertEqual(self.conflicts.get_resolution(user_id="u1", conflict_id="c1"), first)
        self.assertEqual(self.conflicts.get_resolution(user_id="u2", conflict_id="c1"), second)

    def test_each_outcome_retries_by_ids_before_comparing_or_validating_new_body(self):
        for index, (outcome, fields) in enumerate((
            ("KEEP_EXISTING", {}),
            ("ADD_CANDIDATE", {"new_fact_id": "accepted"}),
            ("CORRECT", {"target": {"fact_id": "f1", "version": 1}}),
        )):
            conflict_id = f"retry-{index}"
            self.record(conflict_id)
            expected = self.resolve(outcome, conflict_id=conflict_id, **fields)
            before = self.snapshot()
            changes = self.messages.conn.total_changes
            for replacement in ({}, {
                "outcome": [], "target": "wrong", "new_fact_id": True, "reason": None,
            }, {
                "outcome": "KEEP_EXISTING", "reason": "重试时不同的正文",
            }):
                with self.subTest(outcome=outcome, replacement=replacement):
                    self.assertEqual(self.conflicts.resolve_conflict(
                        user_id="u1", conflict_id=conflict_id, **replacement,
                    ), expected)
                    self.assertEqual(self.snapshot(), before)
                    self.assertEqual(self.messages.conn.total_changes, changes)

    def test_receipts_are_immutable_processing_snapshots_after_later_fact_changes(self):
        for outcome, fields, fact_id, next_version in (
            ("ADD_CANDIDATE", {"new_fact_id": "accepted"}, "accepted", 2),
            ("CORRECT", {"target": {"fact_id": "f1", "version": 1}}, "f1", 3),
        ):
            conflict_id = f"snapshot-{outcome}"
            self.record(conflict_id)
            result = self.resolve(outcome, conflict_id=conflict_id, **fields)
            expected = deepcopy(result)
            self.facts.update_fact(
                user_id="u1", fact_id=fact_id, new_version=next_version,
                object={"type": "literal", "value": "后来纠正"}, sources=[self.sources[2]],
            )
            self.facts.retract_fact(user_id="u1", fact_id=fact_id, expected_version=next_version)
            result["decision"]["object"]["value"] = "改动返回对象"
            result["result"]["fact"]["sources"].clear()
            before = self.snapshot()
            changes = self.messages.conn.total_changes
            with self.subTest(outcome=outcome):
                self.assertEqual(self.conflicts.get_resolution(user_id="u1", conflict_id=conflict_id), expected)
                self.assertEqual(self.conflicts.resolve_conflict(user_id="u1", conflict_id=conflict_id), expected)
                self.assertEqual(expected["result"]["fact"]["status"], "active")
                self.assertEqual(self.facts.get_latest_fact(user_id="u1", fact_id=fact_id)["status"], "retracted")
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(self.messages.conn.total_changes, changes)

    def test_list_keeps_open_and_resolved_records_with_original_metadata(self):
        second = self.record("c2")
        self.resolve("KEEP_EXISTING")
        records = self.conflicts.list_conflicts(user_id="u1")
        self.assertEqual(records, sorted(records, key=lambda record: (record["recorded_at"], record["conflict_id"])))
        self.assertEqual({record["conflict_id"]: record for record in records}, {
            "c1": {**self.original, "status": "resolved"}, "c2": second,
        })

    def atomic_request(self):
        self.record("atomic", candidate={**self.candidate, "sources": self.sources[1:]})

    def test_failures_in_fact_source_receipt_or_status_write_roll_back_the_whole_resolution(self):
        self.atomic_request()
        for outcome, fields, result_id, result_version in (
            ("ADD_CANDIDATE", {"new_fact_id": "accepted"}, "accepted", 1),
            ("CORRECT", {"target": {"fact_id": "f1", "version": 1}}, "f1", 2),
        ):
            events = [
                f"AFTER INSERT ON facts WHEN NEW.user_id = 'u1' AND NEW.fact_id = '{result_id}' AND NEW.version = {result_version}",
                f"AFTER INSERT ON fact_sources WHEN NEW.user_id = 'u1' AND NEW.fact_id = '{result_id}' AND NEW.source_message_id = 'm3'",
                "AFTER INSERT ON conflict_resolutions",
                "AFTER UPDATE OF status ON conflicts WHEN NEW.status = 'resolved'",
            ]
            if outcome == "CORRECT":
                events.append("AFTER UPDATE OF status ON facts WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'f1' AND NEW.status = 'superseded'")
            for event in events:
                with self.subTest(outcome=outcome, event=event):
                    with self.messages.conn:
                        self.messages.conn.execute(f"""
                            CREATE TRIGGER fail_resolution {event}
                            BEGIN
                                UPDATE facts SET predicate = 'trigger-side-effect'
                                WHERE user_id = 'u1' AND fact_id = 'f2' AND version = 1;
                                SELECT RAISE(FAIL, 'resolution write failed');
                            END
                        """)
                    before = self.snapshot()
                    try:
                        with self.assertRaisesRegex(sqlite3.IntegrityError, "resolution write failed"):
                            self.resolve(outcome, conflict_id="atomic", **fields)
                        self.assertEqual(self.snapshot(), before)
                        self.assertIsNone(self.conflicts.get_resolution(user_id="u1", conflict_id="atomic"))
                    finally:
                        with self.messages.conn:
                            self.messages.conn.execute("DROP TRIGGER fail_resolution")

    def test_ignored_inserts_or_status_update_raise_and_roll_back_all_resolution_writes(self):
        self.atomic_request()
        for outcome, fields, result_id, result_version in (
            ("ADD_CANDIDATE", {"new_fact_id": "accepted"}, "accepted", 1),
            ("CORRECT", {"target": {"fact_id": "f1", "version": 1}}, "f1", 2),
        ):
            events = [
                f"BEFORE INSERT ON facts WHEN NEW.user_id = 'u1' AND NEW.fact_id = '{result_id}' AND NEW.version = {result_version}",
                f"BEFORE INSERT ON fact_sources WHEN NEW.user_id = 'u1' AND NEW.fact_id = '{result_id}' AND NEW.source_message_id = 'm3'",
                "BEFORE INSERT ON conflict_resolutions",
                "BEFORE UPDATE OF status ON conflicts WHEN NEW.status = 'resolved'",
            ]
            if outcome == "CORRECT":
                events.append("BEFORE UPDATE OF status ON facts WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'f1' AND NEW.status = 'superseded'")
            for event in events:
                with self.subTest(outcome=outcome, event=event):
                    with self.messages.conn:
                        self.messages.conn.execute(f"""
                            CREATE TRIGGER ignore_resolution {event}
                            BEGIN
                                SELECT RAISE(IGNORE);
                            END
                        """)
                    before = self.snapshot()
                    try:
                        with self.assertRaises(ValueError):
                            self.resolve(outcome, conflict_id="atomic", **fields)
                        self.assertEqual(self.snapshot(), before)
                    finally:
                        with self.messages.conn:
                            self.messages.conn.execute("DROP TRIGGER ignore_resolution")

    def test_bad_stored_candidate_cannot_write_a_fact_but_can_be_explicitly_rejected(self):
        for index, changes in enumerate((
            {"subject_id": None}, {"predicate": ""},
            {"object": {"type": "entity", "value": "entity:1"}},
        )):
            conflict_id = f"bad-candidate-{index}"
            original = self.record(conflict_id, candidate={**self.candidate, **changes})
            before = self.snapshot()
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.resolve("ADD_CANDIDATE", conflict_id=conflict_id, new_fact_id=f"accepted-{index}")
            self.assertEqual(self.snapshot(), before)
            self.assertEqual(self.resolve("KEEP_EXISTING", conflict_id=conflict_id)["result"], {"action": "NOOP", "fact": None})
            self.assert_closed_original(conflict_id, original)

    def test_resolution_receipt_foreign_key_rejects_an_orphan(self):
        before = self.snapshot()
        with self.assertRaises(sqlite3.IntegrityError):
            with self.messages.conn:
                self.messages.conn.execute("""
                    INSERT INTO conflict_resolutions
                    (user_id, conflict_id, outcome, decision_json, result_json, reason, resolved_at)
                    VALUES ('u1', 'missing', 'KEEP_EXISTING', '{"action":"NOOP"}',
                            '{"action":"NOOP","fact":null}', 'test', '2026-10-21T00:00:00+00:00')
                """)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.messages.conn.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_old_database_gains_receipt_table_and_reopen_keeps_the_first_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            messages = MessageStore(path)
            try:
                facts = FactStore(messages)
                self.seed(messages, facts)
                conflicts = ConflictStore(facts)
                original = conflicts.record_conflict(
                    user_id="u1", conflict_id="c1", candidate=self.candidate,
                    related_facts=[{"fact_id": "f1", "version": 1}], reason="旧数据库的争议",
                )
                # 模拟仅具争议记录能力的旧库，原事实与争议均保留。
                with messages.conn:
                    messages.conn.execute("DROP TABLE conflict_resolutions")
            finally:
                messages.close()
            messages = MessageStore(path)
            try:
                conflicts = ConflictStore(FactStore(messages))
                self.assertEqual(conflicts.get_conflict(user_id="u1", conflict_id="c1"), original)
                receipt = conflicts.resolve_conflict(
                    user_id="u1", conflict_id="c1", outcome="ADD_CANDIDATE",
                    new_fact_id="accepted", reason="人工认可并存",
                )
            finally:
                messages.close()
            messages = MessageStore(path)
            try:
                facts = FactStore(messages)
                conflicts = ConflictStore(facts)
                changes = messages.conn.total_changes
                self.assertEqual(conflicts.get_resolution(user_id="u1", conflict_id="c1"), receipt)
                self.assertEqual(conflicts.resolve_conflict(user_id="u1", conflict_id="c1"), receipt)
                self.assertEqual(messages.conn.total_changes, changes)
                self.assertEqual(facts.get_fact(user_id="u1", fact_id="accepted"), receipt["result"]["fact"])
                self.assertIsNone(facts.get_fact(user_id="u1", fact_id="accepted", version=2))
                self.assertEqual(conflicts.get_conflict(user_id="u1", conflict_id="c1"), {**original, "status": "resolved"})
            finally:
                messages.close()

    def test_public_fact_update_still_commits_for_an_independent_disk_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            messages = MessageStore(path)
            try:
                facts = FactStore(messages)
                sources = self.seed(messages, facts)["u1"]
                second = facts.update_fact(
                    user_id="u1", fact_id="f1", new_version=2,
                    object={"type": "literal", "value": "果汁"}, sources=[sources[1]],
                )
                reader = sqlite3.connect(path)
                try:
                    statuses = reader.execute(
                        "SELECT version, status FROM facts WHERE user_id = 'u1' AND fact_id = 'f1' ORDER BY version"
                    ).fetchall()
                    self.assertEqual(statuses, [(1, "superseded"), (2, "active")])
                    count = reader.execute(
                        "SELECT COUNT(*) FROM fact_sources WHERE user_id = 'u1' AND fact_id = 'f1' AND version = 2"
                    ).fetchone()[0]
                    self.assertEqual(count, 1)
                finally:
                    reader.close()
                self.assertEqual(facts.get_latest_fact(user_id="u1", fact_id="f1"), second)
            finally:
                messages.close()


if __name__ == "__main__":
    unittest.main()
