"""争议撤回、时间迁移及旧解决记录表升级的独立验收测试。"""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.conflict_store import ConflictStore
from src.fact_store import FactStore
from src.message_store import MessageStore


class ConflictResolutionActionTests(unittest.TestCase):
    def setUp(self):
        self.messages = MessageStore(":memory:")
        self.addCleanup(self.messages.close)
        self.facts = FactStore(self.messages)
        self.conflicts = ConflictStore(self.facts)
        self.sources_by_user = self.seed(self.messages, self.facts)
        self.sources = self.sources_by_user["u1"]
        self.candidate = {
            "fact_id": "candidate-original-id", "subject_id": "user:u1",
            "predicate": "lives_in", "object": {"type": "literal", "value": "杭州"},
            "sources": [self.sources[1]], "valid_from": "2026-06-01T08:00:00+08:00",
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
            for fact_id, predicate in (("home", "lives_in"), ("other", "works_in")):
                facts.add_fact(
                    user_id=user_id, fact_id=fact_id, subject_id=f"user:{user_id}",
                    predicate=predicate, object={"type": "literal", "value": "北京"},
                    sources=[sources[0]], valid_from="2026-01-01T08:00:00+08:00",
                    valid_to="2027-01-01T08:00:00+08:00",
                )
        return sources_by_user

    def add_home(self, fact_id, *, user_id="u1", **changes):
        return self.facts.add_fact(**{
            "user_id": user_id, "fact_id": fact_id, "subject_id": f"user:{user_id}",
            "predicate": "lives_in", "object": {"type": "literal", "value": "北京"},
            "sources": [self.sources_by_user[user_id][0]],
            "valid_from": "2026-01-01T08:00:00+08:00",
            "valid_to": "2027-01-01T08:00:00+08:00", **changes,
        })

    def record(self, conflict_id, *, candidate=None, related_facts=None, user_id="u1"):
        return self.conflicts.record_conflict(
            user_id=user_id, conflict_id=conflict_id,
            candidate=deepcopy(self.candidate) if candidate is None else candidate,
            related_facts=[{"fact_id": "home", "version": 1}] if related_facts is None else related_facts,
            reason="原争议及来源需要保留",
        )

    def resolve(self, outcome, *, conflict_id="c1", user_id="u1", target=None, **kwargs):
        return self.conflicts.resolve_conflict(
            user_id=user_id, conflict_id=conflict_id, outcome=outcome,
            target={"fact_id": "home", "version": 1} if target is None else target,
            reason=kwargs.pop("reason", "  人工确认本次处理  "), **kwargs,
        )

    def snapshot(self):
        return tuple(
            tuple(map(tuple, self.messages.conn.execute(query).fetchall()))
            for query in (
                "SELECT * FROM messages ORDER BY user_id, session_id, message_id",
                "SELECT * FROM facts ORDER BY user_id, fact_id, version",
                """SELECT * FROM fact_sources
                   ORDER BY user_id, fact_id, version, source_session_id, source_message_id""",
                "SELECT * FROM fact_transitions ORDER BY user_id, new_fact_id",
                "SELECT * FROM conflicts ORDER BY user_id, conflict_id",
                """SELECT * FROM conflict_facts
                   ORDER BY user_id, conflict_id, fact_id, version""",
                """SELECT * FROM conflict_sources
                   ORDER BY user_id, conflict_id, source_session_id, source_message_id""",
                "SELECT * FROM conflict_resolutions ORDER BY user_id, conflict_id",
            )
        )

    def assert_resolved_original(self, conflict_id="c1", original=None):
        original = self.original if original is None else original
        self.assertEqual(self.conflicts.get_conflict(user_id="u1", conflict_id=conflict_id),
                         {**original, "status": "resolved"})

    def test_retract_marks_only_the_explicit_version_without_changing_its_body(self):
        first = self.facts.get_fact(user_id="u1", fact_id="home", version=1)
        target = {"fact_id": "home", "version": 1}
        before_target = deepcopy(target)
        receipt = self.resolve("RETRACT", target=target)
        self.assertEqual(target, before_target)
        self.assertEqual(receipt["outcome"], "RETRACT")
        self.assertEqual(receipt["decision"], {
            "action": "DELETE", "fact_id": "home", "expected_version": 1,
        })
        self.assertEqual(receipt["result"], {
            "action": "DELETE", "fact": {**first, "status": "retracted"},
        })
        self.assertEqual(self.facts.get_latest_fact(user_id="u1", fact_id="home"), receipt["result"]["fact"])
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="home", version=2))
        self.assertEqual(receipt["result"]["fact"]["sources"], [self.sources[0]])
        for as_of in ("2026-05-31T23:59:59+00:00", "2026-06-01T00:00:00+00:00",
                      "2026-06-01T00:00:01+00:00"):
            self.assertEqual(self.facts.list_facts(user_id="u1", predicate="lives_in", as_of=as_of), [])
        self.assert_resolved_original()

    def test_retract_does_not_require_candidate_object_or_valid_time_fields(self):
        for index, changes in enumerate((
            {"object": None, "valid_from": "not-a-time", "valid_to": "also-invalid"},
            {"object": {"type": "entity", "value": "entity:1"}, "valid_from": None},
        )):
            fact_id = f"home-ignore-{index}"
            first = self.add_home(fact_id)
            target = {"fact_id": fact_id, "version": 1}
            candidate = {**self.candidate, **changes}
            if index == 1:
                candidate.pop("object")
            conflict_id = f"ignore-{index}"
            original = self.record(conflict_id, candidate=candidate, related_facts=[target])
            with self.subTest(candidate=candidate):
                result = self.resolve("RETRACT", conflict_id=conflict_id, target=target)["result"]
                self.assertEqual(result["fact"], {**first, "status": "retracted"})
                self.assert_resolved_original(conflict_id, original)

    def test_retract_can_use_a_related_latest_version_greater_than_one(self):
        latest = self.facts.update_fact(
            user_id="u1", fact_id="home", new_version=2,
            object={"type": "literal", "value": "天津"}, sources=[self.sources[2]],
        )
        target = {"fact_id": "home", "version": 2}
        self.record("latest", related_facts=[target])
        receipt = self.resolve("RETRACT", conflict_id="latest", target=target)
        self.assertEqual(receipt["decision"]["expected_version"], 2)
        self.assertEqual(receipt["result"]["fact"], {**latest, "status": "retracted"})
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="home", version=3))

    def test_transition_preserves_original_history_and_attributes_sources_to_both_results(self):
        first = self.facts.get_fact(user_id="u1", fact_id="home", version=1)
        receipt = self.resolve("TRANSITION", new_fact_id="hangzhou")
        self.assertEqual(receipt["outcome"], "TRANSITION")
        decision, result = receipt["decision"], receipt["result"]
        self.assertEqual(decision, {
            "action": "TRANSITION", "fact_id": "home", "new_version": 2,
            "new_fact_id": "hangzhou", "effective_at": self.candidate["valid_from"],
            "object": self.candidate["object"], "sources": self.candidate["sources"],
            "valid_to": None,
        })
        self.assertEqual(set(result), {"action", "previous_fact", "fact"})
        self.assertEqual(result["action"], "TRANSITION")
        old, new = result["previous_fact"], result["fact"]
        self.assertEqual(old["fact_id"], "home")
        self.assertEqual(old["version"], 2)
        self.assertEqual(old["object"], first["object"])
        self.assertEqual(old["valid_from"], first["valid_from"])
        self.assertEqual(old["valid_to"], "2026-06-01T00:00:00+00:00")
        self.assertEqual(old["sources"], self.sources[:2])
        self.assertEqual(new["fact_id"], "hangzhou")
        self.assertEqual(new["version"], 1)
        self.assertEqual(new["object"], self.candidate["object"])
        self.assertEqual(new["valid_from"], old["valid_to"])
        self.assertIsNone(new["valid_to"])
        self.assertEqual(new["sources"], [self.sources[1]])
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="home", version=1),
                         {**first, "status": "superseded"})
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="home", version=2), old)
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="hangzhou", version=1), new)
        self.assertIsNone(self.facts.get_latest_fact(user_id="u1", fact_id="candidate-original-id"))
        self.assert_resolved_original()

    def test_transition_queries_switch_at_exact_boundary_and_new_end_is_independent(self):
        candidate = {**self.candidate, "valid_to": "2028-01-01T08:00:00+08:00"}
        self.record("bounded", candidate=candidate)
        result = self.resolve("TRANSITION", conflict_id="bounded", new_fact_id="hangzhou")["result"]
        self.assertEqual(result["fact"]["valid_to"], "2028-01-01T00:00:00+00:00")
        for as_of, expected in (
            ("2025-12-31T23:59:59+00:00", []),
            ("2026-05-31T23:59:59.999999+00:00", [result["previous_fact"]]),
            ("2026-06-01T00:00:00+00:00", [result["fact"]]),
            ("2026-06-01T00:00:00.000001+00:00", [result["fact"]]),
            ("2027-06-01T00:00:00+00:00", [result["fact"]]),
            ("2028-01-01T00:00:00+00:00", []),
        ):
            with self.subTest(as_of=as_of):
                self.assertEqual(self.facts.list_facts(
                    user_id="u1", subject_id="user:u1", predicate="lives_in", as_of=as_of,
                ), expected)

    def test_transition_inherits_session_scope_and_only_changes_its_selected_target(self):
        self.add_home("scoped", scope="session", session_id="current-chat")
        other = self.add_home("overlapping", scope="session", session_id="current-chat")
        target = {"fact_id": "scoped", "version": 1}
        candidate = {**self.candidate, "scope": "session", "session_id": "current-chat"}
        self.record("scoped-conflict", candidate=candidate, related_facts=[target])
        receipt = self.resolve("TRANSITION", conflict_id="scoped-conflict", target=target, new_fact_id="new-scoped")
        for fact in (receipt["result"]["previous_fact"], receipt["result"]["fact"]):
            self.assertEqual(fact["scope"], "session")
            self.assertEqual(fact["session_id"], "current-chat")
        self.assertEqual(self.facts.get_latest_fact(user_id="u1", fact_id="overlapping"), other)
        self.assertEqual(self.facts.list_facts(user_id="u1", session_id="another-chat",
                                              predicate="lives_in", as_of="2026-07-01T00:00:00+00:00"),
                         self.facts.list_facts(user_id="u1", predicate="lives_in", as_of="2026-07-01T00:00:00+00:00"))

    def test_actions_require_a_related_latest_active_exact_target(self):
        for outcome in ("RETRACT", "TRANSITION"):
            extra = {} if outcome == "RETRACT" else {"new_fact_id": "accepted"}
            for target in (None, {}, "home", {"fact_id": "home", "version": True},
                           {"fact_id": "home", "version": 0},
                           {"fact_id": "home", "version": 1, "user_id": "u2"},
                           {"fact_id": "other", "version": 1}):
                before = self.snapshot()
                with self.subTest(outcome=outcome, target=target), self.assertRaises(ValueError):
                    self.conflicts.resolve_conflict(
                        user_id="u1", conflict_id="c1", outcome=outcome,
                        target=target, reason="人工确认", **extra,
                    )
                self.assertEqual(self.snapshot(), before)
        self.facts.update_fact(user_id="u1", fact_id="home", new_version=2,
                               object={"type": "literal", "value": "天津"}, sources=[self.sources[2]])
        for outcome in ("RETRACT", "TRANSITION"):
            extra = {} if outcome == "RETRACT" else {"new_fact_id": "accepted"}
            for target in ({"fact_id": "home", "version": 1}, {"fact_id": "home", "version": 2}):
                before = self.snapshot()
                with self.subTest(outcome=outcome, stale_or_unrelated=target), self.assertRaises(ValueError):
                    self.resolve(outcome, target=target, **extra)
                self.assertEqual(self.snapshot(), before)
        for status in ("retracted", "superseded"):
            fact_id = f"old-{status}"
            self.add_home(fact_id, status=status)
            target = {"fact_id": fact_id, "version": 1}
            self.record(f"conflict-{status}", related_facts=[target])
            for outcome in ("RETRACT", "TRANSITION"):
                before = self.snapshot()
                extra = {} if outcome == "RETRACT" else {"new_fact_id": "accepted"}
                with self.subTest(outcome=outcome, status=status), self.assertRaises(ValueError):
                    self.resolve(outcome, conflict_id=f"conflict-{status}", target=target, **extra)
                self.assertEqual(self.snapshot(), before)

    def test_actions_reject_subject_predicate_or_exact_scope_mismatch(self):
        for index, changes in enumerate((
            {"subject_id": "user:another"}, {"predicate": "works_in"},
            {"scope": "session", "session_id": "current-chat"},
        )):
            conflict_id = f"mismatch-{index}"
            self.record(conflict_id, candidate={**self.candidate, **changes})
            for outcome in ("RETRACT", "TRANSITION"):
                before = self.snapshot()
                extra = {} if outcome == "RETRACT" else {"new_fact_id": "accepted"}
                with self.subTest(outcome=outcome, changes=changes), self.assertRaises(ValueError):
                    self.resolve(outcome, conflict_id=conflict_id, **extra)
                self.assertEqual(self.snapshot(), before)

    def test_transition_rejects_missing_invalid_or_outside_times_without_writing(self):
        for index, changes in enumerate((
            {"valid_from": None}, {"valid_from": "invalid"}, {"valid_from": "2026-06-01T00:00:00"},
            {"valid_from": "2026-01-01T00:00:00+00:00"},
            {"valid_from": "2025-12-01T00:00:00+00:00"},
            {"valid_from": "2027-01-01T00:00:00+00:00"},
            {"valid_from": "2028-01-01T00:00:00+00:00"},
            {"valid_to": "2026-06-01T00:00:00+00:00"},
            {"valid_to": "2026-05-01T00:00:00+00:00"},
            {"object": {"type": "entity", "value": "entity:1"}},
        )):
            conflict_id = f"bad-time-{index}"
            self.record(conflict_id, candidate={**self.candidate, **changes})
            before = self.snapshot()
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.resolve("TRANSITION", conflict_id=conflict_id, new_fact_id="accepted")
            self.assertEqual(self.snapshot(), before)
        missing = deepcopy(self.candidate)
        missing.pop("valid_from")
        self.record("missing-time", candidate=missing)
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.resolve("TRANSITION", conflict_id="missing-time", new_fact_id="accepted")
        self.assertEqual(self.snapshot(), before)

    def test_transition_cannot_reuse_an_occupied_id_or_an_external_migration_receipt(self):
        self.add_home("occupied-v3", version=3, status="retracted")
        self.add_home("external")
        self.facts.transition_fact(
            user_id="u1", fact_id="external", new_version=2, new_fact_id="external-new",
            effective_at=self.candidate["valid_from"], object=self.candidate["object"], sources=[self.sources[2]],
        )
        for new_fact_id in (None, True, "", " \t", "home", "occupied-v3", "external-new"):
            before = self.snapshot()
            with self.subTest(new_fact_id=new_fact_id), self.assertRaises(ValueError):
                self.resolve("TRANSITION", new_fact_id=new_fact_id)
            self.assertEqual(self.snapshot(), before)
            self.assertIsNone(self.conflicts.get_resolution(user_id="u1", conflict_id="c1"))

    def test_retract_forbids_new_id_and_both_outcomes_require_a_reason(self):
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.resolve("RETRACT", new_fact_id="not-allowed")
        self.assertEqual(self.snapshot(), before)
        for outcome in ("RETRACT", "TRANSITION"):
            extra = {} if outcome == "RETRACT" else {"new_fact_id": "accepted"}
            for reason in (None, True, "", " \t"):
                before = self.snapshot()
                with self.subTest(outcome=outcome, reason=reason), self.assertRaises(ValueError):
                    self.resolve(outcome, reason=reason, **extra)
                self.assertEqual(self.snapshot(), before)

    def test_both_outcomes_isolate_same_conflict_and_fact_ids_between_users(self):
        for outcome in ("RETRACT", "TRANSITION"):
            fact_id, conflict_id = f"target-{outcome}", f"conflict-{outcome}"
            receipts = {}
            for user_id in ("u1", "u2"):
                self.add_home(fact_id, user_id=user_id)
                candidate = {**self.candidate, "subject_id": f"user:{user_id}",
                             "sources": [self.sources_by_user[user_id][1]]}
                target = {"fact_id": fact_id, "version": 1}
                self.record(conflict_id, candidate=candidate, related_facts=[target], user_id=user_id)
                extra = {} if outcome == "RETRACT" else {"new_fact_id": "shared-new"}
                receipts[user_id] = self.resolve(outcome, conflict_id=conflict_id, user_id=user_id, target=target, **extra)
                self.assertEqual(receipts[user_id]["result"]["fact"]["user_id"], user_id)
                self.assertEqual(self.conflicts.get_resolution(user_id=user_id, conflict_id=conflict_id), receipts[user_id])
            self.assertNotEqual(receipts["u1"]["result"]["fact"]["sources"], receipts["u2"]["result"]["fact"]["sources"])

    def test_both_outcomes_retry_by_identity_before_validating_or_comparing_new_body(self):
        self.add_home("transition-target")
        target = {"fact_id": "transition-target", "version": 1}
        self.record("transition", related_facts=[target])
        for outcome, conflict_id, fields in (
            ("RETRACT", "c1", {"target": {"fact_id": "home", "version": 1}}),
            ("TRANSITION", "transition", {"target": target, "new_fact_id": "accepted"}),
        ):
            expected = self.resolve(outcome, conflict_id=conflict_id, **fields)
            before = self.snapshot()
            changes = self.messages.conn.total_changes
            for replacement in ({}, {"outcome": [], "target": "invalid", "new_fact_id": True, "reason": None},
                                {"outcome": "KEEP_EXISTING", "reason": "重试时不同的正文"}):
                with self.subTest(outcome=outcome, replacement=replacement):
                    self.assertEqual(self.conflicts.resolve_conflict(
                        user_id="u1", conflict_id=conflict_id, **replacement,
                    ), expected)
                    self.assertEqual(self.snapshot(), before)
                    self.assertEqual(self.messages.conn.total_changes, changes)

    def test_transition_receipt_remains_a_snapshot_after_both_facts_change_and_retract(self):
        receipt = self.resolve("TRANSITION", new_fact_id="accepted")
        expected = deepcopy(receipt)
        for fact_id, new_version in (("home", 3), ("accepted", 2)):
            self.facts.update_fact(user_id="u1", fact_id=fact_id, new_version=new_version,
                                   object={"type": "literal", "value": "后来纠正"}, sources=[self.sources[2]])
            self.facts.retract_fact(user_id="u1", fact_id=fact_id, expected_version=new_version)
        receipt["result"]["previous_fact"]["sources"].clear()
        receipt["result"]["fact"]["object"]["value"] = "修改导出"
        before = self.snapshot()
        changes = self.messages.conn.total_changes
        self.assertEqual(self.conflicts.get_resolution(user_id="u1", conflict_id="c1"), expected)
        self.assertEqual(self.conflicts.resolve_conflict(user_id="u1", conflict_id="c1"), expected)
        self.assertEqual(expected["result"]["previous_fact"]["status"], "active")
        self.assertEqual(expected["result"]["fact"]["status"], "active")
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.messages.conn.total_changes, changes)

    def assert_fault_rollbacks(self, outcome, conflict_id, fields, events, modes=("FAIL", "IGNORE")):
        for mode in modes:
            for event in events:
                timing = "AFTER" if mode == "FAIL" else "BEFORE"
                with self.subTest(outcome=outcome, mode=mode, event=event):
                    with self.messages.conn:
                        self.messages.conn.execute(f"""
                            CREATE TRIGGER fault_new_resolution {timing} {event}
                            BEGIN
                                UPDATE facts SET predicate = 'trigger-side-effect'
                                WHERE user_id = 'u1' AND fact_id = 'other' AND version = 1;
                                SELECT RAISE({mode}, 'resolution write failed');
                            END
                        """ if mode == "FAIL" else f"""
                            CREATE TRIGGER fault_new_resolution {timing} {event}
                            BEGIN
                                UPDATE facts SET predicate = 'trigger-side-effect'
                                WHERE user_id = 'u1' AND fact_id = 'other' AND version = 1;
                                SELECT RAISE(IGNORE);
                            END
                        """)
                    before = self.snapshot()
                    try:
                        error = sqlite3.IntegrityError if mode == "FAIL" else ValueError
                        with self.assertRaises(error):
                            self.resolve(outcome, conflict_id=conflict_id, **fields)
                        self.assertEqual(self.snapshot(), before)
                        self.assertIsNone(self.conflicts.get_resolution(user_id="u1", conflict_id=conflict_id))
                    finally:
                        with self.messages.conn:
                            self.messages.conn.execute("DROP TRIGGER fault_new_resolution")

    def test_retract_fact_receipt_and_status_failures_or_ignored_writes_roll_back(self):
        self.assert_fault_rollbacks("RETRACT", "c1", {}, (
            "UPDATE OF status ON facts WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'home' AND NEW.status = 'retracted'",
            "INSERT ON conflict_resolutions",
            "UPDATE OF status ON conflicts WHEN NEW.status = 'resolved'",
        ))

    def test_transition_failure_at_every_write_stage_rolls_back_both_receipts_and_facts(self):
        self.record("atomic", candidate={**self.candidate, "sources": self.sources[1:]})
        self.assert_fault_rollbacks("TRANSITION", "atomic", {"new_fact_id": "accepted"}, (
            "INSERT ON facts WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'accepted' AND NEW.version = 1",
            "INSERT ON fact_sources WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'accepted' AND NEW.source_message_id = 'm3'",
            "UPDATE OF status ON facts WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'home' AND NEW.status = 'superseded'",
            "INSERT ON facts WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'home' AND NEW.version = 2",
            "INSERT ON fact_sources WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'home' AND NEW.version = 2 AND NEW.source_message_id = 'm3'",
            "INSERT ON fact_transitions",
            "INSERT ON conflict_resolutions",
            "UPDATE OF status ON conflicts WHEN NEW.status = 'resolved'",
        ), modes=("FAIL",))

    def test_transition_ignored_writes_at_every_stage_are_errors_and_roll_back(self):
        self.record("atomic", candidate={**self.candidate, "sources": self.sources[1:]})
        self.assert_fault_rollbacks("TRANSITION", "atomic", {"new_fact_id": "accepted"}, (
            "INSERT ON facts WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'accepted' AND NEW.version = 1",
            "INSERT ON fact_sources WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'accepted' AND NEW.source_message_id = 'm3'",
            "UPDATE OF status ON facts WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'home' AND NEW.status = 'superseded'",
            "INSERT ON facts WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'home' AND NEW.version = 2",
            "INSERT ON fact_sources WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'home' AND NEW.version = 2 AND NEW.source_message_id = 'm3'",
            "INSERT ON fact_transitions",
            "INSERT ON conflict_resolutions",
            "UPDATE OF status ON conflicts WHEN NEW.status = 'resolved'",
        ), modes=("IGNORE",))

    @staticmethod
    def receipt_rows(conn):
        return tuple(map(tuple, conn.execute("""
            SELECT user_id, conflict_id, outcome, decision_json, result_json, reason, resolved_at
            FROM conflict_resolutions ORDER BY user_id, conflict_id
        """).fetchall()))

    @staticmethod
    def receipt_schema(conn):
        return conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'conflict_resolutions'").fetchone()[0]

    def install_legacy_receipts(self, messages, facts, conflicts, sources):
        home = facts.get_fact(user_id="u1", fact_id="home", version=1)
        candidate = {**self.candidate, "sources": [sources[1]],
                     "valid_from": home["valid_from"], "valid_to": home["valid_to"]}
        for conflict_id, outcome, fields in (
            ("legacy-keep", "KEEP_EXISTING", {}),
            ("legacy-add", "ADD_CANDIDATE", {"new_fact_id": "legacy-accepted"}),
            ("legacy-correct", "CORRECT", {"target": {"fact_id": "home", "version": 1}}),
        ):
            conflicts.record_conflict(user_id="u1", conflict_id=conflict_id, candidate=candidate,
                                      related_facts=[{"fact_id": "home", "version": 1}], reason="旧争议原文")
            conflicts.resolve_conflict(user_id="u1", conflict_id=conflict_id, outcome=outcome,
                                       reason="旧处理原因原文", **fields)
        with messages.conn:
            for row in self.receipt_rows(messages.conn):
                # 升级必须保留原始 JSON 文本，而非重新解码、编码成等价文本。
                decision = " \n" + json.dumps(json.loads(row[3]), ensure_ascii=False, indent=2) + "\n "
                result = "\n" + json.dumps(json.loads(row[4]), ensure_ascii=False, indent=4) + "\n"
                messages.conn.execute("""
                    UPDATE conflict_resolutions SET decision_json = ?, result_json = ?
                    WHERE user_id = ? AND conflict_id = ?
                """, (decision, result, row[0], row[1]))
            messages.conn.execute("""
                CREATE TABLE legacy_resolution_fixture (
                    user_id TEXT NOT NULL, conflict_id TEXT NOT NULL,
                    outcome TEXT NOT NULL CHECK (outcome IN ('KEEP_EXISTING', 'ADD_CANDIDATE', 'CORRECT')),
                    decision_json TEXT NOT NULL, result_json TEXT NOT NULL,
                    reason TEXT NOT NULL, resolved_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, conflict_id),
                    FOREIGN KEY (user_id, conflict_id) REFERENCES conflicts (user_id, conflict_id)
                )
            """)
            messages.conn.execute("""
                INSERT INTO legacy_resolution_fixture
                (user_id, conflict_id, outcome, decision_json, result_json, reason, resolved_at)
                SELECT user_id, conflict_id, outcome, decision_json, result_json, reason, resolved_at
                FROM conflict_resolutions
            """)
            messages.conn.execute("DROP TABLE conflict_resolutions")
            messages.conn.execute("ALTER TABLE legacy_resolution_fixture RENAME TO conflict_resolutions")
        return self.receipt_rows(messages.conn)

    def test_old_three_outcome_schema_upgrades_preserving_raw_receipts_and_resolved_statuses(self):
        before_rows = self.install_legacy_receipts(self.messages, self.facts, self.conflicts, self.sources)
        before_records = self.conflicts.list_conflicts(user_id="u1")
        old_schema = self.receipt_schema(self.messages.conn)
        self.assertNotIn("'RETRACT'", old_schema)
        self.assertNotIn("'TRANSITION'", old_schema)
        upgraded = ConflictStore(self.facts)
        self.assertEqual(self.receipt_rows(self.messages.conn), before_rows)
        self.assertEqual(upgraded.list_conflicts(user_id="u1"), before_records)
        schema = self.receipt_schema(self.messages.conn)
        self.assertIn("'RETRACT'", schema)
        self.assertIn("'TRANSITION'", schema)
        for conflict_id in ("legacy-keep", "legacy-add", "legacy-correct"):
            receipt = upgraded.get_resolution(user_id="u1", conflict_id=conflict_id)
            self.assertEqual(upgraded.resolve_conflict(user_id="u1", conflict_id=conflict_id), receipt)
            self.assertEqual(upgraded.get_conflict(user_id="u1", conflict_id=conflict_id)["status"], "resolved")
        # 已升级库再次初始化不应重建这张表。
        self.messages.conn.set_authorizer(lambda action, arg1, arg2, db, trigger:
            sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_DROP_TABLE and arg1 == "conflict_resolutions"
            else sqlite3.SQLITE_OK)
        try:
            ConflictStore(self.facts)
        finally:
            self.messages.conn.set_authorizer(None)
        self.assertEqual(self.receipt_rows(self.messages.conn), before_rows)
        self.assertEqual(self.receipt_schema(self.messages.conn), schema)
        self.add_home("upgraded-target")
        target = {"fact_id": "upgraded-target", "version": 1}
        self.record("new-transition", related_facts=[target])
        result = upgraded.resolve_conflict(user_id="u1", conflict_id="new-transition", outcome="TRANSITION",
                                           target=target, new_fact_id="upgraded-new", reason="升级后迁移")
        self.assertEqual(result["result"]["fact"]["fact_id"], "upgraded-new")
        self.record("new-retract", related_facts=[{"fact_id": "upgraded-new", "version": 1}])
        result = upgraded.resolve_conflict(user_id="u1", conflict_id="new-retract", outcome="RETRACT",
                                           target={"fact_id": "upgraded-new", "version": 1}, reason="升级后撤回")
        self.assertEqual(result["result"]["fact"]["status"], "retracted")

    def test_schema_upgrade_failure_rolls_back_copied_receipts_and_keeps_old_table(self):
        before_rows = self.install_legacy_receipts(self.messages, self.facts, self.conflicts, self.sources)
        before = self.snapshot()
        old_schema = self.receipt_schema(self.messages.conn)
        self.messages.conn.set_authorizer(lambda action, arg1, arg2, db, trigger:
            sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_DROP_TABLE and arg1 == "conflict_resolutions"
            else sqlite3.SQLITE_OK)
        try:
            with self.assertRaises(sqlite3.DatabaseError):
                ConflictStore(self.facts)
        finally:
            self.messages.conn.set_authorizer(None)
        self.assertEqual(self.receipt_rows(self.messages.conn), before_rows)
        self.assertEqual(self.receipt_schema(self.messages.conn), old_schema)
        self.assertEqual(self.snapshot(), before)
        self.assertIsNone(self.messages.conn.execute("SELECT name FROM sqlite_master WHERE name = 'conflict_resolutions_v2'").fetchone())
        self.assertEqual(self.messages.conn.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_schema_upgrade_does_not_delete_an_unrecognized_temporary_table(self):
        before_rows = self.install_legacy_receipts(self.messages, self.facts, self.conflicts, self.sources)
        with self.messages.conn:
            self.messages.conn.execute("CREATE TABLE conflict_resolutions_v2 (note TEXT NOT NULL)")
            self.messages.conn.execute("INSERT INTO conflict_resolutions_v2 VALUES ('必须保留的未知数据')")
        old_schema = self.receipt_schema(self.messages.conn)
        with self.assertRaises((ValueError, sqlite3.DatabaseError)):
            ConflictStore(self.facts)
        self.assertEqual(self.receipt_rows(self.messages.conn), before_rows)
        self.assertEqual(self.receipt_schema(self.messages.conn), old_schema)
        notes = self.messages.conn.execute("SELECT note FROM conflict_resolutions_v2").fetchall()
        self.assertEqual([row[0] for row in notes], ["必须保留的未知数据"])

    def test_legacy_database_upgrade_and_new_receipts_persist_after_disk_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            messages = MessageStore(path)
            try:
                facts = FactStore(messages)
                sources = self.seed(messages, facts)["u1"]
                conflicts = ConflictStore(facts)
                old_rows = self.install_legacy_receipts(messages, facts, conflicts, sources)
            finally:
                messages.close()
            messages = MessageStore(path)
            try:
                facts = FactStore(messages)
                conflicts = ConflictStore(facts)
                self.assertEqual(self.receipt_rows(messages.conn), old_rows)
                target = {"fact_id": "other", "version": 1}
                candidate = {**self.candidate, "predicate": "works_in", "sources": [sources[1]]}
                conflicts.record_conflict(user_id="u1", conflict_id="disk-transition", candidate=candidate,
                                          related_facts=[target], reason="升级后持久争议")
                receipt = conflicts.resolve_conflict(user_id="u1", conflict_id="disk-transition", outcome="TRANSITION",
                                                     target=target, new_fact_id="disk-new", reason="升级后持久处理")
            finally:
                messages.close()
            messages = MessageStore(path)
            try:
                facts = FactStore(messages)
                conflicts = ConflictStore(facts)
                changes = messages.conn.total_changes
                self.assertEqual(conflicts.get_resolution(user_id="u1", conflict_id="disk-transition"), receipt)
                self.assertEqual(conflicts.resolve_conflict(user_id="u1", conflict_id="disk-transition"), receipt)
                self.assertEqual(messages.conn.total_changes, changes)
                retained = tuple(row for row in self.receipt_rows(messages.conn) if row[1].startswith("legacy-"))
                self.assertEqual(retained, old_rows)
                self.assertEqual(facts.get_fact(user_id="u1", fact_id="disk-new"), receipt["result"]["fact"])
                self.assertIsNone(facts.get_fact(user_id="u1", fact_id="disk-new", version=2))
            finally:
                messages.close()

    def test_public_retract_and_transition_still_commit_for_independent_disk_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            messages = MessageStore(path)
            try:
                facts = FactStore(messages)
                sources = self.seed(messages, facts)["u1"]
                facts.retract_fact(user_id="u1", fact_id="home", expected_version=1)
                reader = sqlite3.connect(path)
                try:
                    self.assertEqual(reader.execute("SELECT status FROM facts WHERE user_id = 'u1' AND fact_id = 'home' AND version = 1").fetchone()[0], "retracted")
                finally:
                    reader.close()
                facts.transition_fact(user_id="u1", fact_id="other", new_version=2, new_fact_id="after",
                                      effective_at=self.candidate["valid_from"], object=self.candidate["object"], sources=[sources[1]])
                reader = sqlite3.connect(path)
                try:
                    self.assertEqual(reader.execute("SELECT version, status FROM facts WHERE user_id = 'u1' AND fact_id = 'other' ORDER BY version").fetchall(),
                                     [(1, "superseded"), (2, "active")])
                    self.assertEqual(reader.execute("SELECT status FROM facts WHERE user_id = 'u1' AND fact_id = 'after' AND version = 1").fetchone()[0], "active")
                    self.assertEqual(reader.execute("SELECT COUNT(*) FROM fact_transitions WHERE user_id = 'u1' AND new_fact_id = 'after'").fetchone()[0], 1)
                finally:
                    reader.close()
            finally:
                messages.close()


if __name__ == "__main__":
    unittest.main()
