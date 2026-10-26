"""显式时间变化的只读规划，以及统一执行器与 SQLite 的衔接。"""

from copy import deepcopy
import unittest

from src.fact_relation_judge import judge_fact_relation
from src.fact_store import FactStore
from src.memory_updater import MemoryUpdater
from src.message_store import MessageStore


class FactTemporalPlanTests(unittest.TestCase):
    def setUp(self):
        self.source = {
            "user_id": "u1", "session_id": "history", "message_id": "m1",
        }
        self.new_source = {**self.source, "message_id": "m2"}
        self.candidate = {
            "intent": "change",
            "fact_id": "hangzhou",
            "subject_id": "user:u1",
            "predicate": "lives_in",
            "object": {"type": "literal", "value": "杭州"},
            "sources": [self.new_source],
            "valid_from": "2026-10-01T08:00:00+08:00",
            "target": {"fact_id": "beijing", "version": 1},
        }

    def fact(self, **changes):
        return deepcopy({
            "user_id": "u1",
            "fact_id": "beijing",
            "version": 1,
            "subject_id": "user:u1",
            "predicate": "lives_in",
            "object": {"type": "literal", "value": "北京"},
            "sources": [self.source],
            "scope": "user",
            "session_id": None,
            "valid_from": "2026-01-01T00:00:00+00:00",
            "valid_to": None,
            "status": "active",
            "recorded_at": "2026-10-21T00:00:00+00:00",
            **changes,
        })

    def plan(self, candidate=None, facts=None, cardinality="unknown"):
        candidate = self.candidate if candidate is None else candidate
        facts = [self.fact()] if facts is None else facts
        before_candidate, before_facts = deepcopy(candidate), deepcopy(facts)
        result = judge_fact_relation(
            user_id="u1", candidate=candidate, existing_facts=facts,
            cardinality=cardinality,
        )
        self.assertEqual(candidate, before_candidate)
        self.assertEqual(facts, before_facts)
        return result

    def assert_transition(self, result):
        self.assertEqual(result["relation"], "TEMPORAL_CHANGE")
        self.assertEqual(result["target"], {"fact_id": "beijing", "version": 1})
        self.assertEqual(result["decision"]["action"], "TRANSITION")

    def assert_deferred(self, result, relation):
        self.assertEqual(result["relation"], relation)
        self.assertIsNone(result["decision"])

    def test_complete_explicit_change_builds_fixed_transition_and_normalizes_times(self):
        candidate = {
            **self.candidate,
            "valid_to": "2027-01-01T08:00:00+08:00",
        }
        result = self.plan(candidate)
        self.assert_transition(result)
        self.assertEqual(result["decision"], {
            "action": "TRANSITION",
            "fact_id": "beijing",
            "new_version": 2,
            "new_fact_id": "hangzhou",
            "effective_at": "2026-10-01T00:00:00+00:00",
            "valid_to": "2027-01-01T00:00:00+00:00",
            "object": candidate["object"],
            "sources": [self.new_source],
        })

    def test_incomplete_change_stays_deferred_without_guessing_target_id_or_time(self):
        for missing in ("target", "fact_id", "valid_from"):
            candidate = deepcopy(self.candidate)
            candidate.pop(missing)
            with self.subTest(missing=missing):
                self.assert_deferred(self.plan(candidate), "TEMPORAL_CHANGE")
        self.assert_deferred(
            self.plan({**self.candidate, "valid_from": None}), "TEMPORAL_CHANGE",
        )

    def test_target_must_be_latest_active_and_match_exact_context(self):
        for target in (
            self.fact(version=2),
            self.fact(status="retracted"),
            self.fact(status="superseded"),
            self.fact(subject_id="user:someone-else"),
            self.fact(predicate="works_in"),
            self.fact(scope="session", session_id="chat-a"),
            self.fact(fact_id="another"),
        ):
            with self.subTest(target=target):
                self.assert_deferred(self.plan(facts=[target]), "UNRESOLVED")
        self.assert_deferred(self.plan(facts=[]), "UNRESOLVED")
        candidate = {**self.candidate, "scope": "session", "session_id": "chat-a"}
        self.assert_deferred(
            self.plan(candidate, [self.fact(scope="session", session_id="chat-b")]),
            "UNRESOLVED",
        )

    def test_change_point_must_be_strictly_inside_old_interval(self):
        target = self.fact(valid_to="2026-11-01T00:00:00+00:00")
        for effective_at in (
            "2025-12-31T23:59:59+00:00",
            "2026-01-01T00:00:00+00:00",
            "2026-11-01T00:00:00+00:00",
            "2026-12-01T00:00:00+00:00",
        ):
            with self.subTest(effective_at=effective_at):
                self.assert_deferred(
                    self.plan({**self.candidate, "valid_from": effective_at}, [target]),
                    "UNRESOLVED",
                )
        self.assert_transition(self.plan(facts=[self.fact(valid_from=None)]))

    def test_used_new_id_never_becomes_an_implicit_retry_or_new_target(self):
        for status in ("active", "superseded", "retracted"):
            occupied = self.fact(fact_id="hangzhou", status=status, predicate="unrelated")
            with self.subTest(status=status):
                self.assert_deferred(
                    self.plan(facts=[self.fact(), occupied]), "UNRESOLVED",
                )
        self.assert_deferred(
            self.plan({**self.candidate, "fact_id": "beijing"}), "UNRESOLVED",
        )

    def test_other_future_interval_blocks_single_or_unknown_policy(self):
        future = self.fact(
            fact_id="future", valid_from="2026-12-01T00:00:00+00:00",
        )
        for cardinality in ("single", "unknown"):
            with self.subTest(cardinality=cardinality):
                self.assert_deferred(
                    self.plan(facts=[self.fact(), future], cardinality=cardinality),
                    "UNRESOLVED",
                )
        # 即使同值也不擅自合并另一个事实的来源或有效区间。
        future["object"] = deepcopy(self.candidate["object"])
        self.assert_deferred(self.plan(facts=[self.fact(), future]), "UNRESOLVED")

    def test_adjacent_intervals_and_other_contexts_do_not_block_transition(self):
        past = self.fact(
            fact_id="past", valid_from=None,
            valid_to="2026-10-01T00:00:00+00:00",
        )
        self.assert_transition(self.plan(facts=[self.fact(), past]))
        candidate = {
            **self.candidate, "valid_to": "2027-01-01T00:00:00+00:00",
        }
        future = self.fact(
            fact_id="future", valid_from="2027-01-01T00:00:00+00:00",
        )
        self.assert_transition(self.plan(candidate, [self.fact(), future]))
        for changes in (
            {"subject_id": "user:someone-else"},
            {"predicate": "works_in"},
            {"scope": "session", "session_id": "another-chat"},
        ):
            unrelated = self.fact(fact_id="other", **changes)
            with self.subTest(changes=changes):
                self.assert_transition(self.plan(facts=[self.fact(), unrelated]))

    def test_inactive_latest_interval_does_not_block_and_multivalue_keeps_other_targets(self):
        future = self.fact(fact_id="future", valid_from="2026-12-01T00:00:00+00:00")
        withdrawn = {**future, "version": 2, "status": "retracted"}
        self.assert_transition(self.plan(facts=[future, self.fact(), withdrawn]))
        result = self.plan(facts=[self.fact(), future], cardinality="multi")
        self.assert_transition(result)
        self.assertEqual(result["decision"]["fact_id"], "beijing")
        self.assertNotIn("future", result["decision"].values())

    def test_new_interval_is_independent_and_explicit_same_value_change_is_allowed(self):
        target = self.fact(valid_to="2026-11-01T00:00:00+00:00")
        self.assertIsNone(self.plan(facts=[target])["decision"]["valid_to"])
        result = self.plan(
            {**self.candidate, "valid_to": "2027-01-01T00:00:00+00:00"},
            [target],
        )
        self.assert_transition(result)
        self.assertEqual(result["decision"]["valid_to"], "2027-01-01T00:00:00+00:00")
        self.assert_transition(self.plan({**self.candidate, "object": target["object"]}))

    def test_plain_assertion_does_not_gain_transition_authority_from_different_interval(self):
        candidate = deepcopy(self.candidate)
        candidate.pop("intent")
        candidate.pop("target")
        self.assert_deferred(self.plan(candidate), "TEMPORAL_CHANGE")

    def test_result_does_not_share_candidate_or_snapshot_mutable_values(self):
        candidate = deepcopy(self.candidate)
        facts = [self.fact()]
        before_candidate, before_facts = deepcopy(candidate), deepcopy(facts)
        result = self.plan(candidate, facts)
        result["decision"]["object"]["value"] = "changed"
        result["decision"]["sources"][0]["message_id"] = "changed"
        result["target"]["version"] = 99
        self.assertEqual(candidate, before_candidate)
        self.assertEqual(facts, before_facts)


class FactTemporalExecutionTests(unittest.TestCase):
    def setUp(self):
        self.messages = MessageStore(":memory:")
        self.addCleanup(self.messages.close)
        self.facts = FactStore(self.messages)
        self.updater = MemoryUpdater(self.facts)
        self.sources = []
        for message_id in ("m1", "m2", "m3"):
            source = {"user_id": "u1", "session_id": "history", "message_id": message_id}
            self.messages.append_message(
                **source, occurred_at="2026-10-21T10:00:00+08:00",
                payload={"role": "user", "content": f"事实来源 {message_id}"},
            )
            self.sources.append(source)
        self.first = self.facts.add_fact(
            user_id="u1", fact_id="beijing", subject_id="user:u1", predicate="lives_in",
            object={"type": "literal", "value": "北京"}, sources=[self.sources[0]],
            scope="session", session_id="current-chat",
            valid_from="2026-01-01T08:00:00+08:00",
        )
        self.candidate = {
            "intent": "change", "fact_id": "hangzhou",
            "subject_id": "user:u1", "predicate": "lives_in",
            "object": {"type": "literal", "value": "杭州"},
            "sources": [self.sources[1]],
            "scope": "session", "session_id": "current-chat",
            "valid_from": "2026-10-01T08:00:00+08:00",
            "target": {"fact_id": "beijing", "version": 1},
        }

    def plan(self, candidate=None):
        candidate = self.candidate if candidate is None else candidate
        snapshot = [self.first]
        before_candidate, before_snapshot = deepcopy(candidate), deepcopy(snapshot)
        changes = self.messages.conn.total_changes
        result = judge_fact_relation(user_id="u1", candidate=candidate, existing_facts=snapshot)
        self.assertEqual(candidate, before_candidate)
        self.assertEqual(snapshot, before_snapshot)
        self.assertEqual(self.messages.conn.total_changes, changes)
        self.assertEqual(result["decision"]["action"], "TRANSITION")
        return result["decision"]

    def test_plan_executes_both_facts_and_preserves_original_version_and_time_boundary(self):
        result = self.updater.apply(user_id="u1", decision=self.plan())
        self.assertEqual(set(result), {"action", "previous_fact", "fact"})
        self.assertEqual(result["action"], "TRANSITION")
        previous, new = result["previous_fact"], result["fact"]
        self.assertEqual(previous["fact_id"], "beijing")
        self.assertEqual(previous["version"], 2)
        self.assertEqual(previous["status"], "active")
        self.assertEqual(previous["object"], self.first["object"])
        self.assertEqual(previous["sources"], self.sources[:2])
        self.assertEqual(previous["valid_from"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(previous["valid_to"], "2026-10-01T00:00:00+00:00")
        self.assertEqual(new["fact_id"], "hangzhou")
        self.assertEqual(new["version"], 1)
        self.assertEqual(new["status"], "active")
        self.assertEqual(new["object"], self.candidate["object"])
        self.assertEqual(new["sources"], [self.sources[1]])
        self.assertEqual(new["scope"], "session")
        self.assertEqual(new["session_id"], "current-chat")
        self.assertEqual(new["valid_from"], "2026-10-01T00:00:00+00:00")
        self.assertIsNone(new["valid_to"])
        self.assertEqual(
            self.facts.get_fact(user_id="u1", fact_id="beijing", version=1),
            {**self.first, "status": "superseded"},
        )
        self.assertEqual(self.first["status"], "active")
        self.assertEqual(
            self.facts.list_facts(user_id="u1", session_id="current-chat",
                                  as_of="2026-09-30T23:59:59.999999+00:00"),
            [previous],
        )
        self.assertEqual(
            self.facts.list_facts(user_id="u1", session_id="current-chat",
                                  as_of="2026-10-01T00:00:00+00:00"),
            [new],
        )
        self.assertEqual(self.facts.list_facts(user_id="u1"), [])
        self.assertIsNone(self.facts.get_latest_fact(user_id="u2", fact_id="hangzhou"))

    def test_receipt_retry_needs_only_stable_new_id_and_never_reapplies_after_later_writes(self):
        decision = self.plan()
        self.updater.apply(user_id="u1", decision=decision)
        self.facts.update_fact(
            user_id="u1", fact_id="beijing", new_version=3,
            object={"type": "literal", "value": "北京朝阳区"}, sources=[self.sources[2]],
        )
        self.facts.update_fact(
            user_id="u1", fact_id="hangzhou", new_version=2,
            object={"type": "literal", "value": "杭州西湖区"}, sources=[self.sources[2]],
        )
        withdrawn = self.facts.retract_fact(user_id="u1", fact_id="hangzhou", expected_version=2)
        expected = {
            "action": "TRANSITION",
            "previous_fact": self.facts.get_fact(user_id="u1", fact_id="beijing", version=2),
            "fact": self.facts.get_fact(user_id="u1", fact_id="hangzhou", version=1),
        }
        changes = self.messages.conn.total_changes
        for retry in (
            {"action": "TRANSITION", "new_fact_id": "hangzhou"},
            {
                **decision, "fact_id": None, "new_version": False, "object": None,
                "sources": [], "effective_at": "invalid", "valid_to": "invalid",
            },
        ):
            with self.subTest(retry=retry):
                self.assertEqual(self.updater.apply(user_id="u1", decision=retry), expected)
                self.assertEqual(self.messages.conn.total_changes, changes)
        self.assertEqual(expected["previous_fact"]["status"], "superseded")
        self.assertEqual(expected["fact"]["status"], "superseded")
        self.assertEqual(self.facts.get_latest_fact(user_id="u1", fact_id="hangzhou"), withdrawn)
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="beijing", version=4))
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="hangzhou", version=3))

    def test_unknown_control_fields_are_rejected_and_missing_real_source_rolls_back(self):
        decision = self.plan()
        changes = self.messages.conn.total_changes
        for field in ("user_id", "subject_id", "predicate", "scope", "version", "status", "target", "valid_from"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.updater.apply(user_id="u1", decision={**decision, field: "override"})
        self.assertEqual(self.messages.conn.total_changes, changes)
        bad_source = {**self.sources[1], "message_id": "missing"}
        decision = self.plan({**self.candidate, "sources": [bad_source]})
        with self.assertRaises(ValueError):
            self.updater.apply(user_id="u1", decision=decision)
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="beijing", version=1), self.first)
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="beijing", version=2))
        self.assertIsNone(self.facts.get_latest_fact(user_id="u1", fact_id="hangzhou"))


if __name__ == "__main__":
    unittest.main()
