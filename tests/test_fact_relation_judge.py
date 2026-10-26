"""事实关系的只读规划、保守决策及现有执行器衔接测试。"""

from copy import deepcopy
import unittest

from src.fact_relation_judge import judge_fact_relation
from src.fact_store import FactStore
from src.memory_updater import MemoryUpdater
from src.message_store import MessageStore


_UNSET = object()


class FactRelationJudgeTests(unittest.TestCase):
    def setUp(self):
        self.source = {
            "user_id": "u1", "session_id": "history", "message_id": "m1",
        }
        self.new_source = {**self.source, "message_id": "m2"}
        self.candidate = {
            "fact_id": "new",
            "subject_id": "user:u1",
            "predicate": "prefers_drink",
            "object": {"type": "literal", "value": "咖啡"},
            "sources": [self.new_source],
        }

    def fact(self, **changes):
        return deepcopy({
            "user_id": "u1",
            "fact_id": "old",
            "version": 1,
            "subject_id": "user:u1",
            "predicate": "prefers_drink",
            "object": {"type": "literal", "value": "咖啡"},
            "sources": [self.source],
            "scope": "user",
            "session_id": None,
            "valid_from": None,
            "valid_to": None,
            "status": "active",
            "recorded_at": "2026-10-21T00:00:00+00:00",
            **changes,
        })

    def judge(self, candidate=_UNSET, facts=_UNSET, cardinality="unknown"):
        return judge_fact_relation(
            user_id="u1",
            candidate=self.candidate if candidate is _UNSET else candidate,
            existing_facts=[] if facts is _UNSET else facts,
            cardinality=cardinality,
        )

    def assert_plan(self, result, relation, action=None, target=None):
        self.assertEqual(set(result), {"relation", "target", "reason", "decision"})
        self.assertEqual(result["relation"], relation)
        self.assertEqual(result["target"], target)
        self.assertIsInstance(result["reason"], str)
        if action is None:
            self.assertIsNone(result["decision"])
        else:
            self.assertEqual(result["decision"]["action"], action)

    def correction(self, **changes):
        candidate = {
            **self.candidate,
            "intent": "correct",
            "object": {"type": "literal", "value": "茶"},
            "target": {"fact_id": "old", "version": 1},
            **changes,
        }
        candidate.pop("fact_id")
        return candidate

    def retraction(self, **changes):
        candidate = {**self.correction(), "intent": "retract", **changes}
        candidate.pop("object", None)
        return candidate

    def test_supplement_ignores_unrelated_subject_predicate_and_scope(self):
        for facts in (
            [],
            [self.fact(subject_id="user:someone-else")],
            [self.fact(predicate="likes_food")],
            [self.fact(scope="session", session_id="other-chat")],
        ):
            with self.subTest(facts=facts):
                result = self.judge(facts=facts)
                self.assert_plan(result, "SUPPLEMENT", "ADD")
                decision = result["decision"]
                for field in ("fact_id", "subject_id", "predicate", "object", "sources"):
                    self.assertEqual(decision[field], self.candidate[field])
                self.assertNotIn("user_id", decision)
                self.assertNotIn("intent", decision)

        candidate = {**self.candidate, "scope": "session", "session_id": "current-chat"}
        self.assert_plan(
            self.judge(candidate, [self.fact()]), "SUPPLEMENT", "ADD",
        )
        self.assert_plan(
            self.judge(candidate, [self.fact(scope="session", session_id="other-chat")]),
            "SUPPLEMENT", "ADD",
        )

    def test_duplicate_uses_stored_json_shape_normalized_times_and_not_sources(self):
        candidate = {
            **self.candidate,
            "object": {"value": {"b": (True, 1), "a": "茶"}, "type": "literal"},
            "valid_from": "2026-01-01T08:00:00+08:00",
            "valid_to": "2027-01-01T08:00:00+08:00",
        }
        old = self.fact(
            object={"type": "literal", "value": {"a": "茶", "b": [True, 1]}},
            valid_from="2026-01-01T00:00:00+00:00",
            valid_to="2027-01-01T00:00:00+00:00",
        )
        result = self.judge(candidate, [old])
        self.assert_plan(
            result, "DUPLICATE", "NOOP", {"fact_id": "old", "version": 1},
        )
        self.assertEqual(result["decision"], {"action": "NOOP"})
        self.assertEqual(old["sources"], [self.source])

    def test_different_values_require_explicit_multivalue_policy(self):
        old = self.fact(object={"type": "literal", "value": "茶"})
        for cardinality in ("single", "unknown"):
            with self.subTest(cardinality=cardinality):
                self.assert_plan(
                    self.judge(facts=[old], cardinality=cardinality), "UNRESOLVED",
                )
        result = self.judge(facts=[old], cardinality="multi")
        self.assert_plan(result, "MULTIVALUE", "ADD")
        self.assertEqual(result["decision"]["fact_id"], "new")
        self.assertEqual(result["decision"]["object"], self.candidate["object"])

    def test_json_boolean_and_number_are_distinct_even_inside_objects(self):
        for left, right in ((True, 1), ({"x": [True]}, {"x": [1]})):
            candidate = {**self.candidate, "object": {"type": "literal", "value": left}}
            old = self.fact(object={"type": "literal", "value": right})
            with self.subTest(left=left, right=right):
                self.assert_plan(self.judge(candidate, [old]), "UNRESOLVED")
                self.assert_plan(self.judge(candidate, [old], "multi"), "MULTIVALUE", "ADD")

    def test_mixed_values_do_not_hide_single_value_conflict(self):
        matching = self.fact()
        other = self.fact(fact_id="other", object={"type": "literal", "value": "茶"})
        for cardinality in ("single", "unknown"):
            with self.subTest(cardinality=cardinality):
                self.assert_plan(
                    self.judge(facts=[matching, other], cardinality=cardinality),
                    "UNRESOLVED",
                )
        self.assert_plan(
            self.judge(facts=[matching, other], cardinality="multi"),
            "DUPLICATE", "NOOP", {"fact_id": "old", "version": 1},
        )
        self.assert_plan(
            self.judge(facts=[matching, self.fact(fact_id="same")]),
            "DUPLICATE", "NOOP",
        )

    def test_interval_changes_are_deferred_even_for_same_value_or_multivalue(self):
        for changes in (
            {"valid_from": "2026-01-01T00:00:00+00:00"},
            {"valid_to": "2025-01-01T00:00:00+00:00"},
            {
                "valid_from": "2024-01-01T00:00:00+00:00",
                "valid_to": "2025-01-01T00:00:00+00:00",
            },
        ):
            for cardinality in ("unknown", "multi"):
                with self.subTest(changes=changes, cardinality=cardinality):
                    self.assert_plan(
                        self.judge(facts=[self.fact(**changes)], cardinality=cardinality),
                        "TEMPORAL_CHANGE",
                    )
        self.assert_plan(
            self.judge(facts=[
                self.fact(),
                self.fact(fact_id="past", valid_to="2025-01-01T00:00:00+00:00"),
            ]),
            "TEMPORAL_CHANGE",
        )

    def test_latest_inactive_version_never_restores_old_active_fact(self):
        for status in ("retracted", "superseded"):
            old = self.fact()
            latest = self.fact(version=2, status=status)
            for snapshot in ([old, latest], [latest, old]):
                with self.subTest(status=status, snapshot=snapshot):
                    self.assert_plan(self.judge(facts=snapshot), "SUPPLEMENT", "ADD")
                    self.assert_plan(
                        self.judge(self.correction(), snapshot), "UNRESOLVED",
                    )

    def test_assert_reused_id_is_not_a_new_plan_or_an_implicit_retry(self):
        for changes in ({}, {"status": "retracted"}, {"predicate": "unrelated"}):
            with self.subTest(changes=changes):
                result = self.judge(facts=[self.fact(fact_id="new", **changes)])
                self.assertEqual(result["relation"], "UNRESOLVED")
                self.assertIsNone(result["decision"])

    def test_explicit_correction_builds_update_for_exact_target(self):
        candidate = self.correction()
        result = self.judge(candidate, [self.fact()])
        self.assert_plan(
            result, "CORRECTION", "UPDATE", {"fact_id": "old", "version": 1},
        )
        self.assertEqual(result["decision"], {
            "action": "UPDATE",
            "fact_id": "old",
            "new_version": 2,
            "object": candidate["object"],
            "sources": candidate["sources"],
        })

    def test_missing_target_is_unresolved_even_if_only_one_fact_matches(self):
        for candidate in (self.correction(), self.retraction()):
            candidate.pop("target")
            with self.subTest(intent=candidate["intent"]):
                self.assert_plan(self.judge(candidate, [self.fact()]), "UNRESOLVED")

    def test_target_never_falls_back_to_another_matching_fact(self):
        for snapshot in (
            [],
            [self.fact(fact_id="another")],
            [self.fact(version=2)],
            [self.fact(), self.fact(version=2)],
            [self.fact(status="retracted")],
            [self.fact(status="superseded")],
            [self.fact(subject_id="user:someone-else")],
            [self.fact(predicate="likes_food")],
            [self.fact(scope="session", session_id="current-chat")],
        ):
            for candidate in (self.correction(), self.retraction()):
                with self.subTest(snapshot=snapshot, intent=candidate["intent"]):
                    result = self.judge(candidate, snapshot)
                    self.assertEqual(result["relation"], "UNRESOLVED")
                    self.assertIsNone(result["decision"])

    def test_correction_same_value_is_noop_and_changed_interval_is_deferred(self):
        target = {"fact_id": "old", "version": 1}
        self.assert_plan(
            self.judge(self.correction(object=self.candidate["object"]), [self.fact()]),
            "DUPLICATE", "NOOP", target,
        )
        result = self.judge(
            self.correction(valid_from="2026-01-01T00:00:00+00:00"),
            [self.fact()],
        )
        self.assertEqual(result["relation"], "TEMPORAL_CHANGE")
        self.assertIsNone(result["decision"])

    def test_retraction_keeps_explicit_target_version_without_matching_times(self):
        result = self.judge(
            self.retraction(valid_from="2026-01-01T00:00:00+00:00"),
            [self.fact()],
        )
        self.assert_plan(
            result, "RETRACTION", "DELETE", {"fact_id": "old", "version": 1},
        )
        self.assertEqual(result["decision"], {
            "action": "DELETE", "fact_id": "old", "expected_version": 1,
        })

    def test_explicit_change_stays_deferred_and_validates_optional_target(self):
        candidate = {**self.candidate, "intent": "change"}
        candidate.pop("fact_id")
        self.assert_plan(self.judge(candidate), "TEMPORAL_CHANGE")
        candidate["target"] = {"fact_id": "old", "version": 1}
        result = self.judge(candidate, [self.fact()])
        self.assertEqual(result["relation"], "TEMPORAL_CHANGE")
        self.assertIsNone(result["decision"])
        result = self.judge(candidate, [self.fact(version=2)])
        self.assertEqual(result["relation"], "UNRESOLVED")
        self.assertIsNone(result["decision"])

    def test_planning_and_result_mutation_leave_candidate_and_snapshot_unchanged(self):
        candidate = deepcopy(self.candidate)
        candidate["object"]["value"] = {"nested": ["咖啡"]}
        snapshot = [self.fact(predicate="unrelated")]
        before_candidate, before_snapshot = deepcopy(candidate), deepcopy(snapshot)
        result = self.judge(candidate, snapshot)
        self.assertEqual(candidate, before_candidate)
        self.assertEqual(snapshot, before_snapshot)
        result["decision"]["object"]["value"]["nested"].append("茶")
        result["decision"]["sources"][0]["message_id"] = "changed"
        self.assertEqual(candidate, before_candidate)
        self.assertEqual(snapshot, before_snapshot)

        candidate = self.correction()
        before_candidate = deepcopy(candidate)
        result = self.judge(candidate, [self.fact()])
        result["target"]["version"] = 99
        self.assertEqual(candidate, before_candidate)

    def test_invalid_envelopes_policies_and_candidate_control_fields_are_rejected(self):
        for user_id in (None, True, "", " \t"):
            with self.subTest(user_id=user_id), self.assertRaises(ValueError):
                judge_fact_relation(user_id=user_id, candidate=self.candidate, existing_facts=[])
        for candidate in (None, [], "assert", {}):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                self.judge(candidate=candidate)
        for cardinality in (None, True, [], "many"):
            with self.subTest(cardinality=cardinality), self.assertRaises(ValueError):
                self.judge(cardinality=cardinality)
        for field in ("user_id", "version", "status", "action", "unexpected"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.judge({**self.candidate, field: "override"})
        for changes in (
            {"intent": "UPDATE"},
            {"target": {"fact_id": "old", "version": 1}},
            {"scope": "session", "session_id": None},
            {"session_id": "wrong-for-user-scope"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.judge({**self.candidate, **changes})

    def test_invalid_candidate_evidence_objects_intervals_and_targets_are_rejected(self):
        for changes in (
            {"sources": []},
            {"sources": [{**self.source, "user_id": "u2"}]},
            {"sources": [{"user_id": "u1", "message_id": "m1"}]},
            {"object": {"type": "literal", "value": float("nan")}},
            {"object": {"type": "entity", "value": "entity:1"}},
            {"valid_from": "2026-01-01T00:00:00"},
            {
                "valid_from": "2026-01-01T00:00:00+00:00",
                "valid_to": "2026-01-01T00:00:00+00:00",
            },
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.judge({**self.candidate, **changes})
        for target in (
            "old", {}, {"fact_id": "old", "version": True},
            {"fact_id": "old", "version": 0},
            {"fact_id": "old", "version": 1, "user_id": "u2"},
        ):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.judge(self.correction(target=target), [self.fact()])

    def test_invalid_snapshots_are_rejected_instead_of_silently_ignored(self):
        missing_subject = self.fact()
        missing_subject.pop("subject_id")
        for snapshot in (
            None, "not-a-list", {}, [None], [missing_subject],
            [self.fact(user_id="u2")],
            [self.fact(version=True)],
            [self.fact(status="deleted")],
            [self.fact(scope="session", session_id=None)],
            [self.fact(valid_from="2026-01-01T00:00:00")],
            [self.fact(), self.fact()],
        ):
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                self.judge(facts=snapshot)


class FactRelationJudgeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.messages = MessageStore(":memory:")
        self.addCleanup(self.messages.close)
        self.facts = FactStore(self.messages)
        self.updater = MemoryUpdater(self.facts)
        self.sources = []
        for message_id in ("m1", "m2", "m3"):
            source = {"user_id": "u1", "session_id": "history", "message_id": message_id}
            self.messages.append_message(
                **source,
                occurred_at="2026-10-21T10:00:00+08:00",
                payload={"role": "user", "content": f"事实来源 {message_id}"},
            )
            self.sources.append(source)
        self.candidate = {
            "fact_id": "f1",
            "subject_id": "user:u1",
            "predicate": "prefers_drink",
            "object": {"type": "literal", "value": "咖啡"},
            "sources": [self.sources[0]],
            "scope": "session",
            "session_id": "current-chat",
            "valid_from": "2026-01-01T08:00:00+08:00",
            "valid_to": "2027-01-01T08:00:00+08:00",
        }

    def plan(self, candidate, snapshot):
        before_candidate, before_snapshot = deepcopy(candidate), deepcopy(snapshot)
        changes = self.messages.conn.total_changes
        result = judge_fact_relation(
            user_id="u1", candidate=candidate, existing_facts=snapshot,
        )
        self.assertEqual(candidate, before_candidate)
        self.assertEqual(snapshot, before_snapshot)
        self.assertEqual(self.messages.conn.total_changes, changes)
        return result["decision"]

    def test_add_update_delete_plans_execute_and_fixed_decisions_retry(self):
        add = self.plan(self.candidate, [])
        first = self.updater.apply(user_id="u1", decision=add)["fact"]
        changes = self.messages.conn.total_changes
        self.assertEqual(self.updater.apply(user_id="u1", decision=add)["fact"], first)
        self.assertEqual(self.messages.conn.total_changes, changes)

        duplicate = self.plan(
            {**self.candidate, "fact_id": "duplicate", "sources": [self.sources[1]]},
            [first],
        )
        self.assertEqual(duplicate, {"action": "NOOP"})
        self.assertEqual(self.updater.apply(user_id="u1", decision=duplicate), {
            "action": "NOOP", "fact": None,
        })
        self.assertEqual(self.messages.conn.total_changes, changes)
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="f1")["sources"], [self.sources[0]])

        correction = {
            **self.candidate,
            "intent": "correct",
            "object": {"type": "literal", "value": "茶"},
            "sources": [self.sources[1]],
            "target": {"fact_id": "f1", "version": 1},
        }
        correction.pop("fact_id")
        update = self.plan(correction, [first])
        self.assertEqual(update["new_version"], 2)
        second = self.updater.apply(user_id="u1", decision=update)["fact"]
        self.assertEqual(second["object"], correction["object"])
        self.assertEqual(second["sources"], [self.sources[1]])
        self.assertEqual(second["scope"], "session")
        self.assertEqual(second["session_id"], "current-chat")
        self.assertEqual(second["valid_from"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(second["valid_to"], "2027-01-01T00:00:00+00:00")
        changes = self.messages.conn.total_changes
        self.assertEqual(self.updater.apply(user_id="u1", decision=update)["fact"], second)
        self.assertEqual(self.messages.conn.total_changes, changes)

        retraction = {
            **correction,
            "intent": "retract",
            "sources": [self.sources[2]],
            "target": {"fact_id": "f1", "version": 2},
        }
        retraction.pop("object")
        first_history = self.facts.get_fact(user_id="u1", fact_id="f1", version=1)
        delete = self.plan(retraction, [first_history, second])
        self.assertEqual(delete["expected_version"], 2)
        retracted = self.updater.apply(user_id="u1", decision=delete)["fact"]
        self.assertEqual(retracted["status"], "retracted")
        changes = self.messages.conn.total_changes
        self.assertEqual(self.updater.apply(user_id="u1", decision=delete)["fact"], retracted)
        self.assertEqual(self.updater.apply(user_id="u1", decision=update)["fact"], retracted)
        retried_add = self.updater.apply(user_id="u1", decision=add)["fact"]
        self.assertEqual(retried_add["version"], 1)
        self.assertEqual(retried_add["status"], "superseded")
        self.assertEqual(self.messages.conn.total_changes, changes)
        self.assertEqual(update["new_version"], 2)
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="f1", version=3))
        self.assertEqual(
            self.facts.get_fact(user_id="u1", fact_id="f1", version=1)["status"],
            "superseded",
        )
        self.assertEqual(
            self.facts.list_facts(
                user_id="u1", session_id="current-chat",
                as_of="2026-10-21T00:00:00+00:00",
            ),
            [],
        )
        self.assertIsNone(self.facts.get_latest_fact(user_id="u2", fact_id="f1"))

    def test_planner_checks_source_shape_but_store_checks_message_existence(self):
        candidate = {
            **self.candidate,
            "sources": [{**self.sources[0], "message_id": "missing"}],
        }
        decision = self.plan(candidate, [])
        self.assertEqual(decision["action"], "ADD")
        with self.assertRaises(ValueError):
            self.updater.apply(user_id="u1", decision=decision)
        self.assertIsNone(self.facts.get_latest_fact(user_id="u1", fact_id="f1"))


if __name__ == "__main__":
    unittest.main()
