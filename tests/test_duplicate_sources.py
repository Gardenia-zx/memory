"""重复事实的可选来源合并、默认兼容及已有 UPDATE 链路的独立测试。"""

from copy import deepcopy
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.fact_relation_judge import judge_fact_relation
from src.fact_store import FactStore
from src.memory_updater import MemoryUpdater
from src.message_store import MessageStore


class DuplicateSourcePlanTests(unittest.TestCase):
    def setUp(self):
        self.old_source = {"user_id": "u1", "session_id": "a", "message_id": "m1"}
        self.other_session = {**self.old_source, "session_id": "b"}
        self.new_source = {**self.old_source, "message_id": "m2"}
        self.candidate = {
            "fact_id": "candidate-new", "subject_id": "user:u1", "predicate": "prefers_drink",
            "object": {"type": "literal", "value": "咖啡"}, "sources": [self.new_source],
        }

    def fact(self, **changes):
        return deepcopy({
            "user_id": "u1", "fact_id": "old", "version": 1,
            "subject_id": "user:u1", "predicate": "prefers_drink",
            "object": {"type": "literal", "value": "咖啡"},
            "scope": "user", "session_id": None, "valid_from": None, "valid_to": None,
            "status": "active", "sources": [self.old_source], **changes,
        })

    def plan(self, candidate=None, facts=None, **options):
        return judge_fact_relation(
            user_id="u1", candidate=self.candidate if candidate is None else candidate,
            existing_facts=[self.fact()] if facts is None else facts, **options,
        )

    def assert_merge(self, plan, version=2):
        self.assertEqual(plan["relation"], "DUPLICATE")
        self.assertEqual(plan["target"], {"fact_id": "old", "version": version - 1})
        self.assertEqual(plan["decision"]["action"], "UPDATE")
        self.assertEqual(plan["decision"]["fact_id"], "old")
        self.assertEqual(plan["decision"]["new_version"], version)

    def test_default_duplicate_stays_noop_without_requiring_existing_sources(self):
        missing_sources = self.fact()
        missing_sources.pop("sources")
        for fact in (missing_sources, self.fact(sources=None), self.fact(sources="unused")):
            for options in ({}, {"merge_duplicate_sources": False}):
                with self.subTest(fact=fact, options=options):
                    plan = self.plan(facts=[fact], **options)
                    self.assertEqual(plan["relation"], "DUPLICATE")
                    self.assertEqual(plan["decision"], {"action": "NOOP"})

    def test_enabled_merge_builds_fixed_update_with_old_then_new_unique_sources(self):
        old = self.fact(sources=[self.other_session, self.old_source, self.other_session])
        candidate = {**self.candidate, "sources": [self.old_source, self.new_source, self.other_session, self.new_source]}
        plan = self.plan(candidate, [old], merge_duplicate_sources=True)
        self.assert_merge(plan)
        self.assertEqual(plan["decision"], {
            "action": "UPDATE", "fact_id": "old", "new_version": 2,
            "object": old["object"],
            "sources": [self.other_session, self.old_source, self.new_source],
        })

    def test_already_known_sources_stay_noop_even_if_reordered_or_repeated(self):
        old = self.fact(sources=[self.old_source, self.other_session, self.old_source])
        candidate = {**self.candidate, "sources": [self.other_session, self.old_source, self.other_session]}
        plan = self.plan(candidate, [old], merge_duplicate_sources=True)
        self.assertEqual(plan["relation"], "DUPLICATE")
        self.assertEqual(plan["target"], {"fact_id": "old", "version": 1})
        self.assertEqual(plan["decision"], {"action": "NOOP"})

    def test_source_identity_uses_user_session_message_triple_and_ignores_extra_fields(self):
        candidate = {
            **self.candidate,
            "sources": [{**self.old_source, "label": "已有来源的元字段"}, self.other_session,
                        {**self.other_session, "label": "重复"}],
        }
        plan = self.plan(candidate, merge_duplicate_sources=True)
        self.assert_merge(plan)
        self.assertEqual(plan["decision"]["sources"], [self.old_source, self.other_session])
        self.assertEqual(plan["decision"]["sources"][0]["message_id"],
                         plan["decision"]["sources"][1]["message_id"])

    def test_same_value_explicit_correction_can_merge_only_its_selected_target(self):
        candidate = {**self.candidate, "intent": "correct", "target": {"fact_id": "old", "version": 1}}
        candidate.pop("fact_id")
        another = self.fact(fact_id="another", sources=None)
        plan = self.plan(candidate, [self.fact(), another], merge_duplicate_sources=True)
        self.assert_merge(plan)
        self.assertEqual(plan["decision"]["sources"], [self.old_source, self.new_source])
        self.assertEqual(self.plan(candidate, [self.fact(sources=None)])["decision"], {"action": "NOOP"})

    def test_unrelated_contexts_never_become_merge_targets_or_require_sources(self):
        for changes in ({"subject_id": "user:another"}, {"predicate": "likes_food"},
                        {"scope": "session", "session_id": "another-chat"}):
            with self.subTest(changes=changes):
                plan = self.plan(facts=[self.fact(sources=None, **changes)], merge_duplicate_sources=True)
                self.assertEqual(plan["relation"], "SUPPLEMENT")
                self.assertEqual(plan["decision"]["action"], "ADD")
        candidate = {**self.candidate, "scope": "session", "session_id": "current-chat"}
        plan = self.plan(candidate, [self.fact(scope="session", session_id="another-chat", sources=None)],
                         merge_duplicate_sources=True)
        self.assertEqual(plan["relation"], "SUPPLEMENT")

    def test_duplicate_matching_normalizes_utc_and_json_but_preserves_real_differences(self):
        candidate = {
            **self.candidate, "object": {"value": {"b": (True, 1), "a": "咖啡"}, "type": "literal"},
            "valid_from": "2026-01-01T08:00:00+08:00", "valid_to": "2027-01-01T08:00:00+08:00",
        }
        old = self.fact(
            object={"type": "literal", "value": {"a": "咖啡", "b": [True, 1]}},
            valid_from="2026-01-01T00:00:00+00:00", valid_to="2027-01-01T00:00:00+00:00",
        )
        plan = self.plan(candidate, [old], merge_duplicate_sources=True)
        self.assert_merge(plan)
        self.assertEqual(plan["decision"]["object"], old["object"])
        different_time = {**old, "valid_to": "2028-01-01T00:00:00+00:00", "sources": None}
        plan = self.plan(candidate, [different_time], merge_duplicate_sources=True)
        self.assertEqual(plan["relation"], "TEMPORAL_CHANGE")
        self.assertIsNone(plan["decision"])
        plan = self.plan({**self.candidate, "object": {"type": "literal", "value": True}},
                         [self.fact(object={"type": "literal", "value": 1}, sources=None)],
                         merge_duplicate_sources=True)
        self.assertEqual(plan["relation"], "UNRESOLVED")
        self.assertIsNone(plan["decision"])

    def test_multiple_same_value_fact_ids_are_ambiguous_only_for_enabled_merge(self):
        facts = [self.fact(sources=None), self.fact(fact_id="another", sources=None)]
        normal = self.plan(facts=facts)
        self.assertEqual(normal["relation"], "DUPLICATE")
        self.assertEqual(normal["decision"], {"action": "NOOP"})
        self.assertIsNone(normal["target"])
        plan = self.plan(facts=facts, merge_duplicate_sources=True)
        self.assertEqual(plan["relation"], "UNRESOLVED")
        self.assertIsNone(plan["decision"])

    def test_mixed_values_remain_unresolved_unless_multivalue_has_one_duplicate_target(self):
        duplicate = self.fact()
        different = self.fact(fact_id="other", object={"type": "literal", "value": "茶"}, sources=None)
        for cardinality in ("single", "unknown"):
            with self.subTest(cardinality=cardinality):
                plan = self.plan(facts=[{**duplicate, "sources": None}, different],
                                 cardinality=cardinality, merge_duplicate_sources=True)
                self.assertEqual(plan["relation"], "UNRESOLVED")
                self.assertIsNone(plan["decision"])
        plan = self.plan(facts=[duplicate, different], cardinality="multi", merge_duplicate_sources=True)
        self.assert_merge(plan)
        self.assertEqual(plan["decision"]["sources"], [self.old_source, self.new_source])

    def test_merge_option_requires_a_real_boolean(self):
        for option in (None, 0, 1, "true", [], {}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.plan(merge_duplicate_sources=option)

    def test_enabled_merge_validates_only_selected_target_sources_and_user_ownership(self):
        missing = self.fact()
        missing.pop("sources")
        invalid = [missing]
        invalid += [self.fact(sources=sources) for sources in (
            None, [], "invalid", [None], [{"session_id": "a", "message_id": "m1"}],
            [{**self.old_source, "user_id": "u2"}], [{**self.old_source, "session_id": " "}],
        )]
        for old in invalid:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.plan(facts=[old], merge_duplicate_sources=True)
        # 非重复纠错不需要已有来源；其 UPDATE 来源仍仅来自新候选。
        candidate = {**self.candidate, "intent": "correct", "target": {"fact_id": "old", "version": 1},
                     "object": {"type": "literal", "value": "茶"}}
        candidate.pop("fact_id")
        plan = self.plan(candidate, [missing], merge_duplicate_sources=True)
        self.assertEqual(plan["relation"], "CORRECTION")
        self.assertEqual(plan["decision"]["sources"], [self.new_source])

    def test_candidate_snapshot_and_result_do_not_share_mutable_source_or_object_data(self):
        candidate = {**deepcopy(self.candidate), "object": {"type": "literal", "value": {"nested": ["咖啡"]}}}
        old = self.fact(object=deepcopy(candidate["object"]))
        facts = [old]
        before_candidate, before_facts = deepcopy(candidate), deepcopy(facts)
        plan = self.plan(candidate, facts, merge_duplicate_sources=True)
        self.assertEqual(candidate, before_candidate)
        self.assertEqual(facts, before_facts)
        plan["decision"]["object"]["value"]["nested"].append("修改返回值")
        plan["decision"]["sources"][0]["message_id"] = "changed"
        plan["target"]["version"] = 99
        self.assertEqual(candidate, before_candidate)
        self.assertEqual(facts, before_facts)
        another = self.plan(candidate, facts, merge_duplicate_sources=True)
        candidate["sources"][0]["message_id"] = "changed-input"
        old["object"]["value"]["nested"].clear()
        self.assertEqual(another["decision"]["object"]["value"], {"nested": ["咖啡"]})
        self.assertEqual(another["decision"]["sources"], [self.old_source, self.new_source])

    def test_latest_version_and_candidate_identity_rules_still_apply_without_reviving_history(self):
        old = self.fact(sources=None)
        latest = self.fact(version=2)
        plan = self.plan(facts=[old, latest], merge_duplicate_sources=True)
        self.assert_merge(plan, version=3)
        for status in ("retracted", "superseded"):
            with self.subTest(status=status):
                plan = self.plan(facts=[old, {**latest, "status": status, "sources": None}],
                                 merge_duplicate_sources=True)
                self.assertEqual(plan["relation"], "SUPPLEMENT")
                self.assertEqual(plan["decision"]["action"], "ADD")
        plan = self.plan(facts=[self.fact(fact_id="candidate-new", sources=None)], merge_duplicate_sources=True)
        self.assertEqual(plan["relation"], "UNRESOLVED")
        self.assertIsNone(plan["decision"])


class DuplicateSourceExecutionTests(unittest.TestCase):
    def setUp(self):
        self.messages = MessageStore(":memory:")
        self.addCleanup(self.messages.close)
        self.facts = FactStore(self.messages)
        self.updater = MemoryUpdater(self.facts)
        self.sources_by_user = self.seed(self.messages, self.facts)
        self.sources = self.sources_by_user["u1"]
        self.first = self.facts.get_fact(user_id="u1", fact_id="f1", version=1)
        self.candidate = {
            "fact_id": "candidate-new", "subject_id": "user:u1", "predicate": "prefers_drink",
            "object": {"type": "literal", "value": "咖啡"}, "sources": [self.sources[1]],
            "scope": "session", "session_id": "current-chat",
            "valid_from": "2026-01-01T08:00:00+08:00", "valid_to": "2027-01-01T08:00:00+08:00",
        }

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
            facts.add_fact(user_id=user_id, fact_id="f1", subject_id=f"user:{user_id}",
                           predicate="prefers_drink", object={"type": "literal", "value": "咖啡"},
                           sources=[sources[0]], scope="session", session_id="current-chat",
                           valid_from="2026-01-01T00:00:00+00:00", valid_to="2027-01-01T00:00:00+00:00")
        return sources_by_user

    def plan(self, candidate=None, facts=None, user_id="u1"):
        candidate = self.candidate if candidate is None else candidate
        facts = [self.facts.get_latest_fact(user_id=user_id, fact_id="f1")] if facts is None else facts
        before_candidate, before_facts = deepcopy(candidate), deepcopy(facts)
        changes = self.messages.conn.total_changes
        result = judge_fact_relation(user_id=user_id, candidate=candidate, existing_facts=facts,
                                     merge_duplicate_sources=True)
        self.assertEqual(candidate, before_candidate)
        self.assertEqual(facts, before_facts)
        self.assertEqual(self.messages.conn.total_changes, changes)
        return result["decision"]

    def snapshot(self):
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

    def test_merge_executes_one_version_preserves_old_sources_and_retries_after_retraction(self):
        decision = self.plan()
        self.assertEqual(decision["new_version"], 2)
        merged = self.updater.apply(user_id="u1", decision=decision)["fact"]
        self.assertEqual(merged["sources"], self.sources[:2])
        self.assertEqual(merged["object"], self.first["object"])
        for field in ("subject_id", "predicate", "scope", "session_id", "valid_from", "valid_to"):
            self.assertEqual(merged[field], self.first[field])
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="f1", version=1),
                         {**self.first, "status": "superseded"})
        known = self.plan()
        self.assertEqual(known, {"action": "NOOP"})
        changes = self.messages.conn.total_changes
        self.assertEqual(self.updater.apply(user_id="u1", decision=known), {"action": "NOOP", "fact": None})
        for retry in ({"action": "UPDATE", "fact_id": "f1", "new_version": 2},
                      {**decision, "object": None, "sources": []}):
            self.assertEqual(self.updater.apply(user_id="u1", decision=retry)["fact"], merged)
        self.assertEqual(self.messages.conn.total_changes, changes)
        retracted = self.facts.retract_fact(user_id="u1", fact_id="f1", expected_version=2)
        changes = self.messages.conn.total_changes
        self.assertEqual(self.updater.apply(user_id="u1", decision=decision)["fact"], retracted)
        self.assertEqual(self.messages.conn.total_changes, changes)
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="f1", version=3))

    def test_fixed_merge_decision_reads_its_original_version_after_later_updates(self):
        decision = self.plan()
        self.updater.apply(user_id="u1", decision=decision)
        self.facts.update_fact(user_id="u1", fact_id="f1", new_version=3,
                               object={"type": "literal", "value": "茶"}, sources=[self.sources[2]])
        latest = self.facts.retract_fact(user_id="u1", fact_id="f1", expected_version=3)
        expected = self.facts.get_fact(user_id="u1", fact_id="f1", version=2)
        before = self.snapshot()
        result = self.updater.apply(user_id="u1", decision={**decision, "object": None, "sources": None})
        self.assertEqual(result["fact"], expected)
        self.assertEqual(expected["status"], "superseded")
        self.assertEqual(expected["sources"], self.sources[:2])
        self.assertEqual(self.facts.get_latest_fact(user_id="u1", fact_id="f1"), latest)
        self.assertEqual(self.snapshot(), before)
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="f1", version=4))

    def test_each_new_candidate_uses_fresh_latest_snapshot_and_accumulates_all_sources(self):
        first_decision = self.plan()
        second = self.updater.apply(user_id="u1", decision=first_decision)["fact"]
        self.assertEqual(second["sources"], self.sources[:2])
        next_candidate = {**self.candidate, "sources": [self.sources[2]]}
        next_decision = self.plan(next_candidate)
        self.assertEqual(next_decision["new_version"], 3)
        self.assertEqual(next_decision["sources"], self.sources)
        third = self.updater.apply(user_id="u1", decision=next_decision)["fact"]
        self.assertEqual(third["sources"], self.sources)
        self.assertEqual(self.facts.get_latest_fact(user_id="u1", fact_id="f1"), third)
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="f1", version=2)["sources"], self.sources[:2])
        self.assertEqual(self.facts.get_fact(user_id="u1", fact_id="f1", version=1)["sources"], self.sources[:1])
        self.assertEqual(self.plan(next_candidate), {"action": "NOOP"})
        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="f1", version=4))

    def test_store_validates_real_messages_for_both_old_and_new_planned_source_refs(self):
        other_only = {"user_id": "u2", "session_id": "private", "message_id": "m9"}
        self.messages.append_message(**other_only, occurred_at="2026-10-21T10:00:00+08:00",
                                     payload={"role": "user", "content": "另一用户独有来源"})
        absent = {"user_id": "u1", "session_id": "private", "message_id": "m9"}
        for candidate, facts in (
            ({**self.candidate, "sources": [absent]}, [self.first]),
            (self.candidate, [{**self.first, "sources": [absent]}]),
        ):
            decision = self.plan(candidate, facts)
            self.assertEqual(decision["action"], "UPDATE")
            before = self.snapshot()
            with self.subTest(candidate=candidate, facts=facts), self.assertRaises(ValueError):
                self.updater.apply(user_id="u1", decision=decision)
            self.assertEqual(self.snapshot(), before)
            self.assertEqual(self.facts.get_latest_fact(user_id="u1", fact_id="f1"), self.first)

    def test_same_fact_and_message_ids_are_merged_only_for_the_current_user(self):
        candidate = {**self.candidate, "subject_id": "user:u2", "sources": [self.sources_by_user["u2"][1]]}
        decision = self.plan(candidate, user_id="u2")
        merged = self.updater.apply(user_id="u2", decision=decision)["fact"]
        self.assertEqual(merged["user_id"], "u2")
        self.assertEqual(merged["sources"], self.sources_by_user["u2"][:2])
        self.assertEqual(self.facts.get_latest_fact(user_id="u1", fact_id="f1"), self.first)
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.plan({**self.candidate, "sources": [self.sources_by_user["u2"][1]]})
        self.assertEqual(self.snapshot(), before)

    def test_fail_or_ignore_at_status_fact_or_late_source_rolls_back_the_entire_merge(self):
        decision = self.plan({**self.candidate, "sources": self.sources[1:]})
        events = (
            "UPDATE OF status ON facts WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'f1' AND NEW.status = 'superseded'",
            "INSERT ON facts WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'f1' AND NEW.version = 2",
            "INSERT ON fact_sources WHEN NEW.user_id = 'u1' AND NEW.fact_id = 'f1' AND NEW.version = 2 AND NEW.source_message_id = 'm3'",
        )
        for mode in ("FAIL", "IGNORE"):
            for event in events:
                timing = "AFTER" if mode == "FAIL" else "BEFORE"
                abort = "RAISE(FAIL, 'merge write failed')" if mode == "FAIL" else "RAISE(IGNORE)"
                with self.subTest(mode=mode, event=event):
                    with self.messages.conn:
                        self.messages.conn.execute(f"""
                            CREATE TRIGGER fail_duplicate_merge {timing} {event}
                            BEGIN
                                UPDATE facts SET predicate = 'trigger-side-effect'
                                WHERE user_id = 'u2' AND fact_id = 'f1' AND version = 1;
                                SELECT {abort};
                            END
                        """)
                    before = self.snapshot()
                    try:
                        error = sqlite3.IntegrityError if mode == "FAIL" else ValueError
                        with self.assertRaises(error):
                            self.updater.apply(user_id="u1", decision=decision)
                        self.assertEqual(self.snapshot(), before)
                        self.assertIsNone(self.facts.get_fact(user_id="u1", fact_id="f1", version=2))
                    finally:
                        with self.messages.conn:
                            self.messages.conn.execute("DROP TRIGGER fail_duplicate_merge")

    def test_merged_sources_and_fixed_update_identity_survive_disk_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            messages = MessageStore(path)
            try:
                facts = FactStore(messages)
                self.seed(messages, facts)
                first = facts.get_fact(user_id="u1", fact_id="f1", version=1)
                decision = judge_fact_relation(user_id="u1", candidate=self.candidate, existing_facts=[first],
                                               merge_duplicate_sources=True)["decision"]
                result = MemoryUpdater(facts).apply(user_id="u1", decision=decision)
            finally:
                messages.close()
            messages = MessageStore(path)
            try:
                facts = FactStore(messages)
                updater = MemoryUpdater(facts)
                changes = messages.conn.total_changes
                self.assertEqual(facts.get_fact(user_id="u1", fact_id="f1", version=1),
                                 {**first, "status": "superseded"})
                self.assertEqual(facts.get_fact(user_id="u1", fact_id="f1", version=2), result["fact"])
                self.assertEqual(updater.apply(user_id="u1", decision={"action": "UPDATE", "fact_id": "f1", "new_version": 2}), result)
                self.assertEqual(messages.conn.total_changes, changes)
                self.assertIsNone(facts.get_fact(user_id="u1", fact_id="f1", version=3))
            finally:
                messages.close()


if __name__ == "__main__":
    unittest.main()
