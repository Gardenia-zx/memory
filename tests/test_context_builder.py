"""验证系统指令、事实和近期消息的组装及完整输入预算。"""

import copy
import json
import unittest
from unittest.mock import Mock

from src.context_builder import ContextBuilder
from src.context_queue import ContextQueue
from src.error import ContextBudgetExceeded
from src.fact_store import FactStore
from src.message_store import MessageStore


class ContextBuilderTests(unittest.TestCase):
    def setUp(self):
        self.store = MessageStore(":memory:")
        self.addCleanup(self.store.close)
        self.facts = FactStore(self.store)
        self.queue = ContextQueue(self.store, user_id="u1", session_id="s1")
        self.builder = ContextBuilder(self.queue, self.facts)
        self.system_prompt = "请用中文回答。"

    def _message(self, message_id, payload, *, user_id="u1", session_id="s1"):
        source = {
            "user_id": user_id,
            "session_id": session_id,
            "message_id": message_id,
        }
        self.store.append_message(
            **source,
            occurred_at="2000-01-01T09:00:00+00:00",
            payload=payload,
        )
        return source

    def _fact(self, fact_id, source, **changes):
        request = {
            "user_id": source["user_id"],
            "fact_id": fact_id,
            "subject_id": f"user:{source['user_id']}",
            "predicate": "likes",
            "object": {"type": "literal", "value": "咖啡"},
            "sources": [source],
        }
        request.update(changes)
        return self.facts.add_fact(**request)

    def _build(self, **changes):
        request = {
            "system_prompt": self.system_prompt,
            "max_tokens": 1000,
            "token_counter": lambda messages: 0,
        }
        request.update(changes)
        return self.builder.build_context(**request)

    def test_empty_facts_and_history_return_only_system_message(self):
        counter = Mock(return_value=0)

        context = self._build(token_counter=counter)

        self.assertEqual(context, [{"role": "system", "content": self.system_prompt}])
        counter.assert_called_once_with(context)
        self.assertEqual(self.queue.last_seq, 0)

    def test_memory_is_json_data_filtered_by_bound_user_session_and_time(self):
        source = self._message(
            "m1", {"role": "user", "content": "喜欢咖啡"}, session_id="history",
        )
        other_source = self._message(
            "m2", {"role": "user", "content": "其他用户"}, user_id="u2",
        )
        validity = {
            "valid_from": "2000-01-01T10:00:00+00:00",
            "valid_to": "2000-01-01T11:00:00+00:00",
        }
        global_fact = self._fact("a-global", source, **validity)
        session_fact = self._fact(
            "b-session", source, scope="session", session_id="s1", **validity,
        )
        self._fact("c-other-session", source, scope="session", session_id="s2", **validity)
        self._fact("d-other-user", other_source, **validity)
        self._fact("e-expired", source, valid_to="2000-01-01T10:30:00+00:00")
        self._fact("f-future", source, valid_from="2000-01-01T10:30:01+00:00")
        before_changes = self.store.conn.total_changes

        context = self._build(as_of="2000-01-01T10:30:00+00:00")

        self.assertEqual(len(context), 2)
        self.assertEqual(context[0], {"role": "system", "content": self.system_prompt})
        self.assertEqual(context[1]["role"], "user")
        self.assertIsInstance(context[1]["content"], str)
        self.assertIn("咖啡", context[1]["content"])
        self.assertEqual(
            json.loads(context[1]["content"]),
            {"type": "memory_context", "facts": [global_fact, session_fact]},
        )
        self.assertEqual(self.queue.last_seq, 0)
        self.assertEqual(self.queue.to_messages(), [])
        self.assertEqual(self.store.conn.total_changes, before_changes)

    def test_recent_messages_remain_flat_and_preserve_tool_payloads(self):
        payloads = [
            {"role": "user", "content": "查一下早餐饮品"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "name": "lookup", "arguments": '{"q":'}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "参数错误"},
            {"role": "assistant", "content": "需要重新查询"},
        ]
        expected = copy.deepcopy(payloads)
        for index, payload in enumerate(payloads):
            self._message(f"m{index}", payload)
        self.queue.refresh()
        watermark = self.queue.last_seq

        context = self._build()

        self.assertEqual(context, [{"role": "system", "content": self.system_prompt}] + expected)
        context[2]["tool_calls"][0]["arguments"] = "修改返回参数"
        context[3]["content"] = "修改返回结果"
        self.assertEqual(self.queue.to_messages(), expected)
        self.assertEqual(self.queue.last_seq, watermark)
        self.assertEqual(
            [row["payload"] for row in self.store.list_messages(user_id="u1", session_id="s1")],
            expected,
        )

    def test_build_does_not_refresh_queue_implicitly(self):
        payload = {"role": "user", "content": "尚未同步到队列"}
        self._message("m1", payload)

        self.assertEqual(self._build(), [{"role": "system", "content": self.system_prompt}])
        self.assertEqual(self.queue.last_seq, 0)
        self.assertEqual(self.queue.refresh(), 1)
        self.assertEqual(
            self._build(), [{"role": "system", "content": self.system_prompt}, payload],
        )

    def test_budget_counts_complete_context_once_and_preserves_state_on_overflow(self):
        source = self._message(
            "source", {"role": "user", "content": "喜欢咖啡"}, session_id="history",
        )
        self._fact("f1", source)
        self._message("question", {"role": "user", "content": "早餐喝什么？"})
        self.queue.refresh()
        original_messages = self.queue.to_messages()
        watermark = self.queue.last_seq
        before_changes = self.store.conn.total_changes
        counter = Mock(return_value=17)

        context = self._build(max_tokens=17, token_counter=counter)

        counter.assert_called_once_with(context)
        self.assertEqual(len(context), 3)
        self.assertEqual(context[0]["role"], "system")
        self.assertEqual(json.loads(context[1]["content"])["type"], "memory_context")
        self.assertEqual(context[2:], original_messages)
        counter.reset_mock()
        with self.assertRaises(ContextBudgetExceeded):
            self._build(max_tokens=16, token_counter=counter)
        counter.assert_called_once()
        self.assertEqual(counter.call_args.args[0], context)
        self.assertEqual(self.queue.to_messages(), original_messages)
        self.assertEqual(self.queue.last_seq, watermark)
        self.assertEqual(self.store.conn.total_changes, before_changes)
        self.assertEqual(len(self._build(max_tokens=0, token_counter=lambda messages: 0)), 3)

    def test_invalid_parameters_raise_value_error(self):
        for field, values in (
            ("system_prompt", (None, 123, True, [])),
            ("max_tokens", (-1, True, False, 1.5, "10", None)),
            ("token_counter", (None, 123, "counter", False)),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self._build(**{field: value})

    def test_invalid_counter_results_raise_value_error(self):
        for value in (-1, True, False, 1.5, "10", None):
            with self.subTest(value=value):
                counter = Mock(return_value=value)
                with self.assertRaises(ValueError):
                    self._build(token_counter=counter)
                counter.assert_called_once()


if __name__ == "__main__":
    unittest.main()
