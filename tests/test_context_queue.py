"""第二关第一小步：上下文队列的显式增量同步与消息导出。"""

import copy
import tempfile
import unittest
from pathlib import Path

from src.context_queue import ContextQueue
from src.message_store import MessageStore


class ContextQueueTests(unittest.TestCase):
    def setUp(self):
        project_root = Path(__file__).resolve().parents[1]
        temp_dir = tempfile.TemporaryDirectory(
            prefix=".context-queue-test-", dir=project_root,
        )
        self.assertTrue(Path(temp_dir.name).resolve().is_relative_to(project_root))
        self.addCleanup(temp_dir.cleanup)
        self.store = MessageStore(Path(temp_dir.name) / "messages.sqlite")
        self.addCleanup(self.store.close)
        self.scope = {"user_id": "u1", "session_id": "s1"}
        self.request = {
            **self.scope,
            "message_id": "m1",
            "occurred_at": "2026-10-20T10:00:00+08:00",
            "payload": {"role": "user", "content": "你好"},
        }
        self.queue = ContextQueue(self.store, **self.scope)

    def _seed_conversation(self):
        payloads = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么需要帮忙？"},
        ]
        for index, payload in enumerate(payloads, start=1):
            self.store.append_message(
                **{
                    **self.request,
                    "message_id": f"m{index}",
                    "payload": payload,
                }
            )
        return payloads

    def test_empty_store_refresh_keeps_queue_empty(self):
        self.assertEqual(self.queue.last_seq, 0)
        self.assertEqual(self.queue.to_messages(), [])

        for _ in range(2):
            self.assertEqual(self.queue.refresh(), 0)
            self.assertEqual(self.queue.to_messages(), [])
            self.assertEqual(self.queue.last_seq, 0)

    def test_constructor_and_export_do_not_load_history_implicitly(self):
        self._seed_conversation()
        queue = ContextQueue(self.store, **self.scope)

        self.assertEqual(queue.last_seq, 0)
        self.assertEqual(queue.to_messages(), [])
        self.assertEqual(queue.last_seq, 0)
        self.assertEqual(len(self.store.list_messages(**self.scope)), 2)

    def test_refresh_loads_only_bound_scope_in_sequence_order(self):
        first_payload = {"role": "user", "content": "本会话第一条消息"}
        second_payload = {"role": "assistant", "content": "本会话第二条消息"}
        requests = [
            {**self.request, "payload": first_payload},
            {
                **self.request,
                "user_id": "u2",
                "message_id": "foreign-user",
                "payload": {"role": "user", "content": "其他用户"},
            },
            {
                **self.request,
                "session_id": "s2",
                "message_id": "foreign-session",
                "payload": {"role": "user", "content": "其他会话"},
            },
            {
                **self.request,
                "message_id": "m2",
                # 晚写入的消息发生时间更早，导出仍按 seq 排序。
                "occurred_at": "2026-10-20T09:00:00+08:00",
                "payload": second_payload,
            },
        ]
        for request in requests:
            self.store.append_message(**request)

        history = self.store.list_messages(**self.scope)
        self.assertEqual(self.queue.refresh(), 2)
        self.assertEqual(self.queue.last_seq, 2)
        self.assertEqual(
            self.queue.to_messages(), [first_payload, second_payload]
        )
        # 队列操作不改变完整历史。
        self.assertEqual(self.store.list_messages(**self.scope), history)

    def test_repeated_refresh_does_not_duplicate_messages(self):
        payloads = self._seed_conversation()
        self.assertEqual(self.queue.refresh(), 2)
        self.assertEqual(self.queue.last_seq, 2)

        for _ in range(2):
            self.assertEqual(self.queue.refresh(), 0)
            self.assertEqual(self.queue.to_messages(), payloads)
            self.assertEqual(self.queue.last_seq, 2)

    def test_new_messages_are_loaded_only_after_explicit_refresh(self):
        payloads = self._seed_conversation()
        self.assertEqual(self.queue.refresh(), 2)
        self.assertEqual(self.queue.last_seq, 2)

        new_payload = {"role": "user", "content": "这是一条新增消息"}
        self.store.append_message(
            **{
                **self.request,
                "message_id": "m3",
                "payload": new_payload,
            }
        )
        self.assertEqual(self.queue.to_messages(), payloads)
        self.assertEqual(self.queue.last_seq, 2)

        self.assertEqual(self.queue.refresh(), 1)
        self.assertEqual(self.queue.to_messages(), [*payloads, new_payload])
        self.assertEqual(self.queue.last_seq, 3)

        self.assertEqual(self.queue.refresh(), 0)
        self.assertEqual(self.queue.last_seq, 3)
        self.assertEqual(len(self.store.list_messages(**self.scope)), 3)

    def test_export_preserves_tool_payloads_and_returns_independent_copies(self):
        payloads = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "lookup_city",
                        "arguments": '{"city":',
                    }
                ],
                "metadata": {"tags": ["原始"]},
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"ok": false, "error": "参数格式错误"}',
            },
        ]
        expected = copy.deepcopy(payloads)
        for index, payload in enumerate(payloads, start=1):
            self.store.append_message(
                **{
                    **self.request,
                    "message_id": f"m{index}",
                    "payload": payload,
                }
            )

        self.assertEqual(self.queue.refresh(), 2)
        exported = self.queue.to_messages()
        self.assertEqual(exported, expected)

        exported[0]["tool_calls"][0]["arguments"] = "修改工具参数"
        exported[0]["metadata"]["tags"].append("修改嵌套数据")
        exported[1]["content"] = "修改工具结果"
        exported.pop()
        exported.append({"role": "user", "content": "修改导出列表"})

        self.assertEqual(self.queue.to_messages(), expected)
        self.assertEqual(self.queue.last_seq, 2)
        self.assertEqual(
            [record["payload"] for record in self.store.list_messages(**self.scope)],
            expected,
        )
