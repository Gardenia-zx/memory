"""第一关的回归测试：仅验证已约定的单写入者消息存储行为。"""

import copy
import tempfile
import unittest
from pathlib import Path

from src.message_store import MessageStore


class MessageStoreRegressionTests(unittest.TestCase):
    def setUp(self):
        project_root = Path(__file__).resolve().parents[1]
        temp_dir = tempfile.TemporaryDirectory(
            prefix=".message-store-test-", dir=project_root,
        )
        self.assertTrue(Path(temp_dir.name).resolve().is_relative_to(project_root))
        self.addCleanup(temp_dir.cleanup)
        self.db_path = Path(temp_dir.name) / "messages.sqlite"
        self.store = MessageStore(self.db_path)
        # 后注册的清理先执行；lambda 在重开数据库后关闭新的连接。
        self.addCleanup(lambda: self.store.close())
        self.scope = {"user_id": "u1", "session_id": "s1"}
        self.request = {
            **self.scope,
            "message_id": "m1",
            "occurred_at": "2026-10-20T10:00:00+08:00",
            "payload": {"role": "user", "content": "以后请用中文回答。"},
        }

    def _assert_no_messages(self):
        # 检查全表，避免漏掉意外写入非法用户或会话下的记录。
        count = self.store.conn.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_retry_returns_original_without_consuming_sequence(self):
        first = self.store.append_message(**self.request)

        for _ in range(2):
            repeated = self.store.append_message(**self.request)
            self.assertEqual(repeated, first)

        self.assertEqual(self.store.list_messages(**self.scope), [first])
        following = self.store.append_message(
            **{**self.request, "message_id": "m2"}
        )
        self.assertEqual(following["seq"], 2)
        # 相同正文使用新 ID，仍然应当保存为另一条消息。
        self.assertEqual(
            self.store.list_messages(**self.scope), [first, following]
        )

    def test_existing_id_returns_original_without_comparing_new_content(self):
        first = self.store.append_message(**self.request)
        changes = [
            {"payload": {"role": "user", "content": "不同的正文"}},
            {"occurred_at": "2026-10-20T11:00:00+08:00"},
            {"payload": None, "occurred_at": "不是有效时间"},
        ]

        for change in changes:
            with self.subTest(change=change):
                repeated = self.store.append_message(
                    **{**self.request, **change}
                )
                self.assertEqual(repeated, first)
                self.assertEqual(
                    self.store.list_messages(**self.scope), [first]
                )

        following = self.store.append_message(
            **{**self.request, "message_id": "m2"}
        )
        self.assertEqual(following["seq"], 2)

    def test_retry_remains_idempotent_after_reopen(self):
        first = self.store.append_message(**self.request)
        self.store.close()
        self.store = MessageStore(self.db_path)

        repeated = self.store.append_message(**self.request)
        self.assertEqual(repeated, first)
        self.assertEqual(self.store.list_messages(**self.scope), [first])
        following = self.store.append_message(
            **{**self.request, "message_id": "m2"}
        )
        self.assertEqual(following["seq"], 2)

    def test_append_rejects_invalid_ids_without_writing(self):
        for field in ("user_id", "session_id", "message_id"):
            for value in (None, "", " \t\n", 123, True):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.store.append_message(
                            **{**self.request, field: value}
                        )
                    self._assert_no_messages()

        self.assertEqual(self.store.append_message(**self.request)["seq"], 1)

    def test_get_rejects_invalid_ids(self):
        identity = {**self.scope, "message_id": "m1"}
        for field in identity:
            for value in (None, "", " \t\n", 123, True):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.store.get_message(**{**identity, field: value})

        self._assert_no_messages()

    def test_list_rejects_invalid_scope_ids(self):
        for field in self.scope:
            for value in (None, "", " \t\n", 123, True):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.store.list_messages(**{**self.scope, field: value})

        self._assert_no_messages()

    def test_list_rejects_invalid_after_seq(self):
        for value in (-1, 1.5, "1", True, False, None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.store.list_messages(**self.scope, after_seq=value)

        self._assert_no_messages()

    def test_append_rejects_invalid_payload_shapes_and_roles(self):
        invalid_payloads = [
            None,
            [],
            "消息正文",
            {},
            {"content": "缺少角色"},
            {"role": "system"},
            {"role": None},
            {"role": []},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    self.store.append_message(
                        **{**self.request, "payload": payload}
                    )
                self._assert_no_messages()

        self.assertEqual(self.store.append_message(**self.request)["seq"], 1)

    def test_append_rejects_invalid_timestamps_without_writing(self):
        invalid_times = [
            None,
            123,
            "",
            "不是有效时间",
            "2026-10-20T10:00:00",
            "2026-02-30T10:00:00+00:00",
        ]
        for occurred_at in invalid_times:
            with self.subTest(occurred_at=occurred_at):
                with self.assertRaises(ValueError):
                    self.store.append_message(
                        **{**self.request, "occurred_at": occurred_at}
                    )
                self._assert_no_messages()

        self.assertEqual(self.store.append_message(**self.request)["seq"], 1)

    def test_unserializable_payload_does_not_write(self):
        invalid_values = [
            ("set", {1, 2}),
            ("bytes", b"text"),
            ("nan", float("nan")),
            ("infinity", float("inf")),
            ("negative_infinity", float("-inf")),
        ]
        for label, value in invalid_values:
            with self.subTest(value=label):
                payload = {"role": "user", "content": value}
                with self.assertRaises((TypeError, ValueError)):
                    self.store.append_message(
                        **{**self.request, "payload": payload}
                    )
                self._assert_no_messages()

        self.assertEqual(self.store.append_message(**self.request)["seq"], 1)

    def test_tool_messages_round_trip_verbatim(self):
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
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": (
                    '{"ok": false, "error": '
                    '{"type": "invalid_json", "message": "参数未闭合"}}'
                ),
            },
        ]
        for index, payload in enumerate(payloads, start=1):
            saved = self.store.append_message(
                **{
                    **self.request,
                    "message_id": f"m{index}",
                    "payload": payload,
                }
            )
            self.assertEqual(saved["payload"], payload)

        messages = self.store.list_messages(**self.scope)
        self.assertEqual([message["payload"] for message in messages], payloads)
        self.assertEqual([message["seq"] for message in messages], [1, 2])

    def test_payload_mutations_do_not_change_persisted_data(self):
        payload = {
            "role": "user",
            "content": "包含中文和嵌套数据",
            "metadata": {"tags": ["原始"], "options": {"enabled": True}},
        }
        expected = copy.deepcopy(payload)
        saved = self.store.append_message(
            **{**self.request, "payload": payload}
        )
        self.assertEqual(payload, expected)

        payload["metadata"]["tags"].append("修改入参")
        saved["payload"]["metadata"]["options"]["enabled"] = False
        fetched = self.store.get_message(**self.scope, message_id="m1")
        self.assertEqual(fetched["payload"], expected)

        fetched["payload"]["metadata"]["tags"].append("修改查询结果")
        listed = self.store.list_messages(**self.scope)
        self.assertEqual(listed[0]["payload"], expected)

        listed[0]["payload"]["content"] = "修改列表结果"
        self.assertEqual(
            self.store.get_message(**self.scope, message_id="m1")["payload"],
            expected,
        )

    def test_missing_messages_and_scopes_return_empty_results(self):
        first = self.store.append_message(**self.request)
        identity = {**self.scope, "message_id": "m1"}
        for change in (
            {"message_id": "missing"},
            {"user_id": "missing"},
            {"session_id": "missing"},
        ):
            with self.subTest(change=change):
                self.assertIsNone(self.store.get_message(**{**identity, **change}))

        self.assertEqual(
            self.store.list_messages(user_id="missing", session_id="s1"), []
        )
        self.assertEqual(
            self.store.list_messages(user_id="u1", session_id="missing"), []
        )
        self.assertEqual(
            self.store.list_messages(**self.scope, after_seq=100), []
        )
        self.assertEqual(self.store.list_messages(**self.scope), [first])

    def test_nonempty_ids_are_preserved_without_trimming(self):
        identity = {
            "user_id": " 用户甲 ",
            "session_id": " 会话甲 ",
            "message_id": " 消息甲 ",
        }
        saved = self.store.append_message(**{**self.request, **identity})
        for field, value in identity.items():
            self.assertEqual(saved[field], value)

        self.assertEqual(self.store.get_message(**identity), saved)
        self.assertIsNone(
            self.store.get_message(
                **{field: value.strip() for field, value in identity.items()}
            )
        )
