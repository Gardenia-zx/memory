import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.message_store import MessageStore


class MessageStoreInitializationTests(unittest.TestCase):
    def test_create_and_reopen(self):
        project_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory(
            dir=project_root,
        ) as temp_dir:
            db_path = Path(temp_dir) / "messages.sqlite"

            store = MessageStore(db_path)
            store.close()

            # 使用独立连接，验证建表结果确实已保存。
            conn = sqlite3.connect(db_path)
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM messages"
                ).fetchone()
                self.assertEqual(row[0], 0)
            finally:
                conn.close()

            # 同一个数据库重复初始化应当成功。
            reopened = MessageStore(db_path)
            reopened.close()

    def test_append_and_get(self):
        project_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory(
                dir=project_root,
        ) as temp_dir:
            db_path = Path(temp_dir) / "messages.sqlite"
            payload = {
                "role": "user",
                "content": "以后请用中文回答。",
            }

            store = MessageStore(db_path)
            try:
                saved = store.append_message(
                    user_id="u1",
                    session_id="s1",
                    message_id="m1",
                    occurred_at="2026-10-20T09:00:00+08:00",
                    payload=payload,
                )

                self.assertEqual(saved["seq"], 1)
                self.assertEqual(saved["payload"], payload)
                self.assertEqual(
                    saved["occurred_at"],
                    "2026-10-20T01:00:00+00:00",
                )
            finally:
                store.close()

            reopened = MessageStore(db_path)
            try:
                restored = reopened.get_message(
                    user_id="u1",
                    session_id="s1",
                    message_id="m1",
                )
                self.assertEqual(restored, saved)
            finally:
                reopened.close()

    def test_sequence_and_scoped_listing(self):
        project_root = Path(__file__).resolve().parents[1]

        cases = [
            ("u1", "s1", "m1", "2026-10-20T10:00:00+08:00"),
            ("u1", "s2", "m1", "2026-10-20T11:00:00+08:00"),
            ("u2", "s1", "m1", "2026-10-20T12:00:00+08:00"),
            ("u1", "s1", "m2", "2026-10-20T09:00:00+08:00"),
        ]

        with tempfile.TemporaryDirectory(
                dir=project_root,
        ) as temp_dir:
            store = MessageStore(
                Path(temp_dir) / "messages.sqlite"
            )

            try:
                saved = []

                for uid, sid, mid, occurred_at in cases:
                    saved.append(
                        store.append_message(
                            user_id=uid,
                            session_id=sid,
                            message_id=mid,
                            occurred_at=occurred_at,
                            payload={
                                "role": "user",
                                "content": f"{uid}/{sid}/{mid}",
                            },
                        )
                    )

                self.assertEqual(
                    [message["seq"] for message in saved],
                    [1, 1, 1, 2],
                )

                main = [saved[0], saved[3]]

                self.assertEqual(
                    store.list_messages(
                        user_id="u1", session_id="s1",
                    ),
                    main,
                )
                self.assertEqual(
                    store.list_messages(
                        user_id="u1", session_id="s1",
                        after_seq=1,
                    ),
                    [saved[3]],
                )
                self.assertEqual(
                    store.list_messages(
                        user_id="u1", session_id="s1",
                        after_seq=2,
                    ),
                    [],
                )
                self.assertEqual(
                    store.list_messages(
                        user_id="u1", session_id="s2",
                    ),
                    [saved[1]],
                )
                self.assertEqual(
                    store.list_messages(
                        user_id="u2", session_id="s1",
                    ),
                    [saved[2]],
                )
                self.assertEqual(
                    store.list_messages(
                        user_id="missing", session_id="s1",
                    ),
                    [],
                )
            finally:
                store.close()