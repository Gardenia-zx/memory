import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


class MessageStore:
    def __init__(self, db_path: str | Path):
        self.conn = sqlite3.connect(
            db_path,
            autocommit=False,
        )
        self.conn.row_factory = sqlite3.Row

        try:
            self._create_schema()
        except Exception:
            self.conn.close()
            raise

    @staticmethod
    def _validate_ids(**fields: str) -> None:
        for field_name, value in fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"'{field_name}' 必须是非空字符串")

    def _create_schema(self) -> None:
        """创建消息表，并正确提交事务。"""
        cursor = self.conn.cursor()
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS messages
                       (
                           message_id
                           TEXT
                           NOT
                           NULL,
                           user_id
                           TEXT
                           NOT
                           NULL,
                           session_id
                           TEXT
                           NOT
                           NULL,
                           seq
                           INTEGER
                           NOT
                           NULL
                           CHECK
                       (
                           seq >
                           0
                       ),
                           occurred_at TEXT NOT NULL,
                           recorded_at TEXT NOT NULL,
                           payload_json TEXT NOT NULL,
                           PRIMARY KEY
                       (
                           user_id,
                           session_id,
                           message_id
                       ),
                           UNIQUE
                       (
                           user_id,
                           session_id,
                           seq
                       )
                           )
                       ''')
        self.conn.commit()
        cursor.close()

    def get_message(self, *, user_id: str, session_id: str, message_id: str) -> dict | None:
        self._validate_ids(user_id=user_id, session_id=session_id, message_id=message_id)
        cursor = self.conn.cursor()
        rows = cursor.execute(
            "SELECT * FROM messages WHERE user_id = ? AND session_id = ? AND message_id = ?",
            (user_id, session_id, message_id),
        ).fetchone()
        cursor.close()
        if rows:
            msg = dict(rows)
            msg["payload_json"] = json.loads(msg["payload_json"])
            msg["payload"] = msg.pop("payload_json")
            return msg
        return None

    def append_message(
            self,
            *,
            user_id: str,
            session_id: str,
            message_id: str,
            occurred_at: str,
            payload: dict,
    ) -> dict | None:
        self._validate_ids(user_id=user_id, session_id=session_id, message_id=message_id)
        # 判断当前消息是否已写入数据库；若已存在则幂等返回，忽略新传入的 payload/occurred_at
        existing = self.get_message(user_id=user_id, session_id=session_id, message_id=message_id)
        if existing:
            return existing
        else:
            if not isinstance(payload, dict):
                raise ValueError("payload 必须是字典")

            if payload.get("role") not in ("user", "assistant", "tool"):
                raise ValueError("payload.role 不合法")

            if not isinstance(occurred_at, str):
                raise ValueError("occurred_at 必须是时间字符串")
            # 时间处理，消息转为json数据
            occurred = datetime.fromisoformat(occurred_at)
            if occurred.utcoffset() is None:
                raise ValueError("occurred_at 必须包含时区")
            normalized_time = occurred.astimezone(timezone.utc).isoformat()
            recorded_at = datetime.now(timezone.utc).isoformat()

            payload_json = json.dumps(payload, ensure_ascii=False, allow_nan=False,sort_keys=True)

            with self.conn:
                cursor = self.conn.cursor()
                row = cursor.execute(
                    "SELECT COALESCE(MAX(seq),0) + 1 AS next_seq FROM messages WHERE user_id = ? AND session_id = ?",
                    (user_id, session_id),
                ).fetchone()
                cursor.execute('''
                               INSERT INTO messages (message_id,
                                                     user_id,
                                                     session_id,
                                                     seq,
                                                     occurred_at,
                                                     recorded_at,
                                                     payload_json)
                               VALUES (?, ?, ?, ?, ?, ?, ?)
                               ''', (message_id, user_id, session_id, row["next_seq"], normalized_time, recorded_at,
                                     payload_json))
                cursor.close()
            return self.get_message(user_id=user_id, session_id=session_id, message_id=message_id)

    # 返回会话内在队列号seq之后的消息列表
    def list_messages(
            self,
            *,
            user_id: str,
            session_id: str,
            after_seq: int = 0,
    ) -> list[dict]:
        self._validate_ids(user_id=user_id, session_id=session_id)
        if type(after_seq) is not int or after_seq < 0:
            raise ValueError("after_seq 必须是非负整数")
        cursor = self.conn.cursor()
        rows = cursor.execute('''
                              SELECT *
                              FROM messages
                              WHERE user_id = ?
                                AND session_id = ?
                                AND seq > ?
                              ORDER BY seq ASC
                              ''', (user_id, session_id, after_seq)).fetchall()
        for i in range(len(rows)):
            rows[i] = dict(rows[i])
            rows[i]["payload_json"] = json.loads(rows[i]["payload_json"])
            rows[i]["payload"] = rows[i].pop("payload_json")
        cursor.close()
        return rows
    def close(self) -> None:
        self.conn.close()

    def list_user_messages(
            self,
            *,
            user_id: str,
    )->list[dict]:
        self._validate_ids(user_id=user_id)
        cursor = self.conn.cursor()
        rows = cursor.execute('''
        SELECT *
        FROM messages
        WHERE user_id = ?
        ORDER BY session_id ASC, seq ASC
        ''', (user_id,)).fetchall()
        cursor.close()
        for i in range(len(rows)):
            rows[i] = dict(rows[i])
            rows[i]["payload_json"] = json.loads(rows[i]["payload_json"])
            rows[i]["payload"] = rows[i].pop("payload_json")
        return rows
