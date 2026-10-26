import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


class MessageStore:
    def __init__(self, db_path: str | Path):
        self.conn = sqlite3.connect(
            db_path,
            autocommit=True,
        )
        self.conn.row_factory = sqlite3.Row

        try:
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA busy_timeout = 5000")
            self.conn.autocommit = False
            self._create_schema()
        except Exception:
            self.conn.close()
            raise

    @staticmethod
    def validate_ids(**fields: str) -> None:
        for field_name, value in fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"'{field_name}' 必须是非空字符串")

    def _create_schema(self) -> None:
        """创建消息表，并正确提交事务。"""
        cursor = self.conn.cursor()
        existing_tables = cursor.execute("""
            SELECT COUNT(*) AS count FROM sqlite_master
            WHERE type = 'table'
              AND name IN ('messages', 'message_sequences', 'memory_forget_tombstones')
        """).fetchone()["count"]
        if existing_tables == 3:
            self.conn.commit()
            cursor.close()
            return
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
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS message_sequences (
                           user_id TEXT NOT NULL,
                           session_id TEXT NOT NULL,
                           next_seq INTEGER NOT NULL CHECK (next_seq >= 1),
                           PRIMARY KEY (user_id, session_id)
                       )
                       ''')
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS memory_forget_tombstones (
                           user_id TEXT NOT NULL PRIMARY KEY,
                           forgotten_at TEXT NOT NULL,
                           reason TEXT NOT NULL,
                           deleted_rows_json TEXT NOT NULL
                       )
                       ''')
        cursor.execute('''
                       INSERT INTO message_sequences (user_id, session_id, next_seq)
                       SELECT user_id, session_id, MAX(seq) + 1
                       FROM messages
                       GROUP BY user_id, session_id
                       ON CONFLICT (user_id, session_id) DO UPDATE SET
                           next_seq = MAX(message_sequences.next_seq, excluded.next_seq)
                       ''')
        self.conn.commit()
        cursor.close()

    def is_forgotten(self, *, user_id: str) -> bool:
        self.validate_ids(user_id=user_id)
        return self.conn.execute(
            "SELECT 1 FROM memory_forget_tombstones WHERE user_id = ?",
            (user_id,),
        ).fetchone() is not None

    def assert_user_active(self, *, user_id: str) -> None:
        if self.is_forgotten(user_id=user_id):
            raise ValueError("该用户已执行物理遗忘，不能重新写入派生数据")

    def get_message(self, *, user_id: str, session_id: str, message_id: str) -> dict | None:
        self.validate_ids(user_id=user_id, session_id=session_id, message_id=message_id)
        self.assert_user_active(user_id=user_id)
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
        self.validate_ids(user_id=user_id, session_id=session_id, message_id=message_id)
        # 判断当前消息是否已写入数据库；若已存在则幂等返回，忽略新传入的 payload/occurred_at
        existing = self.get_message(user_id=user_id, session_id=session_id, message_id=message_id)
        if existing:
            return existing
        else:
            # autocommit=False 的 SELECT 会保留共享锁；写入前结束它，让其他连接可等待串行写锁。
            self.conn.commit()
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

            try:
                with self.conn:
                    cursor = self.conn.cursor()
                    try:
                        row = cursor.execute('''
                            INSERT INTO message_sequences (user_id, session_id, next_seq)
                            VALUES (?, ?, 2)
                            ON CONFLICT (user_id, session_id) DO UPDATE SET
                                next_seq = message_sequences.next_seq + 1
                            RETURNING next_seq - 1 AS seq
                        ''', (user_id, session_id)).fetchone()
                        # 分配序列已取得写锁，防止首次检查后另一连接完成遗忘。
                        self.assert_user_active(user_id=user_id)
                        cursor.execute('''
                                       INSERT INTO messages (message_id,
                                                             user_id,
                                                             session_id,
                                                             seq,
                                                             occurred_at,
                                                             recorded_at,
                                                             payload_json)
                                       VALUES (?, ?, ?, ?, ?, ?, ?)
                                       ''', (message_id, user_id, session_id, row["seq"], normalized_time,
                                             recorded_at, payload_json))
                    finally:
                        cursor.close()
            except sqlite3.IntegrityError:
                # 两个写入者同时重试同一身份时，输家读取已经提交的首次记录。
                existing = self.get_message(
                    user_id=user_id, session_id=session_id, message_id=message_id,
                )
                if existing is not None:
                    return existing
                raise
            return self.get_message(user_id=user_id, session_id=session_id, message_id=message_id)

    # 返回会话内在队列号seq之后的消息列表
    def list_messages(
            self,
            *,
            user_id: str,
            session_id: str,
            after_seq: int = 0,
    ) -> list[dict]:
        self.validate_ids(user_id=user_id, session_id=session_id)
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
        self.validate_ids(user_id=user_id)
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
