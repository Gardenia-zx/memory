"""保存已生成的会话摘要；覆盖范围指向永久保留的原始消息。"""

from datetime import datetime, timezone
import json

from src.message_store import MessageStore


class SummaryStore:
    def __init__(self, message_store: MessageStore):
        self.message_store = message_store
        self.conn = message_store.conn
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    summary_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    start_seq INTEGER NOT NULL CHECK (start_seq >= 1),
                    end_seq INTEGER NOT NULL CHECK (end_seq >= start_seq),
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, session_id, summary_id),
                    FOREIGN KEY (user_id, session_id, start_seq)
                        REFERENCES messages (user_id, session_id, seq),
                    FOREIGN KEY (user_id, session_id, end_seq)
                        REFERENCES messages (user_id, session_id, seq)
                )
            """)

    def get_summary(self, *, user_id: str, session_id: str, summary_id: str) -> dict | None:
        self.message_store.validate_ids(user_id=user_id, session_id=session_id, summary_id=summary_id)
        row = self.conn.execute(
            "SELECT * FROM summaries WHERE user_id = ? AND session_id = ? AND summary_id = ?",
            (user_id, session_id, summary_id),
        ).fetchone()
        return None if row is None else dict(row)

    def save_summary(self, *, user_id: str, session_id: str, summary_id: str,
                     content: str = None, start_seq: int = None, end_seq: int = None) -> dict:
        """同一用户/会话/摘要 ID 首次写入；重试忽略正文及覆盖范围。"""
        self.message_store.validate_ids(user_id=user_id, session_id=session_id, summary_id=summary_id)
        with self.conn:
            existing = self.get_summary(user_id=user_id, session_id=session_id, summary_id=summary_id)
            if existing is not None:
                return existing
            self.message_store.assert_user_active(user_id=user_id)
            if not isinstance(content, str) or not content.strip():
                raise ValueError("content 必须是非空摘要文本")
            if (type(start_seq) is not int or type(end_seq) is not int
                    or start_seq < 1 or end_seq < start_seq):
                raise ValueError("摘要覆盖范围必须为正整数闭区间")
            count = self.conn.execute(
                """SELECT COUNT(*) FROM messages WHERE user_id = ? AND session_id = ?
                   AND seq BETWEEN ? AND ?""", (user_id, session_id, start_seq, end_seq),
            ).fetchone()[0]
            if count != end_seq - start_seq + 1:
                raise ValueError("摘要覆盖范围必须包含连续且真实存在的同会话消息")
            boundary = self.conn.execute("""
                SELECT seq, payload_json FROM messages
                WHERE user_id = ? AND session_id = ? AND seq IN (?, ?)
                ORDER BY seq
            """, (user_id, session_id, end_seq, end_seq + 1)).fetchall()
            if len(boundary) == 2:
                end_payload = json.loads(boundary[0]["payload_json"])
                next_payload = json.loads(boundary[1]["payload_json"])
                if (next_payload.get("role") == "tool"
                        and (end_payload.get("role") == "tool" or end_payload.get("tool_calls"))):
                    raise ValueError("摘要覆盖边界不能拆分 assistant 工具调用及其 tool 结果")
            cursor = self.conn.execute(
                "INSERT INTO summaries VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, session_id, summary_id, content, start_seq, end_seq,
                 datetime.now(timezone.utc).isoformat()),
            )
            if cursor.rowcount != 1:
                raise ValueError("摘要未保存")
            return self.get_summary(user_id=user_id, session_id=session_id, summary_id=summary_id)

    def list_summaries(self, *, user_id: str, session_id: str) -> list[dict]:
        self.message_store.validate_ids(user_id=user_id, session_id=session_id)
        return [dict(row) for row in self.conn.execute(
            """SELECT * FROM summaries WHERE user_id = ? AND session_id = ?
               ORDER BY end_seq DESC, recorded_at DESC, summary_id ASC""", (user_id, session_id),
        ).fetchall()]
