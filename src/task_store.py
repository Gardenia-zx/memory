"""持久化待办；通过 list_tasks(status='pending') 恢复未完成工作。"""

from datetime import datetime, timezone
import json

from src.message_store import MessageStore


class TaskStore:
    def __init__(self, message_store: MessageStore):
        self.message_store = message_store
        self.conn = message_store.conn
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    user_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'done', 'cancelled')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, task_id)
                )
            """)

    @staticmethod
    def _export(row):
        if row is None:
            return None
        task = dict(row)
        task["payload"] = json.loads(task.pop("payload_json"))
        return task

    def get_task(self, *, user_id: str, task_id: str) -> dict | None:
        self.message_store.validate_ids(user_id=user_id, task_id=task_id)
        return self._export(self.conn.execute(
            "SELECT * FROM tasks WHERE user_id = ? AND task_id = ?", (user_id, task_id),
        ).fetchone())

    def create_task(self, *, user_id: str, task_id: str, title: str = None,
                    payload: dict | None = None) -> dict:
        self.message_store.validate_ids(user_id=user_id, task_id=task_id)
        with self.conn:
            existing = self.get_task(user_id=user_id, task_id=task_id)
            if existing is not None:
                return existing
            self.message_store.assert_user_active(user_id=user_id)
            if not isinstance(title, str) or not title.strip():
                raise ValueError("title 必须是非空字符串")
            if payload is None:
                payload = {}
            if not isinstance(payload, dict):
                raise ValueError("payload 必须是字典")
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            now = datetime.now(timezone.utc).isoformat()
            cursor = self.conn.execute(
                "INSERT INTO tasks VALUES (?, ?, ?, ?, 'pending', ?, ?)",
                (user_id, task_id, title, encoded, now, now),
            )
            if cursor.rowcount != 1:
                raise ValueError("任务未保存")
            return self.get_task(user_id=user_id, task_id=task_id)

    def set_status(self, *, user_id: str, task_id: str, status: str) -> dict:
        self.message_store.validate_ids(user_id=user_id, task_id=task_id)
        if not isinstance(status, str) or status not in {"pending", "done", "cancelled"}:
            raise ValueError("status 必须为 pending、done 或 cancelled")
        with self.conn:
            task = self.get_task(user_id=user_id, task_id=task_id)
            if task is None:
                raise ValueError("任务不存在")
            if task["status"] == status:
                return task
            cursor = self.conn.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE user_id = ? AND task_id = ?",
                (status, datetime.now(timezone.utc).isoformat(), user_id, task_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("任务状态未保存")
            return self.get_task(user_id=user_id, task_id=task_id)

    def list_tasks(self, *, user_id: str, status: str | None = None) -> list[dict]:
        self.message_store.validate_ids(user_id=user_id)
        if status is not None and (not isinstance(status, str) or status not in {"pending", "done", "cancelled"}):
            raise ValueError("status 必须为 pending、done 或 cancelled")
        sql = "SELECT * FROM tasks WHERE user_id = ?"
        parameters = [user_id]
        if status is not None:
            sql += " AND status = ?"
            parameters.append(status)
        return [self._export(row) for row in self.conn.execute(
            sql + " ORDER BY created_at, task_id", parameters,
        ).fetchall()]
