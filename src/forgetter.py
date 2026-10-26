"""用户级物理遗忘，并用墓碑阻止旧任务重新写入。"""

from datetime import datetime, timezone
import json

from src.message_store import MessageStore


class MemoryForgetter:
    def __init__(self, message_store: MessageStore):
        self.message_store = message_store
        self.conn = message_store.conn

    def get_receipt(self, *, user_id: str) -> dict | None:
        self.message_store.validate_ids(user_id=user_id)
        row = self.conn.execute("""
            SELECT * FROM memory_forget_tombstones WHERE user_id = ?
        """, (user_id,)).fetchone()
        if row is None:
            return None
        receipt = dict(row)
        receipt["deleted_rows"] = json.loads(receipt.pop("deleted_rows_json"))
        return receipt

    def _ordered_user_tables(self) -> list[str]:
        tables = [row["name"] for row in self.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """).fetchall()]
        tables = [table for table in tables if table != "memory_forget_tombstones" and any(
            column["name"] == "user_id"
            for column in self.conn.execute('PRAGMA table_info("' + table.replace('"', '""') + '")').fetchall()
        )]
        table_set = set(tables)
        edges = {table: set() for table in tables}
        for child in tables:
            for foreign_key in self.conn.execute('PRAGMA foreign_key_list("' + child.replace('"', '""') + '")').fetchall():
                parent = foreign_key["table"]
                if parent in table_set and parent != child:
                    edges[child].add(parent)

        ordered = []
        remaining = set(tables)
        while remaining:
            ready = sorted(
                table for table in remaining
                if not any(table in edges[other] for other in remaining)
            )
            if not ready:
                # 当前 schema 没有跨表环；保留明确错误，避免静默留下派生数据。
                raise ValueError("用户数据表存在无法安全排序的外键环")
            for table in ready:
                ordered.append(table)
                remaining.remove(table)
        return ordered

    def forget_user(self, *, user_id: str, reason: str | None = None) -> dict:
        self.message_store.validate_ids(user_id=user_id)
        existing = self.get_receipt(user_id=user_id)
        if existing is not None:
            return existing
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason 必须是非空字符串")
        forgotten_at = datetime.now(timezone.utc).isoformat()
        # 结束首次只读快照；零行 UPDATE 先取得写锁，使并发遗忘串行看到同一收据。
        self.conn.commit()
        with self.conn:
            self.conn.execute("""
                UPDATE memory_forget_tombstones
                SET forgotten_at = forgotten_at WHERE 0
            """)
            existing = self.get_receipt(user_id=user_id)
            if existing is not None:
                return existing
            cursor = self.conn.execute("""
                INSERT INTO memory_forget_tombstones (
                    user_id, forgotten_at, reason, deleted_rows_json
                ) VALUES (?, ?, ?, '{}')
            """, (user_id, forgotten_at, reason))
            if cursor.rowcount != 1:
                raise ValueError("遗忘墓碑未保存")
            deleted_rows = {}
            for table in self._ordered_user_tables():
                quoted = '"' + table.replace('"', '""') + '"'
                cursor = self.conn.execute(
                    f'DELETE FROM {quoted} WHERE user_id = ?', (user_id,),
                )
                deleted_rows[table] = cursor.rowcount
                if self.conn.execute(
                    f'SELECT 1 FROM {quoted} WHERE user_id = ? LIMIT 1', (user_id,),
                ).fetchone() is not None:
                    raise ValueError(f"{table} 仍有用户数据，遗忘未完成")
            cursor = self.conn.execute("""
                UPDATE memory_forget_tombstones SET deleted_rows_json = ?
                WHERE user_id = ?
            """, (
                json.dumps(deleted_rows, ensure_ascii=False, allow_nan=False, sort_keys=True),
                user_id,
            ))
            if cursor.rowcount != 1:
                raise ValueError("遗忘处理记录未保存")
        saved = self.get_receipt(user_id=user_id)
        assert saved is not None
        return saved
