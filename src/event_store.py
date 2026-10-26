"""保存有真实消息来源的经历，不把待办状态推断为已发生事件。"""

from datetime import datetime, timezone
import json

from src.message_store import MessageStore


class EventStore:
    def __init__(self, message_store: MessageStore):
        self.message_store = message_store
        self.conn = message_store.conn
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    user_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    text TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    phase TEXT NOT NULL DEFAULT 'executed'
                        CHECK (phase IN ('planned', 'executed', 'verified')),
                    participants_json TEXT NOT NULL DEFAULT '[]',
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, event_id)
                )
            """)
            columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(events)")}
            if "phase" not in columns:
                self.conn.execute("""
                    ALTER TABLE events ADD COLUMN phase TEXT NOT NULL DEFAULT 'executed'
                        CHECK (phase IN ('planned', 'executed', 'verified'))
                """)
            if "participants_json" not in columns:
                self.conn.execute("""
                    ALTER TABLE events ADD COLUMN participants_json TEXT NOT NULL DEFAULT '[]'
                """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS event_sources (
                    user_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    source_message_id TEXT NOT NULL,
                    PRIMARY KEY (user_id, event_id, source_session_id, source_message_id),
                    FOREIGN KEY (user_id, event_id) REFERENCES events (user_id, event_id),
                    FOREIGN KEY (user_id, source_session_id, source_message_id)
                        REFERENCES messages (user_id, session_id, message_id)
                )
            """)

    @staticmethod
    def _time(value, field_name):
        if not isinstance(value, str):
            raise ValueError(f"{field_name} 必须是带时区的时间字符串")
        parsed = datetime.fromisoformat(value)
        if parsed.utcoffset() is None:
            raise ValueError(f"{field_name} 必须包含时区")
        return parsed.astimezone(timezone.utc).isoformat()

    def get_event(self, *, user_id: str, event_id: str) -> dict | None:
        self.message_store.validate_ids(user_id=user_id, event_id=event_id)
        row = self.conn.execute(
            "SELECT * FROM events WHERE user_id = ? AND event_id = ?", (user_id, event_id),
        ).fetchone()
        if row is None:
            return None
        event = dict(row)
        event["payload"] = json.loads(event.pop("payload_json"))
        event["participants"] = json.loads(event.pop("participants_json"))
        event["sources"] = [{
            "user_id": user_id, "session_id": source[0], "message_id": source[1],
        } for source in self.conn.execute(
            """SELECT source_session_id, source_message_id FROM event_sources
               WHERE user_id = ? AND event_id = ? ORDER BY source_session_id, source_message_id""",
            (user_id, event_id),
        ).fetchall()]
        return event

    def record_event(self, *, user_id: str, event_id: str, occurred_at: str = None,
                     text: str = "", payload: dict | None = None, sources: list[dict] = None,
                     phase: str = "executed", participants: list[str] | None = None) -> dict:
        self.message_store.validate_ids(user_id=user_id, event_id=event_id)
        with self.conn:
            existing = self.get_event(user_id=user_id, event_id=event_id)
            if existing is not None:
                return existing
            self.message_store.assert_user_active(user_id=user_id)
            occurred = self._time(occurred_at, "occurred_at")
            if payload is None:
                payload = {}
            if not isinstance(payload, dict) or not isinstance(text, str) or not (text.strip() or payload):
                raise ValueError("事件必须有文本或非空字典 payload")
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            if phase not in {"planned", "executed", "verified"}:
                raise ValueError("phase 必须为 planned、executed 或 verified")
            if participants is None:
                participants = []
            if not isinstance(participants, list):
                raise ValueError("participants 必须是字符串列表")
            participant_values = []
            seen_participants = set()
            for participant in participants:
                self.message_store.validate_ids(participant=participant)
                if participant not in seen_participants:
                    participant_values.append(participant)
                    seen_participants.add(participant)
            participants_json = json.dumps(participant_values, ensure_ascii=False, allow_nan=False)
            if not isinstance(sources, list) or not sources:
                raise ValueError("sources 必须是非空列表")
            refs = set()
            for source in sources:
                if not isinstance(source, dict):
                    raise ValueError("来源必须是字典")
                ref = {name: source.get(name) for name in ("user_id", "session_id", "message_id")}
                self.message_store.validate_ids(**ref)
                if ref["user_id"] != user_id or self.message_store.get_message(**ref) is None:
                    raise ValueError("来源必须是真实存在的同用户消息")
                refs.add((ref["session_id"], ref["message_id"]))
            cursor = self.conn.execute(
                """INSERT INTO events (
                       user_id, event_id, occurred_at, text, payload_json,
                       phase, participants_json, recorded_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, event_id, occurred, text, encoded, phase, participants_json,
                 datetime.now(timezone.utc).isoformat()),
            )
            if cursor.rowcount != 1:
                raise ValueError("事件未保存")
            cursor = self.conn.executemany(
                "INSERT INTO event_sources VALUES (?, ?, ?, ?)",
                [(user_id, event_id, session, message) for session, message in sorted(refs)],
            )
            if cursor.rowcount != len(refs):
                raise ValueError("事件来源未完整保存")
            return self.get_event(user_id=user_id, event_id=event_id)

    def list_events(self, *, user_id: str, start_at: str | None = None,
                    end_at: str | None = None) -> list[dict]:
        """按事件时间返回左闭右开区间；不按写入时间冒充发生时间。"""
        self.message_store.validate_ids(user_id=user_id)
        start = None if start_at is None else self._time(start_at, "start_at")
        end = None if end_at is None else self._time(end_at, "end_at")
        if start is not None and end is not None and start >= end:
            raise ValueError("end_at 必须晚于 start_at")
        sql = "SELECT event_id FROM events WHERE user_id = ?"
        parameters = [user_id]
        for boundary, value in ((">=", start), ("<", end)):
            if value is not None:
                sql += f" AND occurred_at {boundary} ?"
                parameters.append(value)
        return [self.get_event(user_id=user_id, event_id=row[0]) for row in self.conn.execute(
            sql + " ORDER BY occurred_at, event_id", parameters,
        ).fetchall()]
