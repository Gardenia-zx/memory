import json
from datetime import datetime, timezone

from src.message_store import MessageStore


class FactStore:
    def __init__(self, message_store: MessageStore):
        self.message_store = message_store
        self.conn = message_store.conn
        self._create_schema()

    def _create_schema(self) -> None:
        cursor = self.conn.cursor()
        try:
            with self.conn:
                cursor.execute('''
                               CREATE TABLE IF NOT EXISTS facts
                               (
                                   user_id
                                   TEXT
                                   NOT
                                   NULL,
                                   fact_id
                                   TEXT
                                   NOT
                                   NULL,

                                   version
                                   INTEGER
                                   NOT
                                   NULL
                                   DEFAULT
                                   1
                                   CHECK
                               (
                                   version
                                   >=
                                   1
                               ),

                                   subject_id TEXT NOT NULL,
                                   predicate TEXT NOT NULL,
                                   object_json TEXT NOT NULL,
                                   scope TEXT NOT NULL
                                   CHECK
                               (
                                   scope
                                   IN
                               (
                                   'user',
                                   'session'
                               )),

                                   session_id TEXT,

                                   valid_from TEXT,
                                   valid_to TEXT,

                                   status TEXT NOT NULL DEFAULT 'active'
                                   CHECK
                               (
                                   status
                                   IN
                               (
                                   'active',
                                   'superseded',
                                   'retracted'
                               )
                                   ),

                                   recorded_at TEXT NOT NULL,
                                   PRIMARY KEY
                               (
                                   user_id,
                                   fact_id,
                                   version
                               ),
                                   CHECK
                               (
                               (
                                   scope =
                                   'user'
                                   AND
                                   session_id
                                   IS
                                   NULL
                               )
                                   OR
                               (
                                   scope =
                                   'session'
                                   AND
                                   session_id
                                   IS
                                   NOT
                                   NULL
                                   AND
                                   length (
                                   trim
                               (
                                   session_id
                               )) > 0
                                   )
                                   )
                                   );''')

                cursor.execute('''
                               CREATE TABLE IF NOT EXISTS fact_sources
                               (
                                   user_id
                                   TEXT
                                   NOT
                                   NULL,
                                   fact_id
                                   TEXT
                                   NOT
                                   NULL,
                                   version
                                   INTEGER
                                   NOT
                                   NULL,
                                   source_session_id
                                   TEXT
                                   NOT
                                   NULL,
                                   source_message_id
                                   TEXT
                                   NOT
                                   NULL,

                                   PRIMARY
                                   KEY
                               (
                                   user_id,
                                   fact_id,
                                   version,
                                   source_session_id,
                                   source_message_id
                               ),
                                   FOREIGN KEY
                               (
                                   user_id,
                                   fact_id,
                                   version
                               )
                                   REFERENCES facts
                               (
                                   user_id,
                                   fact_id,
                                   version
                               ),
                                   FOREIGN KEY
                               (
                                   user_id,
                                   source_session_id,
                                   source_message_id
                               )
                                   REFERENCES messages
                               (
                                   user_id,
                                   session_id,
                                   message_id
                               )
                                   );
                               ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS fact_transitions (
                        user_id TEXT NOT NULL,
                        new_fact_id TEXT NOT NULL,
                        new_fact_version INTEGER NOT NULL DEFAULT 1
                            CHECK (new_fact_version = 1),
                        previous_fact_id TEXT NOT NULL,
                        previous_version INTEGER NOT NULL CHECK (previous_version >= 2),
                        PRIMARY KEY (user_id, new_fact_id),
                        FOREIGN KEY (user_id, new_fact_id, new_fact_version)
                            REFERENCES facts (user_id, fact_id, version),
                        FOREIGN KEY (user_id, previous_fact_id, previous_version)
                            REFERENCES facts (user_id, fact_id, version)
                    )
                ''')
        finally:
            cursor.close()

    def get_fact(
            self,
            *,
            user_id: str,
            fact_id: str,
            version: int = 1,
    ) -> dict | None:
        self.message_store.validate_ids(user_id=user_id, fact_id=fact_id)
        if type(version) is not int or version < 1:
            raise ValueError("version 必须为正整数")
        cursor = self.conn.cursor()
        try:
            row = cursor.execute(
                "SELECT * FROM facts WHERE user_id = ? AND fact_id = ? AND version = ?",
                (user_id, fact_id, version),
            ).fetchone()
            if row is None:
                return None
            fact = dict(row)
            fact["object"] = json.loads(fact.pop("object_json"))
            source_rows = cursor.execute(
                """SELECT user_id, source_session_id, source_message_id
                   FROM fact_sources
                   WHERE user_id = ? AND fact_id = ? AND version = ?
                   ORDER BY source_session_id ASC, source_message_id ASC""",
                (user_id, fact_id, version),
            ).fetchall()
            fact["sources"] = [
                {
                    "user_id": source["user_id"],
                    "session_id": source["source_session_id"],
                    "message_id": source["source_message_id"],
                }
                for source in source_rows
            ]
            return fact
        finally:
            cursor.close()

    @staticmethod
    def _parse_optional_time(value: str | None, field_name: str) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{field_name} 必须是带时区的时间字符串或 None")
        parsed = datetime.fromisoformat(value)
        if parsed.utcoffset() is None:
            raise ValueError(f"{field_name} 必须包含时区")
        return parsed.astimezone(timezone.utc)

    def add_fact(
            self,
            *,
            user_id: str,
            fact_id: str,
            subject_id: str,
            predicate: str,
            object: dict,
            sources: list[dict],
            scope: str = "user",
            version: int = 1,
            status: str | None = None,
            session_id: str | None = None,
            valid_from: str | None = None,
            valid_to: str | None = None,
    ) -> dict:
        with self.conn:
            saved = self._insert_fact(
                user_id=user_id,
                fact_id=fact_id,
                subject_id=subject_id,
                predicate=predicate,
                object=object,
                sources=sources,
                scope=scope,
                version=version,
                status=status,
                session_id=session_id,
                valid_from=valid_from,
                valid_to=valid_to,
            )
        return saved
    def _insert_fact(
            self,
            *,
            user_id: str,
            fact_id: str,
            subject_id: str,
            predicate: str,
            object: dict,
            sources: list[dict],
            scope: str = "user",
            version: int = 1,
            status: str | None = None,
            session_id: str | None = None,
            valid_from: str | None = None,
            valid_to: str | None = None,
    ) -> dict:
        """复用事实写入逻辑，事务的提交和回滚由调用方负责。"""
        self.message_store.validate_ids(user_id=user_id, fact_id=fact_id)
        self.message_store.assert_user_active(user_id=user_id)
        if type(version) is not int or version < 1:
            raise ValueError("version必须为正整数")
        existing = self.get_fact(user_id=user_id, fact_id=fact_id, version=version)
        if existing is not None:
            return existing

        self.message_store.validate_ids(subject_id=subject_id, predicate=predicate)
        if not isinstance(object, dict) or object.get("type") != "literal" or "value" not in object:
            raise ValueError("object 必须是包含 type='literal' 和 value 的字典")
        object_json = json.dumps(object, ensure_ascii=False, allow_nan=False)

        if status is None:
            status = "active"

        if not isinstance(status, str) or status not in ["active","superseded","retracted"]:
            raise ValueError("status必须为字符串且为有效字段")

        if scope == "user":
            if session_id is not None:
                raise ValueError("用户级事实的 session_id 必须为 None")
        elif scope == "session":
            self.message_store.validate_ids(session_id=session_id)
        else:
            raise ValueError("scope 必须为 user 或 session")

        valid_start = self._parse_optional_time(valid_from, "valid_from")
        valid_end = self._parse_optional_time(valid_to, "valid_to")
        if valid_start is not None and valid_end is not None and valid_end <= valid_start:
            raise ValueError("valid_to 必须晚于 valid_from")


        if not isinstance(sources, list) or not sources:
            raise ValueError("sources 必须是非空列表")
        source_refs = []
        seen_sources = set()
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("每条来源必须是字典")
            source_user_id = source.get("user_id")
            source_session_id = source.get("session_id")
            source_message_id = source.get("message_id")
            self.message_store.validate_ids(
                user_id=source_user_id,
                session_id=source_session_id,
                message_id=source_message_id,
            )
            if source_user_id != user_id:
                raise ValueError("来源消息必须属于同一用户")
            source_key = (source_user_id, source_session_id, source_message_id)
            if source_key not in seen_sources:
                source_refs.append(source_key)
                seen_sources.add(source_key)

        recorded_at = datetime.now(timezone.utc).isoformat()
        for source_user_id, source_session_id, source_message_id in source_refs:
            if self.message_store.get_message(
                user_id=source_user_id,
                session_id=source_session_id,
                message_id=source_message_id,
            ) is None:
                raise ValueError("来源消息不存在")

        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """INSERT INTO facts (
                       user_id, fact_id, version, subject_id, predicate, object_json,
                       scope, session_id, valid_from, valid_to, status, recorded_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id, fact_id, version, subject_id, predicate, object_json,
                    scope, session_id,
                    valid_start.isoformat() if valid_start is not None else None,
                    valid_end.isoformat() if valid_end is not None else None,
                    status,
                    recorded_at,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("事实记录未保存")
            cursor.executemany(
                """INSERT INTO fact_sources (
                       user_id, fact_id, version, source_session_id, source_message_id
                   ) VALUES (?, ?, ?, ?, ?)""",
                [
                    (source_user_id, fact_id, version, source_session_id, source_message_id)
                    for source_user_id, source_session_id, source_message_id in source_refs
                ],
            )
            if cursor.rowcount != len(source_refs):
                raise ValueError("事实来源未完整保存")
        finally:
            cursor.close()

        saved = self.get_fact(user_id=user_id, fact_id=fact_id, version=version)
        assert saved is not None
        return saved

    def list_facts(
            self,
            *,
            user_id: str,
            subject_id: str | None = None,
            predicate: str | None = None,
            session_id: str | None = None,
            as_of: str | None = None,
    ) -> list[dict]:
        """列出指定用户在某会话和有效时间内可用的 active 事实。"""
        self.message_store.validate_ids(user_id=user_id)
        for field_name, value in (
            ("subject_id", subject_id),
            ("predicate", predicate),
            ("session_id", session_id),
        ):
            if value is not None:
                self.message_store.validate_ids(**{field_name: value})
        query_time = self._parse_optional_time(as_of, "as_of")
        if query_time is None:
            query_time = datetime.now(timezone.utc)

        conditions = ["user_id = ?", "status = 'active'"]
        parameters = [user_id]
        if subject_id is not None:
            conditions.append("subject_id = ?")
            parameters.append(subject_id)
        if predicate is not None:
            conditions.append("predicate = ?")
            parameters.append(predicate)
        if session_id is None:
            conditions.append("scope = 'user'")
        else:
            conditions.append("(scope = 'user' OR (scope = 'session' AND session_id = ?))")
            parameters.append(session_id)

        cursor = self.conn.cursor()
        try:
            rows = cursor.execute(
                "SELECT fact_id, version, valid_from, valid_to FROM facts WHERE "
                + " AND ".join(conditions)
                + " ORDER BY fact_id ASC, version ASC",
                parameters,
            ).fetchall()
        finally:
            cursor.close()

        facts = []
        for row in rows:
            valid_start = self._parse_optional_time(row["valid_from"], "valid_from")
            valid_end = self._parse_optional_time(row["valid_to"], "valid_to")
            if valid_start is not None and query_time < valid_start:
                continue
            if valid_end is not None and query_time >= valid_end:
                continue
            fact = self.get_fact(user_id=user_id, fact_id=row["fact_id"], version=row["version"])
            assert fact is not None
            facts.append(fact)
        return facts

    def get_latest_fact(
            self,
            *,
            user_id: str,
            fact_id: str,
    ) -> dict | None:
        self.message_store.validate_ids(user_id=user_id,fact_id=fact_id)
        cursor = self.conn.cursor()
        try:
            row = cursor.execute('''
                                     SELECT version
                                     FROM facts
                                     WHERE user_id = ?
                                       AND fact_id = ?
                                     ORDER BY version DESC LIMIT 1''',(user_id,fact_id)).fetchone()
            if row is None:
                return None
            return self.get_fact(user_id=user_id, fact_id=fact_id, version=row["version"])
        finally:
            cursor.close()

    def update_fact(
            self,
            *,
            user_id: str,
            fact_id: str,
            new_version: int,
            object: dict,
            sources: list[dict],
    ) -> dict:
        with self.conn:
            saved = self._update_fact(
                user_id=user_id, fact_id=fact_id, new_version=new_version,
                object=object, sources=sources,
            )
        return saved

    def _update_fact(
            self,
            *,
            user_id: str,
            fact_id: str,
            new_version: int,
            object: dict,
            sources: list[dict],
    ) -> dict:
        """不提交事务，供公开更新入口及争议解决组合写入复用。"""
        self.message_store.validate_ids(user_id=user_id, fact_id=fact_id)
        self.message_store.assert_user_active(user_id=user_id)
        if type(new_version) is not int or new_version < 2:
            raise ValueError("new_version 必须是至少为 2 的整数")
        existing_fact = self.get_fact(user_id=user_id, fact_id=fact_id, version=new_version)
        if existing_fact is not None:
            return existing_fact
        current_fact = self.get_latest_fact(user_id=user_id, fact_id=fact_id)
        if current_fact is None:
            raise ValueError("事实不存在，无法更新")
        if current_fact["status"] != "active":
            raise ValueError("只有 active 状态的最新事实才能更新")
        if new_version != current_fact["version"] + 1:
            raise ValueError("new_version 必须等于最新版本号加 1")

        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """UPDATE facts SET status = 'superseded'
                   WHERE user_id = ? AND fact_id = ? AND version = ?
                     AND status = 'active'""",
                (user_id, fact_id, current_fact["version"]),
            )
            if cursor.rowcount != 1:
                raise ValueError("事实状态已变化，无法更新")
            saved = self._insert_fact(
                user_id=user_id,
                fact_id=fact_id,
                subject_id=current_fact["subject_id"],
                predicate=current_fact["predicate"],
                object=object,
                scope=current_fact["scope"],
                session_id=current_fact["session_id"],
                valid_from=current_fact["valid_from"],
                valid_to=current_fact["valid_to"],
                sources=sources,
                version=new_version,
                status="active",
            )
        finally:
            cursor.close()
        return saved

    def transition_fact(
            self,
            *,
            user_id: str,
            fact_id: str,
            new_version: int,
            new_fact_id: str,
            effective_at: str,
            object: dict,
            sources: list[dict],
            valid_to: str | None = None,
    ) -> dict:
        """原子关闭旧区间并新增事实；迁移身份为 (user_id, new_fact_id)。"""
        with self.conn:
            saved = self._transition_fact(
                user_id=user_id, fact_id=fact_id, new_version=new_version,
                new_fact_id=new_fact_id, effective_at=effective_at,
                object=object, sources=sources, valid_to=valid_to,
            )
        return saved

    def _transition_fact(
            self,
            *,
            user_id: str,
            fact_id: str,
            new_version: int,
            new_fact_id: str,
            effective_at: str,
            object: dict,
            sources: list[dict],
            valid_to: str | None = None,
    ) -> dict:
        """不提交事务，供公开入口及争议解决组合写入复用。"""
        self.message_store.validate_ids(user_id=user_id, new_fact_id=new_fact_id)
        self.message_store.assert_user_active(user_id=user_id)
        cursor = self.conn.cursor()
        try:
            receipt = cursor.execute(
                """SELECT previous_fact_id, previous_version FROM fact_transitions
                       WHERE user_id = ? AND new_fact_id = ?""",
                (user_id, new_fact_id),
            ).fetchone()
            if receipt is not None:
                return {
                    "previous_fact": self.get_fact(
                        user_id=user_id, fact_id=receipt["previous_fact_id"],
                        version=receipt["previous_version"],
                    ),
                    "fact": self.get_fact(user_id=user_id, fact_id=new_fact_id, version=1),
                }

            self.message_store.validate_ids(fact_id=fact_id)
            if fact_id == new_fact_id:
                raise ValueError("时间变化的新事实必须使用不同的 fact_id")
            if type(new_version) is not int or new_version < 2:
                raise ValueError("new_version 必须是至少为 2 的整数")
            if self.get_latest_fact(user_id=user_id, fact_id=new_fact_id) is not None:
                raise ValueError("新事实 ID 已存在，且不是已完成的迁移")
            current = self.get_latest_fact(user_id=user_id, fact_id=fact_id)
            if current is None or current["status"] != "active":
                raise ValueError("只有存在且最新版本为 active 的事实才能迁移")
            if new_version != current["version"] + 1:
                raise ValueError("new_version 必须等于最新版本号加 1")

            effective = self._parse_optional_time(effective_at, "effective_at")
            if effective is None:
                raise ValueError("effective_at 必须为带时区的时间字符串")
            start = self._parse_optional_time(current["valid_from"], "valid_from")
            end = self._parse_optional_time(current["valid_to"], "valid_to")
            if ((start is not None and effective <= start)
                    or (end is not None and effective >= end)):
                raise ValueError("effective_at 必须严格位于旧事实的有效区间内部")

            new_fact = self._insert_fact(
                user_id=user_id, fact_id=new_fact_id, version=1, status="active",
                subject_id=current["subject_id"], predicate=current["predicate"],
                scope=current["scope"], session_id=current["session_id"],
                valid_from=effective.isoformat(), valid_to=valid_to,
                object=object, sources=sources,
            )
            cursor.execute(
                """UPDATE facts SET status = 'superseded'
                       WHERE user_id = ? AND fact_id = ? AND version = ?
                         AND status = 'active'""",
                (user_id, fact_id, current["version"]),
            )
            if cursor.rowcount != 1:
                raise ValueError("事实状态已变化，无法迁移")
            previous_fact = self._insert_fact(
                user_id=user_id, fact_id=fact_id, version=new_version, status="active",
                subject_id=current["subject_id"], predicate=current["predicate"],
                scope=current["scope"], session_id=current["session_id"],
                valid_from=current["valid_from"], valid_to=effective.isoformat(),
                object=current["object"], sources=current["sources"] + new_fact["sources"],
            )
            cursor.execute(
                """INSERT INTO fact_transitions
                       (user_id, new_fact_id, previous_fact_id, previous_version)
                       VALUES (?, ?, ?, ?)""",
                (user_id, new_fact_id, fact_id, new_version),
            )
            if cursor.rowcount != 1:
                raise ValueError("迁移记录未保存，无法完成迁移")
            result = {"previous_fact": previous_fact, "fact": new_fact}
        finally:
            cursor.close()
        return result

    def retract_fact(
            self,
            *,
            user_id: str,
            fact_id: str,
            expected_version: int,
    ) -> dict:
        """撤回指定的最新版本，保留事实内容及来源。"""
        with self.conn:
            saved = self._retract_fact(
                user_id=user_id, fact_id=fact_id, expected_version=expected_version,
            )
        return saved

    def _retract_fact(
            self,
            *,
            user_id: str,
            fact_id: str,
            expected_version: int,
    ) -> dict:
        """不提交事务，供公开入口及争议解决组合写入复用。"""
        self.message_store.validate_ids(user_id=user_id, fact_id=fact_id)
        self.message_store.assert_user_active(user_id=user_id)
        if type(expected_version) is not int or expected_version < 1:
            raise ValueError("expected_version 必须是至少为 1 的整数")

        current_fact = self.get_latest_fact(user_id=user_id, fact_id=fact_id)
        if current_fact is None:
            raise ValueError("事实不存在，无法撤回")
        if current_fact["version"] != expected_version:
            raise ValueError("expected_version 必须等于最新版本号")
        if current_fact["status"] == "retracted":
            return current_fact
        if current_fact["status"] != "active":
            raise ValueError("只有 active 状态的最新事实才能撤回")

        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """UPDATE facts SET status = 'retracted'
                       WHERE user_id = ? AND fact_id = ? AND version = ?
                         AND status = 'active'""",
                (user_id, fact_id, expected_version),
            )
            if cursor.rowcount != 1:
                raise ValueError("事实状态已变化，无法撤回")
        finally:
            cursor.close()

        saved = self.get_fact(user_id=user_id, fact_id=fact_id, version=expected_version)
        assert saved is not None
        return saved

