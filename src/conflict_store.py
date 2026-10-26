"""保存争议依据，并原子执行人工指定的解决结果。"""

from datetime import datetime, timezone
import json

from src.fact_store import FactStore


_RESOLUTION_SCHEMA = """
    CREATE TABLE {table} (
        user_id TEXT NOT NULL,
        conflict_id TEXT NOT NULL,
        outcome TEXT NOT NULL CHECK (
            outcome IN ('KEEP_EXISTING', 'ADD_CANDIDATE', 'CORRECT', 'RETRACT', 'TRANSITION')
        ),
        decision_json TEXT NOT NULL,
        result_json TEXT NOT NULL,
        reason TEXT NOT NULL,
        resolved_at TEXT NOT NULL,
        PRIMARY KEY (user_id, conflict_id),
        FOREIGN KEY (user_id, conflict_id)
            REFERENCES conflicts (user_id, conflict_id)
    )
"""


class ConflictStore:
    def __init__(self, fact_store: FactStore):
        self.fact_store = fact_store
        self.message_store = fact_store.message_store
        self.conn = fact_store.conn
        self._create_schema()

    def _create_schema(self) -> None:
        cursor = self.conn.cursor()
        try:
            with self.conn:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conflicts (
                        user_id TEXT NOT NULL,
                        conflict_id TEXT NOT NULL,
                        candidate_json TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'open',
                        recorded_at TEXT NOT NULL,
                        PRIMARY KEY (user_id, conflict_id)
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conflict_facts (
                        user_id TEXT NOT NULL,
                        conflict_id TEXT NOT NULL,
                        fact_id TEXT NOT NULL,
                        version INTEGER NOT NULL CHECK (version >= 1),
                        PRIMARY KEY (user_id, conflict_id, fact_id, version),
                        FOREIGN KEY (user_id, conflict_id)
                            REFERENCES conflicts (user_id, conflict_id),
                        FOREIGN KEY (user_id, fact_id, version)
                            REFERENCES facts (user_id, fact_id, version)
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conflict_sources (
                        user_id TEXT NOT NULL,
                        conflict_id TEXT NOT NULL,
                        source_session_id TEXT NOT NULL,
                        source_message_id TEXT NOT NULL,
                        PRIMARY KEY (
                            user_id, conflict_id, source_session_id, source_message_id
                        ),
                        FOREIGN KEY (user_id, conflict_id)
                            REFERENCES conflicts (user_id, conflict_id),
                        FOREIGN KEY (user_id, source_session_id, source_message_id)
                            REFERENCES messages (user_id, session_id, message_id)
                    )
                """)
                self._ensure_resolution_schema(cursor)
        finally:
            cursor.close()

    def _ensure_resolution_schema(self, cursor) -> None:
        """在已有建表事务内升级三种 outcome 的旧 CHECK，原样保留处理快照。"""
        row = cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'conflict_resolutions'"
        ).fetchone()
        if row is None:
            cursor.execute(_RESOLUTION_SCHEMA.format(table="conflict_resolutions"))
            return
        schema_sql = row["sql"].upper()
        if "'RETRACT'" in schema_sql and "'TRANSITION'" in schema_sql:
            return
        # 表名固定在代码中；名称已占用时拒绝迁移，不清理未知对象。
        cursor.execute(_RESOLUTION_SCHEMA.format(table="conflict_resolutions_v2"))
        cursor.execute("""
            INSERT INTO conflict_resolutions_v2
                (user_id, conflict_id, outcome, decision_json, result_json, reason, resolved_at)
            SELECT user_id, conflict_id, outcome, decision_json, result_json, reason, resolved_at
            FROM conflict_resolutions
        """)
        cursor.execute("DROP TABLE conflict_resolutions")
        cursor.execute("ALTER TABLE conflict_resolutions_v2 RENAME TO conflict_resolutions")

    def get_conflict(self, *, user_id: str, conflict_id: str) -> dict | None:
        self.message_store.validate_ids(user_id=user_id, conflict_id=conflict_id)
        cursor = self.conn.cursor()
        try:
            row = cursor.execute(
                "SELECT * FROM conflicts WHERE user_id = ? AND conflict_id = ?",
                (user_id, conflict_id),
            ).fetchone()
            if row is None:
                return None
            conflict = dict(row)
            candidate = json.loads(conflict.pop("candidate_json"))
            source_rows = cursor.execute(
                """SELECT user_id, source_session_id, source_message_id
                   FROM conflict_sources WHERE user_id = ? AND conflict_id = ?
                   ORDER BY source_session_id, source_message_id""",
                (user_id, conflict_id),
            ).fetchall()
            candidate["sources"] = [
                {
                    "user_id": source["user_id"],
                    "session_id": source["source_session_id"],
                    "message_id": source["source_message_id"],
                }
                for source in source_rows
            ]
            conflict["candidate"] = candidate
            fact_rows = cursor.execute(
                """SELECT fact_id, version FROM conflict_facts
                   WHERE user_id = ? AND conflict_id = ? ORDER BY fact_id, version""",
                (user_id, conflict_id),
            ).fetchall()
            conflict["related_facts"] = [dict(fact) for fact in fact_rows]
            return conflict
        finally:
            cursor.close()

    def record_conflict(
        self, *, user_id: str, conflict_id: str,
        candidate: dict | None = None,
        related_facts: list[dict] | None = None,
        reason: str | None = None,
    ) -> dict:
        """同 ID 返回首次记录；新记录的三表写入在一个事务内提交。"""
        self.message_store.validate_ids(user_id=user_id, conflict_id=conflict_id)
        self.message_store.assert_user_active(user_id=user_id)
        with self.conn:
            existing = self.get_conflict(user_id=user_id, conflict_id=conflict_id)
            if existing is not None:
                return existing
            if not isinstance(candidate, dict) or "user_id" in candidate:
                raise ValueError("candidate 必须是字典，且不能包含顶层 user_id")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("reason 必须是非空字符串")

            # 来源只有关系表这一份；复制后剥离，避免修改调用方的候选。
            payload = dict(candidate)
            sources = payload.pop("sources", None)
            candidate_json = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            if not isinstance(sources, list) or not sources:
                raise ValueError("candidate.sources 必须是非空列表")
            source_refs = []
            seen_sources = set()
            for source in sources:
                if not isinstance(source, dict):
                    raise ValueError("每条来源必须是字典")
                ref = {name: source.get(name) for name in ("user_id", "session_id", "message_id")}
                self.message_store.validate_ids(**ref)
                if ref["user_id"] != user_id:
                    raise ValueError("候选来源必须属于同一用户")
                key = (ref["user_id"], ref["session_id"], ref["message_id"])
                if key not in seen_sources:
                    if self.message_store.get_message(**ref) is None:
                        raise ValueError("候选来源消息不存在")
                    source_refs.append(key)
                    seen_sources.add(key)

            if not isinstance(related_facts, list) or not related_facts:
                raise ValueError("related_facts 必须是非空列表")
            fact_refs = []
            seen_facts = set()
            for ref in related_facts:
                if not isinstance(ref, dict) or set(ref) != {"fact_id", "version"}:
                    raise ValueError("每个相关事实引用必须只包含 fact_id 和 version")
                # 精确版本读取负责 ID/正整数校验，不以最新 active 版本替代历史。
                fact = self.fact_store.get_fact(user_id=user_id, **ref)
                if fact is None:
                    raise ValueError("相关事实版本在当前用户下不存在")
                key = (ref["fact_id"], ref["version"])
                if key not in seen_facts:
                    fact_refs.append(key)
                    seen_facts.add(key)

            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    """INSERT INTO conflicts
                       (user_id, conflict_id, candidate_json, reason, status, recorded_at)
                       VALUES (?, ?, ?, ?, 'open', ?)""",
                    (user_id, conflict_id, candidate_json, reason,
                     datetime.now(timezone.utc).isoformat()),
                )
                if cursor.rowcount != 1:
                    raise ValueError("争议记录未保存")
                for fact_id, version in fact_refs:
                    cursor.execute(
                        """INSERT INTO conflict_facts (user_id, conflict_id, fact_id, version)
                           VALUES (?, ?, ?, ?)""",
                        (user_id, conflict_id, fact_id, version),
                    )
                    if cursor.rowcount != 1:
                        raise ValueError("相关事实引用未保存")
                for source_user_id, session_id, message_id in source_refs:
                    cursor.execute(
                        """INSERT INTO conflict_sources
                           (user_id, conflict_id, source_session_id, source_message_id)
                           VALUES (?, ?, ?, ?)""",
                        (source_user_id, conflict_id, session_id, message_id),
                    )
                    if cursor.rowcount != 1:
                        raise ValueError("候选来源引用未保存")
                saved = self.get_conflict(user_id=user_id, conflict_id=conflict_id)
                assert saved is not None
            finally:
                cursor.close()
        return saved

    def list_conflicts(self, *, user_id: str) -> list[dict]:
        self.message_store.validate_ids(user_id=user_id)
        cursor = self.conn.cursor()
        try:
            rows = cursor.execute(
                """SELECT conflict_id FROM conflicts WHERE user_id = ?
                   ORDER BY recorded_at, conflict_id""",
                (user_id,),
            ).fetchall()
        finally:
            cursor.close()
        conflicts = []
        for row in rows:
            conflict = self.get_conflict(user_id=user_id, conflict_id=row["conflict_id"])
            assert conflict is not None
            conflicts.append(conflict)
        return conflicts

    def get_resolution(self, *, user_id: str, conflict_id: str) -> dict | None:
        """读取处理时的决定和结果快照，不将其作为当前事实状态。"""
        self.message_store.validate_ids(user_id=user_id, conflict_id=conflict_id)
        cursor = self.conn.cursor()
        try:
            row = cursor.execute(
                """SELECT * FROM conflict_resolutions
                   WHERE user_id = ? AND conflict_id = ?""",
                (user_id, conflict_id),
            ).fetchone()
            if row is None:
                return None
            resolution = dict(row)
            resolution["decision"] = json.loads(resolution.pop("decision_json"))
            resolution["result"] = json.loads(resolution.pop("result_json"))
            return resolution
        finally:
            cursor.close()

    def resolve_conflict(
        self, *, user_id: str, conflict_id: str,
        outcome: str | None = None,
        target: dict | None = None,
        new_fact_id: str | None = None,
        reason: str | None = None,
    ) -> dict:
        """决定、事实写入和关闭争议整体提交；已解决 ID 只返回首次记录。"""
        self.message_store.validate_ids(user_id=user_id, conflict_id=conflict_id)
        self.message_store.assert_user_active(user_id=user_id)
        with self.conn:
            existing = self.get_resolution(user_id=user_id, conflict_id=conflict_id)
            if existing is not None:
                return existing
            conflict = self.get_conflict(user_id=user_id, conflict_id=conflict_id)
            if conflict is None or conflict["status"] != "open":
                raise ValueError("争议必须存在且为 open 状态")
            if not isinstance(outcome, str) or outcome not in {
                "KEEP_EXISTING", "ADD_CANDIDATE", "CORRECT", "RETRACT", "TRANSITION",
            }:
                raise ValueError("outcome 必须是 KEEP_EXISTING、ADD_CANDIDATE、CORRECT、RETRACT 或 TRANSITION")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("解决原因必须是非空字符串")
            if outcome not in {"CORRECT", "RETRACT", "TRANSITION"} and target is not None:
                raise ValueError("只有 CORRECT、RETRACT、TRANSITION 可以指定 target")
            if outcome not in {"ADD_CANDIDATE", "TRANSITION"} and new_fact_id is not None:
                raise ValueError("只有 ADD_CANDIDATE、TRANSITION 可以指定 new_fact_id")

            candidate = conflict["candidate"]
            fact = None
            previous_fact = None
            if outcome == "KEEP_EXISTING":
                decision = {"action": "NOOP"}
            elif outcome == "ADD_CANDIDATE":
                self.message_store.validate_ids(new_fact_id=new_fact_id)
                if self.fact_store.get_latest_fact(user_id=user_id, fact_id=new_fact_id) is not None:
                    raise ValueError("新事实 ID 已被使用，不能作为本次解决结果")
                decision = {
                    "action": "ADD", "fact_id": new_fact_id,
                    "subject_id": candidate.get("subject_id"),
                    "predicate": candidate.get("predicate"),
                    "object": candidate.get("object"), "sources": candidate["sources"],
                    "scope": candidate.get("scope", "user"),
                    "session_id": candidate.get("session_id"),
                    "valid_from": candidate.get("valid_from"),
                    "valid_to": candidate.get("valid_to"),
                }
                fact = self.fact_store._insert_fact(
                    user_id=user_id, version=1, status="active",
                    **{key: value for key, value in decision.items() if key != "action"},
                )
            else:
                if not isinstance(target, dict) or set(target) != {"fact_id", "version"}:
                    raise ValueError("target 必须只包含 fact_id 和 version")
                self.message_store.validate_ids(fact_id=target["fact_id"])
                if type(target["version"]) is not int or target["version"] < 1:
                    raise ValueError("target.version 必须为正整数")
                if target not in conflict["related_facts"]:
                    raise ValueError("target 必须是争议中已记录的相关事实版本")
                current = self.fact_store.get_latest_fact(user_id=user_id, fact_id=target["fact_id"])
                if (current is None or current["version"] != target["version"]
                        or current["status"] != "active"):
                    raise ValueError("目标已变化，不能按旧版本解决争议")
                for field in ("subject_id", "predicate", "scope", "session_id"):
                    value = candidate.get(field, "user" if field == "scope" else None)
                    if value != current[field]:
                        raise ValueError("候选与目标的主体、谓词或精确作用域不匹配")
                if outcome == "CORRECT":
                    for field in ("valid_from", "valid_to"):
                        candidate_time = self.fact_store._parse_optional_time(candidate.get(field), field)
                        current_time = self.fact_store._parse_optional_time(current[field], field)
                        if candidate_time != current_time:
                            raise ValueError("纠错的有效区间必须相同，时间变化应另行处理")
                    decision = {
                        "action": "UPDATE", "fact_id": target["fact_id"],
                        "new_version": target["version"] + 1,
                        "object": candidate.get("object"), "sources": candidate["sources"],
                    }
                    fact = self.fact_store._update_fact(
                        user_id=user_id,
                        **{key: value for key, value in decision.items() if key != "action"},
                    )
                elif outcome == "RETRACT":
                    decision = {
                        "action": "DELETE", "fact_id": target["fact_id"],
                        "expected_version": target["version"],
                    }
                    fact = self.fact_store._retract_fact(
                        user_id=user_id, fact_id=target["fact_id"],
                        expected_version=target["version"],
                    )
                else:
                    self.message_store.validate_ids(new_fact_id=new_fact_id)
                    if self.fact_store.get_latest_fact(user_id=user_id, fact_id=new_fact_id) is not None:
                        raise ValueError("新事实 ID 已被使用，不能作为本次解决结果")
                    decision = {
                        "action": "TRANSITION", "fact_id": target["fact_id"],
                        "new_version": target["version"] + 1, "new_fact_id": new_fact_id,
                        "effective_at": candidate.get("valid_from"),
                        "valid_to": candidate.get("valid_to"),
                        "object": candidate.get("object"), "sources": candidate["sources"],
                    }
                    transition = self.fact_store._transition_fact(
                        user_id=user_id,
                        **{key: value for key, value in decision.items() if key != "action"},
                    )
                    fact = transition["fact"]
                    previous_fact = transition["previous_fact"]

            result = {"action": decision["action"], "fact": fact}
            if outcome == "TRANSITION":
                result["previous_fact"] = previous_fact
            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    """INSERT INTO conflict_resolutions
                       (user_id, conflict_id, outcome, decision_json, result_json, reason, resolved_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (user_id, conflict_id, outcome,
                     json.dumps(decision, ensure_ascii=False, allow_nan=False),
                     json.dumps(result, ensure_ascii=False, allow_nan=False),
                     reason, datetime.now(timezone.utc).isoformat()),
                )
                if cursor.rowcount != 1:
                    raise ValueError("争议解决记录未保存")
                cursor.execute(
                    """UPDATE conflicts SET status = 'resolved'
                       WHERE user_id = ? AND conflict_id = ? AND status = 'open'""",
                    (user_id, conflict_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("争议状态已变化，无法完成解决")
                saved = self.get_resolution(user_id=user_id, conflict_id=conflict_id)
                assert saved is not None
            finally:
                cursor.close()
        return saved
