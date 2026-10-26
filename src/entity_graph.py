"""用户隔离的实体别名、关系来源与有界图扩展。"""

from collections import deque
from datetime import datetime, timezone
import json
import unicodedata

from src.message_store import MessageStore


class EntityGraphStore:
    def __init__(self, message_store: MessageStore):
        self.message_store = message_store
        self.conn = message_store.conn
        self._create_schema()

    def _create_schema(self) -> None:
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    user_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    attributes_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('active', 'merged')),
                    merged_into TEXT,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, entity_id),
                    FOREIGN KEY (user_id, merged_into) REFERENCES entities (user_id, entity_id)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_aliases (
                    user_id TEXT NOT NULL,
                    normalized_alias TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    PRIMARY KEY (user_id, normalized_alias, entity_id),
                    FOREIGN KEY (user_id, entity_id) REFERENCES entities (user_id, entity_id)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_relations (
                    user_id TEXT NOT NULL,
                    relation_id TEXT NOT NULL,
                    subject_entity_id TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_entity_id TEXT NOT NULL,
                    attributes_json TEXT NOT NULL,
                    valid_from TEXT,
                    valid_to TEXT,
                    status TEXT NOT NULL CHECK (status IN ('active', 'retracted')),
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, relation_id),
                    FOREIGN KEY (user_id, subject_entity_id) REFERENCES entities (user_id, entity_id),
                    FOREIGN KEY (user_id, object_entity_id) REFERENCES entities (user_id, entity_id)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_relation_sources (
                    user_id TEXT NOT NULL,
                    relation_id TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    source_message_id TEXT NOT NULL,
                    PRIMARY KEY (
                        user_id, relation_id, source_session_id, source_message_id
                    ),
                    FOREIGN KEY (user_id, relation_id)
                        REFERENCES entity_relations (user_id, relation_id),
                    FOREIGN KEY (user_id, source_session_id, source_message_id)
                        REFERENCES messages (user_id, session_id, message_id)
                )
            """)

    @staticmethod
    def normalize_alias(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("alias 必须是非空字符串")
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    @staticmethod
    def _strict_json(value, name: str) -> str:
        if not isinstance(value, dict):
            raise ValueError(f"{name} 必须是字典")
        try:
            return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} 必须能严格 JSON 编码") from error

    @staticmethod
    def _time(value: str | None, name: str):
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{name} 必须是带时区的时间字符串或 None")
        parsed = datetime.fromisoformat(value)
        if parsed.utcoffset() is None:
            raise ValueError(f"{name} 必须包含时区")
        return parsed.astimezone(timezone.utc)

    def _entity(self, row) -> dict | None:
        if row is None:
            return None
        entity = dict(row)
        entity["attributes"] = json.loads(entity.pop("attributes_json"))
        aliases = self.conn.execute("""
            SELECT alias FROM entity_aliases
            WHERE user_id = ? AND entity_id = ? ORDER BY normalized_alias, alias
        """, (entity["user_id"], entity["entity_id"])).fetchall()
        entity["aliases"] = [item["alias"] for item in aliases]
        return entity

    def get_entity(self, *, user_id: str, entity_id: str) -> dict | None:
        self.message_store.validate_ids(user_id=user_id, entity_id=entity_id)
        row = self.conn.execute(
            "SELECT * FROM entities WHERE user_id = ? AND entity_id = ?",
            (user_id, entity_id),
        ).fetchone()
        return self._entity(row)

    def _canonical_entity(self, *, user_id: str, entity_id: str) -> dict | None:
        seen = set()
        while entity_id not in seen:
            seen.add(entity_id)
            entity = self.get_entity(user_id=user_id, entity_id=entity_id)
            if entity is None or entity["status"] == "active":
                return entity
            entity_id = entity["merged_into"]
        raise ValueError("实体重定向存在循环")

    def add_entity(
        self,
        *,
        user_id: str,
        entity_id: str,
        entity_type: str | None = None,
        canonical_name: str | None = None,
        aliases: list[str] | None = None,
        attributes: dict | None = None,
    ) -> dict:
        self.message_store.validate_ids(user_id=user_id, entity_id=entity_id)
        existing = self.get_entity(user_id=user_id, entity_id=entity_id)
        if existing is not None:
            return existing
        self.message_store.assert_user_active(user_id=user_id)
        self.message_store.validate_ids(
            entity_type=entity_type, canonical_name=canonical_name,
        )
        if aliases is None:
            aliases = []
        if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
            raise ValueError("aliases 必须是字符串列表")
        attributes_json = self._strict_json({} if attributes is None else attributes, "attributes")
        alias_values = []
        seen = set()
        for alias in [canonical_name, *aliases]:
            normalized = self.normalize_alias(alias)
            if normalized not in seen:
                alias_values.append((normalized, alias))
                seen.add(normalized)
        recorded_at = datetime.now(timezone.utc).isoformat()
        with self.conn:
            cursor = self.conn.execute("""
                INSERT INTO entities (
                    user_id, entity_id, entity_type, canonical_name, attributes_json,
                    status, merged_into, recorded_at
                ) VALUES (?, ?, ?, ?, ?, 'active', NULL, ?)
            """, (
                user_id, entity_id, entity_type, canonical_name, attributes_json, recorded_at,
            ))
            if cursor.rowcount != 1:
                raise ValueError("实体未保存")
            cursor = self.conn.executemany("""
                INSERT INTO entity_aliases (user_id, normalized_alias, entity_id, alias)
                VALUES (?, ?, ?, ?)
            """, [
                (user_id, normalized, entity_id, alias)
                for normalized, alias in alias_values
            ])
            if cursor.rowcount != len(alias_values):
                raise ValueError("实体别名未完整保存")
        saved = self.get_entity(user_id=user_id, entity_id=entity_id)
        assert saved is not None
        return saved

    def resolve_entity(self, *, user_id: str, name: str) -> list[dict]:
        self.message_store.validate_ids(user_id=user_id)
        normalized = self.normalize_alias(name)
        rows = self.conn.execute("""
            SELECT e.* FROM entity_aliases a
            JOIN entities e ON e.user_id = a.user_id AND e.entity_id = a.entity_id
            WHERE a.user_id = ? AND a.normalized_alias = ?
            ORDER BY e.entity_id
        """, (user_id, normalized)).fetchall()
        result = []
        seen = set()
        for row in rows:
            entity = self._canonical_entity(user_id=user_id, entity_id=row["entity_id"])
            if entity is not None and entity["entity_id"] not in seen:
                result.append(entity)
                seen.add(entity["entity_id"])
        return result

    def merge_entity(self, *, user_id: str, source_entity_id: str, target_entity_id: str) -> dict:
        self.message_store.validate_ids(
            user_id=user_id, source_entity_id=source_entity_id, target_entity_id=target_entity_id,
        )
        if source_entity_id == target_entity_id:
            raise ValueError("不能把实体合并到自身")
        self.message_store.assert_user_active(user_id=user_id)
        with self.conn:
            source = self.get_entity(user_id=user_id, entity_id=source_entity_id)
            target = self.get_entity(user_id=user_id, entity_id=target_entity_id)
            if source is not None and source["status"] == "merged":
                if source["merged_into"] == target_entity_id:
                    return source
                raise ValueError("源实体已经合并到其他实体")
            if source is None or target is None or target["status"] != "active":
                raise ValueError("源实体和目标 active 实体必须存在")
            cursor = self.conn.execute("""
                UPDATE entities SET status = 'merged', merged_into = ?
                WHERE user_id = ? AND entity_id = ? AND status = 'active'
            """, (target_entity_id, user_id, source_entity_id))
            if cursor.rowcount != 1:
                raise ValueError("实体状态已变化")
        saved = self.get_entity(user_id=user_id, entity_id=source_entity_id)
        assert saved is not None
        return saved

    def _relation(self, row) -> dict | None:
        if row is None:
            return None
        relation = dict(row)
        relation["attributes"] = json.loads(relation.pop("attributes_json"))
        sources = self.conn.execute("""
            SELECT user_id, source_session_id, source_message_id
            FROM entity_relation_sources
            WHERE user_id = ? AND relation_id = ?
            ORDER BY source_session_id, source_message_id
        """, (relation["user_id"], relation["relation_id"])).fetchall()
        relation["sources"] = [{
            "user_id": source["user_id"],
            "session_id": source["source_session_id"],
            "message_id": source["source_message_id"],
        } for source in sources]
        return relation

    def get_relation(self, *, user_id: str, relation_id: str) -> dict | None:
        self.message_store.validate_ids(user_id=user_id, relation_id=relation_id)
        row = self.conn.execute(
            "SELECT * FROM entity_relations WHERE user_id = ? AND relation_id = ?",
            (user_id, relation_id),
        ).fetchone()
        return self._relation(row)

    def add_relation(
        self,
        *,
        user_id: str,
        relation_id: str,
        subject_entity_id: str | None = None,
        predicate: str | None = None,
        object_entity_id: str | None = None,
        sources: list[dict] | None = None,
        attributes: dict | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
    ) -> dict:
        self.message_store.validate_ids(user_id=user_id, relation_id=relation_id)
        existing = self.get_relation(user_id=user_id, relation_id=relation_id)
        if existing is not None:
            return existing
        self.message_store.assert_user_active(user_id=user_id)
        self.message_store.validate_ids(
            subject_entity_id=subject_entity_id, predicate=predicate,
            object_entity_id=object_entity_id,
        )
        for entity_id in (subject_entity_id, object_entity_id):
            entity = self.get_entity(user_id=user_id, entity_id=entity_id)
            if entity is None or entity["status"] != "active":
                raise ValueError("关系端点必须是当前用户的 active 实体")
        attributes_json = self._strict_json({} if attributes is None else attributes, "attributes")
        start = self._time(valid_from, "valid_from")
        end = self._time(valid_to, "valid_to")
        if start is not None and end is not None and end <= start:
            raise ValueError("valid_to 必须晚于 valid_from")
        if not isinstance(sources, list) or not sources:
            raise ValueError("sources 必须是非空列表")
        source_refs = []
        seen = set()
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("每条来源必须是字典")
            ref = {name: source.get(name) for name in ("user_id", "session_id", "message_id")}
            self.message_store.validate_ids(**ref)
            if ref["user_id"] != user_id or self.message_store.get_message(**ref) is None:
                raise ValueError("关系来源必须是当前用户的真实消息")
            key = (ref["session_id"], ref["message_id"])
            if key not in seen:
                source_refs.append(key)
                seen.add(key)
        recorded_at = datetime.now(timezone.utc).isoformat()
        with self.conn:
            cursor = self.conn.execute("""
                INSERT INTO entity_relations (
                    user_id, relation_id, subject_entity_id, predicate, object_entity_id,
                    attributes_json, valid_from, valid_to, status, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
            """, (
                user_id, relation_id, subject_entity_id, predicate, object_entity_id,
                attributes_json, None if start is None else start.isoformat(),
                None if end is None else end.isoformat(), recorded_at,
            ))
            if cursor.rowcount != 1:
                raise ValueError("关系未保存")
            cursor = self.conn.executemany("""
                INSERT INTO entity_relation_sources (
                    user_id, relation_id, source_session_id, source_message_id
                ) VALUES (?, ?, ?, ?)
            """, [
                (user_id, relation_id, session_id, message_id)
                for session_id, message_id in source_refs
            ])
            if cursor.rowcount != len(source_refs):
                raise ValueError("关系来源未完整保存")
        saved = self.get_relation(user_id=user_id, relation_id=relation_id)
        assert saved is not None
        return saved

    def expand(
        self,
        *,
        user_id: str,
        entity_ids: list[str],
        max_depth: int = 2,
        max_nodes: int = 50,
        as_of: str | None = None,
    ) -> dict:
        self.message_store.validate_ids(user_id=user_id)
        if not isinstance(entity_ids, list) or not entity_ids:
            raise ValueError("entity_ids 必须是非空列表")
        if type(max_depth) is not int or max_depth < 0:
            raise ValueError("max_depth 必须是非负整数")
        if type(max_nodes) is not int or max_nodes < 1:
            raise ValueError("max_nodes 必须是正整数")
        instant = self._time(as_of, "as_of") or datetime.now(timezone.utc)
        queue = deque()
        depths = {}
        for entity_id in entity_ids:
            self.message_store.validate_ids(entity_id=entity_id)
            entity = self._canonical_entity(user_id=user_id, entity_id=entity_id)
            if entity is None:
                raise ValueError("起始实体不存在")
            resolved_id = entity["entity_id"]
            if resolved_id not in depths and len(depths) < max_nodes:
                depths[resolved_id] = 0
                queue.append(resolved_id)
        relations = {}
        while queue:
            entity_id = queue.popleft()
            depth = depths[entity_id]
            if depth >= max_depth:
                continue
            rows = self.conn.execute("""
                WITH RECURSIVE redirects(entity_id) AS (
                    SELECT ? UNION
                    SELECT e.entity_id FROM entities e JOIN redirects d
                        ON e.merged_into = d.entity_id
                    WHERE e.user_id = ?
                )
                SELECT * FROM entity_relations
                WHERE user_id = ? AND status = 'active'
                  AND (subject_entity_id IN (SELECT entity_id FROM redirects)
                       OR object_entity_id IN (SELECT entity_id FROM redirects))
                ORDER BY relation_id
            """, (entity_id, user_id, user_id)).fetchall()
            for row in rows:
                start = self._time(row["valid_from"], "valid_from")
                end = self._time(row["valid_to"], "valid_to")
                if (start is not None and instant < start) or (end is not None and instant >= end):
                    continue
                relation = self._relation(row)
                # 查询导出规范端点；get_relation 仍保留原始关系和证据。
                for field in ("subject_entity_id", "object_entity_id"):
                    relation[field] = self._canonical_entity(
                        user_id=user_id, entity_id=relation[field],
                    )["entity_id"]
                neighbor = (
                    relation["object_entity_id"]
                    if relation["subject_entity_id"] == entity_id
                    else relation["subject_entity_id"]
                )
                if neighbor not in depths:
                    if len(depths) >= max_nodes:
                        continue
                    depths[neighbor] = depth + 1
                    queue.append(neighbor)
                relations[relation["relation_id"]] = relation
        entities = []
        for entity_id, depth in sorted(depths.items(), key=lambda item: (item[1], item[0])):
            entity = self.get_entity(user_id=user_id, entity_id=entity_id)
            if entity is not None:
                entities.append({**entity, "depth": depth})
        return {
            "entities": entities,
            "relations": [relations[key] for key in sorted(relations)],
        }

    def find_query_entities(self, *, user_id: str, query: str) -> list[dict]:
        self.message_store.validate_ids(user_id=user_id)
        normalized_query = self.normalize_alias(query)
        rows = self.conn.execute("""
            SELECT DISTINCT e.* FROM entity_aliases a
            JOIN entities e ON e.user_id = a.user_id AND e.entity_id = a.entity_id
            WHERE a.user_id = ?
            ORDER BY e.entity_id
        """, (user_id,)).fetchall()
        result = {}
        for row in rows:
            entity = self._entity(row)
            if any(self.normalize_alias(alias) in normalized_query for alias in entity["aliases"]):
                canonical = self._canonical_entity(user_id=user_id, entity_id=entity["entity_id"])
                result[canonical["entity_id"]] = canonical
        return [result[key] for key in sorted(result)]


class GraphHistoryRetriever:
    def __init__(self, history_retriever, graph_store: EntityGraphStore):
        self.history_retriever = history_retriever
        self.graph_store = graph_store

    def search(
        self, *, user_id: str, query: str, top_k: int = 5,
        use_graph: bool = True, max_depth: int = 2,
    ) -> list[dict]:
        if type(use_graph) is not bool:
            raise ValueError("use_graph 必须为布尔值")
        if type(max_depth) is not int or max_depth < 0:
            raise ValueError("max_depth 必须是非负整数")
        baseline = self.history_retriever.search(user_id=user_id, query=query, top_k=top_k)
        if not use_graph or top_k == 0 or not query.strip():
            return baseline
        seeds = self.graph_store.find_query_entities(user_id=user_id, query=query)
        if not seeds:
            return baseline
        graph = self.graph_store.expand(
            user_id=user_id, entity_ids=[item["entity_id"] for item in seeds],
            max_depth=max_depth,
        )
        hits = list(baseline)
        identities = {
            (hit["message"]["session_id"], hit["message"]["message_id"])
            for hit in hits
        }
        for relation in graph["relations"]:
            for source in relation["sources"]:
                key = (source["session_id"], source["message_id"])
                if key in identities:
                    continue
                message = self.graph_store.message_store.get_message(**source)
                if message is not None:
                    hits.append({"message": message, "score": 0.5, "via": "graph"})
                    identities.add(key)
        return sorted(hits, key=lambda hit: hit["score"], reverse=True)[:top_k]

    def compare(self, *, user_id: str, query: str, top_k: int = 5) -> dict:
        return {
            "without_graph": self.search(
                user_id=user_id, query=query, top_k=top_k, use_graph=False,
            ),
            "with_graph": self.search(
                user_id=user_id, query=query, top_k=top_k, use_graph=True,
            ),
        }
