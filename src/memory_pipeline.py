"""单写入者的消息提取、事实规划和执行；首候选及固定计划持久化后再写事实。"""

from copy import deepcopy
from datetime import datetime, timezone
import json
from typing import Callable

from src.candidate_extractor import CandidateExtractor
from src.conflict_store import ConflictStore
from src.fact_relation_judge import judge_fact_relation
from src.fact_store import FactStore
from src.memory_updater import MemoryUpdater, _ALLOWED_FIELDS
from src.model_adapter import ModelAdapter, ModelConfig, default_id_factory


class MemoryPipeline:
    def __init__(
        self, fact_store: FactStore, adapter: ModelAdapter | None = None, *,
        config: ModelConfig | None = None,
        id_factory: Callable[[str, dict, int], str] = default_id_factory,
        conflict_store: ConflictStore | None = None,
        cardinality_by_predicate: dict | None = None,
        merge_duplicate_sources: bool = False,
    ):
        if type(merge_duplicate_sources) is not bool:
            raise ValueError("merge_duplicate_sources 必须为布尔值")
        rules = {} if cardinality_by_predicate is None else cardinality_by_predicate
        if not isinstance(rules, dict):
            raise ValueError("cardinality_by_predicate 必须为字典")
        for predicate, mode in rules.items():
            fact_store.message_store.validate_ids(predicate=predicate)
            if not isinstance(mode, str) or mode not in {"single", "multi", "unknown"}:
                raise ValueError("谓词规则必须为 single、multi 或 unknown")
        if conflict_store is not None and conflict_store.conn is not fact_store.conn:
            raise ValueError("争议存储必须与事实存储使用同一连接")
        self.fact_store = fact_store
        self.message_store = fact_store.message_store
        self.conn = fact_store.conn
        self.updater = MemoryUpdater(fact_store)
        self.extractor = CandidateExtractor(adapter, config=config, id_factory=id_factory)
        self.id_factory = id_factory
        self.conflict_store = conflict_store
        self.rules = dict(rules)
        self.merge_duplicate_sources = merge_duplicate_sources
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_pipeline_runs (
                    user_id TEXT NOT NULL, session_id TEXT NOT NULL, message_id TEXT NOT NULL,
                    state_json TEXT NOT NULL, recorded_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, session_id, message_id),
                    FOREIGN KEY (user_id, session_id, message_id)
                        REFERENCES messages (user_id, session_id, message_id)
                )
            """)

    def load_fact_snapshot(self, *, user_id: str) -> list[dict]:
        """完整读取该用户所有版本/状态/有效区间，不使用单时间点 active 视图。"""
        self.message_store.validate_ids(user_id=user_id)
        self.message_store.assert_user_active(user_id=user_id)
        rows = self.conn.execute(
            "SELECT fact_id, version FROM facts WHERE user_id = ? ORDER BY fact_id, version",
            (user_id,),
        ).fetchall()
        result = []
        for row in rows:
            fact = self.fact_store.get_fact(user_id=user_id, fact_id=row["fact_id"], version=row["version"])
            assert fact is not None
            result.append(fact)
        return result

    def _load(self, source):
        row = self.conn.execute(
            """SELECT state_json FROM memory_pipeline_runs
               WHERE user_id = ? AND session_id = ? AND message_id = ?""",
            (source["user_id"], source["session_id"], source["message_id"]),
        ).fetchone()
        if row is None:
            return None
        state = json.loads(row["state_json"])
        self._validate_state(source, state)
        return state

    @staticmethod
    def _result(source, state):
        return {
            **source, "status": "completed",
            "items": [{"candidate": candidate, **item}
                      for candidate, item in zip(state["candidates"], state["items"])],
        }

    def _validate_state(self, source, state):
        """损坏的进度不能触发模型或写入，也不能从缓存返回另一用户的数据。"""
        try:
            keys = {"candidates", "conflict_ids", "items", "result", "rules",
                    "merge_duplicate_sources", "record_conflicts"}
            if not isinstance(state, dict) or set(state) != keys:
                raise ValueError("流水线运行记录结构不合法")
            if any(not isinstance(state[key], list) for key in ("candidates", "conflict_ids", "items")):
                raise ValueError("流水线运行记录的候选和进度必须为列表")
            if not len(state["candidates"]) == len(state["conflict_ids"]) == len(state["items"]):
                raise ValueError("流水线候选与进度数量不一致")
            if type(state["merge_duplicate_sources"]) is not bool or type(state["record_conflicts"]) is not bool:
                raise ValueError("流水线运行记录的开关必须为布尔值")
            if not isinstance(state["rules"], dict):
                raise ValueError("流水线运行记录的谓词规则不合法")
            for predicate, mode in state["rules"].items():
                self.message_store.validate_ids(predicate=predicate)
                if not isinstance(mode, str) or mode not in {"single", "multi", "unknown"}:
                    raise ValueError("流水线运行记录的谓词规则不合法")
            for conflict_id in state["conflict_ids"]:
                self.message_store.validate_ids(conflict_id=conflict_id)
            if len(set(state["conflict_ids"])) != len(state["conflict_ids"]):
                raise ValueError("流水线运行记录中的争议身份重复")
            for candidate, item in zip(state["candidates"], state["items"]):
                judge_fact_relation(user_id=source["user_id"], candidate=candidate, existing_facts=[])
                if candidate["sources"] != [source]:
                    raise ValueError("流水线候选的来源不属于输入消息")
                if candidate.get("scope") == "session" and candidate.get("session_id") != source["session_id"]:
                    raise ValueError("流水线候选的会话作用域不匹配")
                if item is None:
                    continue
                if not isinstance(item, dict) or set(item) != {"plan", "related_facts", "status", "result", "conflict"}:
                    raise ValueError("流水线候选进度不合法")
                if item["status"] not in {"planned", "applied", "noop", "conflict", "deferred"}:
                    raise ValueError("流水线候选进度状态不合法")
                plan = item["plan"]
                if not isinstance(plan, dict) or set(plan) != {"relation", "target", "reason", "decision"}:
                    raise ValueError("流水线计划结构不合法")
                if plan["relation"] not in {"SUPPLEMENT", "DUPLICATE", "MULTIVALUE", "CORRECTION",
                                            "RETRACTION", "TEMPORAL_CHANGE", "UNRESOLVED"}:
                    raise ValueError("流水线计划关系不合法")
                if not isinstance(plan["reason"], str) or not plan["reason"].strip():
                    raise ValueError("流水线计划缺少原因")
                if not isinstance(item["related_facts"], list):
                    raise ValueError("流水线相关事实引用不合法")
                for ref in item["related_facts"] + ([] if plan["target"] is None else [plan["target"]]):
                    if not isinstance(ref, dict) or set(ref) != {"fact_id", "version"}:
                        raise ValueError("流水线事实引用不合法")
                    self.message_store.validate_ids(fact_id=ref["fact_id"])
                    if type(ref["version"]) is not int or ref["version"] < 1:
                        raise ValueError("流水线事实版本不合法")
                decision = plan["decision"]
                if decision is not None:
                    if not isinstance(decision, dict) or not isinstance(decision.get("action"), str):
                        raise ValueError("流水线决策不合法")
                    action = decision["action"]
                    if action not in _ALLOWED_FIELDS or set(decision) - _ALLOWED_FIELDS[action]:
                        raise ValueError("流水线决策包含不支持的动作或字段")
                result = item["result"]
                if result is not None:
                    if not isinstance(result, dict) or decision is None or result.get("action") != decision["action"]:
                        raise ValueError("流水线执行结果与决策不一致")
                    for key in ("fact", "previous_fact"):
                        fact = result.get(key)
                        if fact is not None and (not isinstance(fact, dict) or fact.get("user_id") != source["user_id"]):
                            raise ValueError("流水线执行结果包含另一用户的数据")
                        if fact is not None:
                            if not isinstance(fact.get("sources"), list) or not fact["sources"]:
                                raise ValueError("流水线执行结果缺少事实来源")
                            for ref in fact["sources"]:
                                if not isinstance(ref, dict) or ref.get("user_id") != source["user_id"]:
                                    raise ValueError("流水线执行结果来源不属于当前用户")
                conflict = item["conflict"]
                if conflict is not None and (not isinstance(conflict, dict) or conflict.get("user_id") != source["user_id"]):
                    raise ValueError("流水线争议结果包含另一用户的数据")
                if item["status"] in {"applied", "noop"} and result is None:
                    raise ValueError("已执行的候选缺少结果")
                if item["status"] == "applied" and (decision["action"] == "NOOP" or not isinstance(result.get("fact"), dict)):
                    raise ValueError("已执行的事实动作缺少事实结果")
                if item["status"] == "noop" and (decision["action"] != "NOOP" or result.get("fact") is not None):
                    raise ValueError("NOOP 进度的执行结果不合法")
                if item["status"] == "conflict" and conflict is None:
                    raise ValueError("已记录的争议缺少结果")
                if item["status"] in {"planned", "deferred"} and (result is not None or conflict is not None):
                    raise ValueError("未执行的候选不能含写入结果")
            if state["result"] is not None:
                if any(item is None or item["status"] == "planned" for item in state["items"]):
                    raise ValueError("未完成的流水线不能含完成结果")
                if state["result"] != self._result(source, state):
                    raise ValueError("流水线完成结果与原进度不一致")
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("流水线运行记录结构损坏") from error

    def _save(self, source, state, *, new=False):
        self._validate_state(source, state)
        encoded = json.dumps(state, ensure_ascii=False, allow_nan=False)
        identity = (source["user_id"], source["session_id"], source["message_id"])
        with self.conn:
            self.message_store.assert_user_active(user_id=source["user_id"])
            if new:
                cursor = self.conn.execute(
                    """INSERT INTO memory_pipeline_runs
                       (user_id, session_id, message_id, state_json, recorded_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (*identity, encoded, datetime.now(timezone.utc).isoformat()),
                )
            else:
                cursor = self.conn.execute(
                    """UPDATE memory_pipeline_runs SET state_json = ?
                       WHERE user_id = ? AND session_id = ? AND message_id = ?""",
                    (encoded, *identity),
                )
            if cursor.rowcount != 1:
                raise ValueError("流水线进度未保存")

    @staticmethod
    def _related_refs(candidate, snapshot):
        fields = ("subject_id", "predicate", "scope", "session_id")
        context = {field: candidate.get(field, "user" if field == "scope" else None) for field in fields}
        return [
            {"fact_id": fact["fact_id"], "version": fact["version"]}
            for fact in snapshot if all(fact[field] == context[field] for field in fields)
        ]

    def process_message(self, *, user_id: str, session_id: str, message_id: str) -> dict:
        """处理已保存消息。异常向外传播；重试复用首候选与已保存的固定计划。"""
        source = {"user_id": user_id, "session_id": session_id, "message_id": message_id}
        self.message_store.validate_ids(**source)
        self.message_store.assert_user_active(user_id=user_id)
        state = self._load(source)
        if state is None:
            message = self.message_store.get_message(**source)
            if message is None:
                raise ValueError("输入消息在当前用户/会话下不存在")
            candidates = self.extractor.extract(
                user_id=user_id, message=message, existing_facts=self.load_fact_snapshot(user_id=user_id),
            )
            conflict_ids = [self.id_factory("conflict", dict(source), index) for index in range(len(candidates))]
            for conflict_id in conflict_ids:
                self.message_store.validate_ids(conflict_id=conflict_id)
            if len(set(conflict_ids)) != len(conflict_ids):
                raise ValueError("id_factory 为不同候选返回了相同争议 ID")
            state = {
                "candidates": candidates, "conflict_ids": conflict_ids,
                "items": [None] * len(candidates), "result": None,
                "rules": self.rules, "merge_duplicate_sources": self.merge_duplicate_sources,
                "record_conflicts": self.conflict_store is not None,
            }
            self._save(source, state, new=True)
        if state["result"] is not None:
            return deepcopy(state["result"])

        for index, candidate in enumerate(state["candidates"]):
            item = state["items"][index]
            if item is None:
                snapshot = self.load_fact_snapshot(user_id=user_id)
                plan = judge_fact_relation(
                    user_id=user_id, candidate=candidate, existing_facts=snapshot,
                    cardinality=state["rules"].get(candidate["predicate"], "unknown"),
                    merge_duplicate_sources=state["merge_duplicate_sources"],
                )
                item = {
                    "plan": plan, "related_facts": self._related_refs(candidate, snapshot),
                    "status": "planned", "result": None, "conflict": None,
                }
                state["items"][index] = item
                self._save(source, state)
            if item["status"] != "planned":
                continue
            self.message_store.assert_user_active(user_id=user_id)
            plan = item["plan"]
            if plan["decision"] is not None:
                item["result"] = self.updater.apply(user_id=user_id, decision=plan["decision"])
                item["status"] = "noop" if plan["decision"]["action"] == "NOOP" else "applied"
            elif plan["relation"] == "UNRESOLVED" and item["related_facts"] and state["record_conflicts"]:
                if self.conflict_store is None:
                    raise ValueError("此任务需要争议存储，请使用原有配置重试")
                item["conflict"] = self.conflict_store.record_conflict(
                    user_id=user_id, conflict_id=state["conflict_ids"][index],
                    candidate=candidate, related_facts=item["related_facts"], reason=plan["reason"],
                )
                item["status"] = "conflict"
            else:
                item["status"] = "deferred"
            self._save(source, state)

        state["result"] = self._result(source, state)
        self._save(source, state)
        return deepcopy(state["result"])

    def handle_job(self, payload: dict, job: dict) -> dict:
        """BackgroundWorker 处理器；任务身份不能覆盖可信 job.user_id。"""
        if not isinstance(payload, dict) or set(payload) != {"user_id", "session_id", "message_id"}:
            raise ValueError("提取任务 payload 必须只含 user_id/session_id/message_id")
        if not isinstance(job, dict) or job.get("user_id") != payload["user_id"]:
            raise ValueError("提取任务用户与后台任务归属不一致")
        return self.process_message(**payload)
