from src.fact_store import FactStore


_ALLOWED_FIELDS = {
    "ADD": {
        "action", "fact_id", "subject_id", "predicate", "object", "sources",
        "scope", "session_id", "valid_from", "valid_to",
    },
    "UPDATE": {"action", "fact_id", "new_version", "object", "sources"},
    "DELETE": {"action", "fact_id", "expected_version"},
    "TRANSITION": {
        "action", "fact_id", "new_version", "new_fact_id", "effective_at",
        "object", "sources", "valid_to",
    },
    "NOOP": {"action"},
}


class MemoryUpdater:
    def __init__(self, fact_store: FactStore):
        self.fact_store = fact_store

    def apply(self, *, user_id: str, decision: dict) -> dict:
        """执行一条已确定的决策，写入事务由 FactStore 管理。"""
        self.fact_store.message_store.validate_ids(user_id=user_id)
        if not isinstance(decision, dict):
            raise ValueError("decision 必须是字典")

        action = decision.get("action")
        if not isinstance(action, str) or action not in _ALLOWED_FIELDS:
            raise ValueError("action 必须是 ADD、UPDATE、DELETE、TRANSITION 或 NOOP")
        extra_fields = set(decision) - _ALLOWED_FIELDS[action]
        if extra_fields:
            raise ValueError(f"{action} 包含不支持的字段：{extra_fields!r}")

        if action == "NOOP":
            return {"action": action, "fact": None}

        # 缺失字段交给存储层校验，使已有目标版本的重试仍能直接返回。
        if action == "ADD":
            fact = self.fact_store.add_fact(
                user_id=user_id,
                fact_id=decision.get("fact_id"),
                subject_id=decision.get("subject_id"),
                predicate=decision.get("predicate"),
                object=decision.get("object"),
                sources=decision.get("sources"),
                scope=decision.get("scope", "user"),
                session_id=decision.get("session_id"),
                valid_from=decision.get("valid_from"),
                valid_to=decision.get("valid_to"),
                version=1,
                status="active",
            )
        elif action == "UPDATE":
            fact = self.fact_store.update_fact(
                user_id=user_id,
                fact_id=decision.get("fact_id"),
                new_version=decision.get("new_version"),
                object=decision.get("object"),
                sources=decision.get("sources"),
            )
        elif action == "TRANSITION":
            result = self.fact_store.transition_fact(
                user_id=user_id,
                fact_id=decision.get("fact_id"),
                new_version=decision.get("new_version"),
                new_fact_id=decision.get("new_fact_id"),
                effective_at=decision.get("effective_at"),
                object=decision.get("object"),
                sources=decision.get("sources"),
                valid_to=decision.get("valid_to"),
            )
            return {"action": action, **result}
        else:
            fact = self.fact_store.retract_fact(
                user_id=user_id,
                fact_id=decision.get("fact_id"),
                expected_version=decision.get("expected_version"),
            )
        return {"action": action, "fact": fact}
