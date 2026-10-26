"""对手工结构化的新候选生成只读计划；执行重试直接复用已生成的 decision。"""

from copy import deepcopy
from datetime import datetime, timezone
import json

from src.message_store import MessageStore


_COMMON_FIELDS = {
    "intent", "subject_id", "predicate", "sources", "scope", "session_id",
    "valid_from", "valid_to",
}
_INTENT_FIELDS = {
    "assert": _COMMON_FIELDS | {"fact_id", "object"},
    "correct": _COMMON_FIELDS | {"object", "target"},
    "retract": _COMMON_FIELDS | {"target"},
    "change": _COMMON_FIELDS | {"fact_id", "object", "target"},
}
_CONTEXT_FIELDS = ("subject_id", "predicate", "scope", "session_id")


def _normalise_time(value, field_name):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} 必须是带时区的时间字符串或 None")
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError(f"{field_name} 必须包含时区")
    return parsed.astimezone(timezone.utc).isoformat()


def _normalise_fields(record, *, needs_object=True):
    fields = {name: record.get(name) for name in _CONTEXT_FIELDS}
    MessageStore.validate_ids(
        subject_id=fields["subject_id"], predicate=fields["predicate"],
    )
    fields["scope"] = record.get("scope", "user")
    if fields["scope"] == "user":
        if fields["session_id"] is not None:
            raise ValueError("用户级事实的 session_id 必须为 None")
    elif fields["scope"] == "session":
        MessageStore.validate_ids(session_id=fields["session_id"])
    else:
        raise ValueError("scope 必须为 user 或 session")

    fields["valid_from"] = _normalise_time(record.get("valid_from"), "valid_from")
    fields["valid_to"] = _normalise_time(record.get("valid_to"), "valid_to")
    if (fields["valid_from"] is not None and fields["valid_to"] is not None
            and datetime.fromisoformat(fields["valid_to"])
            <= datetime.fromisoformat(fields["valid_from"])):
        raise ValueError("valid_to 必须晚于 valid_from")

    if needs_object:
        value = record.get("object")
        if not isinstance(value, dict) or value.get("type") != "literal" or "value" not in value:
            raise ValueError("object 必须是包含 type='literal' 和 value 的字典")
        # 先按存储层的 JSON 形态归一，避免 tuple/list、键顺序影响比较。
        fields["object"] = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return fields


def _normalise_sources(sources, user_id):
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources 必须是非空列表")
    result = []
    seen = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("每条来源必须是字典")
        ref = {name: source.get(name) for name in ("user_id", "session_id", "message_id")}
        MessageStore.validate_ids(**ref)
        if ref["user_id"] != user_id:
            raise ValueError("来源消息必须属于同一用户")
        key = tuple(ref.values())
        if key not in seen:
            result.append(ref)
            seen.add(key)
    return result


def _normalise_target(target):
    if target is None:
        return None
    if not isinstance(target, dict) or set(target) != {"fact_id", "version"}:
        raise ValueError("target 必须只包含 fact_id 和 version")
    MessageStore.validate_ids(fact_id=target["fact_id"])
    if type(target["version"]) is not int or target["version"] < 1:
        raise ValueError("target.version 必须为正整数")
    return dict(target)


def _latest_facts(existing_facts, user_id):
    if not isinstance(existing_facts, list):
        raise ValueError("existing_facts 必须是列表")
    latest = {}
    seen = set()
    for record in existing_facts:
        if not isinstance(record, dict):
            raise ValueError("每条已有事实必须是字典")
        MessageStore.validate_ids(user_id=record.get("user_id"), fact_id=record.get("fact_id"))
        if record["user_id"] != user_id:
            raise ValueError("已有事实必须属于同一用户")
        version = record.get("version")
        if type(version) is not int or version < 1:
            raise ValueError("已有事实的 version 必须为正整数")
        status = record.get("status")
        if not isinstance(status, str) or status not in {"active", "superseded", "retracted"}:
            raise ValueError("已有事实的 status 不合法")
        key = (record["fact_id"], version)
        if key in seen:
            raise ValueError("快照中不能重复提供同一事实版本")
        seen.add(key)
        fact = {
            **_normalise_fields(record), "fact_id": record["fact_id"],
            "version": version, "status": status, "sources": record.get("sources"),
        }
        previous = latest.get(fact["fact_id"])
        if previous is None or version > previous["version"]:
            latest[fact["fact_id"]] = fact
    return latest


def _object_key(fact):
    # JSON 编码区分 true 和 1；Python 的直接相等比较会混淆两者。
    return json.dumps(fact["object"], ensure_ascii=False, allow_nan=False, sort_keys=True)


def _result(relation, reason, *, target=None, decision=None):
    return {
        "relation": relation,
        "target": None if target is None else {
            "fact_id": target["fact_id"], "version": target["version"],
        },
        "reason": reason,
        "decision": deepcopy(decision),
    }


def _duplicate_result(targets, sources, user_id, merge_duplicate_sources, reason):
    target = targets[0] if len(targets) == 1 else None
    if not merge_duplicate_sources:
        return _result("DUPLICATE", reason, target=target, decision={"action": "NOOP"})
    if target is None:
        return _result("UNRESOLVED", "存在多个同值事实，无法唯一选择来源合并目标")

    existing_sources = _normalise_sources(target["sources"], user_id)
    merged_sources = _normalise_sources(existing_sources + sources, user_id)
    if len(merged_sources) == len(existing_sources):
        return _result("DUPLICATE", "相同事实的来源也已全部保存", target=target,
                       decision={"action": "NOOP"})
    return _result("DUPLICATE", "保留旧版来源，在新版本合并重复事实的新增来源",
                   target=target, decision={
                       "action": "UPDATE", "fact_id": target["fact_id"],
                       "new_version": target["version"] + 1,
                       "object": target["object"], "sources": merged_sources,
                   })


def judge_fact_relation(
    *, user_id: str, candidate: dict, existing_facts: list[dict],
    cardinality: str = "unknown", merge_duplicate_sources: bool = False,
) -> dict:
    """判断新候选，不读写数据库，也不执行自然语言提取。

    existing_facts 须覆盖相关事实的所有有效区间，并含每个 fact_id 的最新
    版本（包括非 active 版本）；也可同时提供旧版本。单次 list_facts(as_of)
    不保证满足此契约。调用方负责快照完整性；实际写入仍由 FactStore 校验。
    开启来源合并时，选中的重复目标须带完整 sources；每次执行后用新快照
    规划下一候选，重试仅复用原 decision，不从旧快照生成另一条合并计划。
    """
    MessageStore.validate_ids(user_id=user_id)
    if not isinstance(candidate, dict):
        raise ValueError("candidate 必须是字典")
    intent = candidate.get("intent", "assert")
    if not isinstance(intent, str) or intent not in _INTENT_FIELDS:
        raise ValueError("intent 必须是 assert、correct、retract 或 change")
    if set(candidate) - _INTENT_FIELDS[intent]:
        raise ValueError(f"{intent} 候选包含不支持的字段")
    if not isinstance(cardinality, str) or cardinality not in {"single", "multi", "unknown"}:
        raise ValueError("cardinality 必须是 single、multi 或 unknown")
    if type(merge_duplicate_sources) is not bool:
        raise ValueError("merge_duplicate_sources 必须为布尔值")

    fields = _normalise_fields(candidate, needs_object=intent != "retract")
    sources = _normalise_sources(candidate.get("sources"), user_id)
    if intent == "assert" or "fact_id" in candidate:
        MessageStore.validate_ids(fact_id=candidate.get("fact_id"))
    target_ref = _normalise_target(candidate.get("target"))
    latest = _latest_facts(existing_facts, user_id)

    target = None
    if intent in {"correct", "retract"} or target_ref is not None:
        if target_ref is None:
            return _result("UNRESOLVED", "纠错或撤回需要明确的目标事实版本")
        target = latest.get(target_ref["fact_id"])
        if target is None or target["version"] != target_ref["version"]:
            return _result("UNRESOLVED", "目标不存在或不是快照中的最新版本")
        if target["status"] != "active":
            return _result("UNRESOLVED", "目标的最新版本不是 active", target=target)
        if any(fields[name] != target[name] for name in _CONTEXT_FIELDS):
            return _result("UNRESOLVED", "目标的主体、谓词或精确作用域不匹配", target=target)

    if intent == "retract":
        return _result("RETRACTION", "明确撤回指定版本", target=target, decision={
            "action": "DELETE", "fact_id": target["fact_id"],
            "expected_version": target["version"],
        })
    if intent == "change":
        if target is None or "fact_id" not in candidate or fields["valid_from"] is None:
            return _result("TEMPORAL_CHANGE", "时间迁移需要目标、新事实 ID 和明确生效时间",
                           target=target)
        if candidate["fact_id"] in latest:
            return _result("UNRESOLVED", "新增事实 ID 已存在；执行重试应复用原 decision",
                           target=target)
        effective = datetime.fromisoformat(fields["valid_from"])
        start = None if target["valid_from"] is None else datetime.fromisoformat(target["valid_from"])
        end = None if target["valid_to"] is None else datetime.fromisoformat(target["valid_to"])
        if ((start is not None and effective <= start)
                or (end is not None and effective >= end)):
            return _result("UNRESOLVED", "生效时间必须严格位于目标有效区间内部", target=target)

        new_end = None if fields["valid_to"] is None else datetime.fromisoformat(fields["valid_to"])
        if cardinality != "multi":
            for fact in latest.values():
                if (fact["fact_id"] == target["fact_id"] or fact["status"] != "active"
                        or any(fields[name] != fact[name] for name in _CONTEXT_FIELDS)):
                    continue
                other_start = None if fact["valid_from"] is None else datetime.fromisoformat(fact["valid_from"])
                other_end = None if fact["valid_to"] is None else datetime.fromisoformat(fact["valid_to"])
                if ((new_end is None or other_start is None or other_start < new_end)
                        and (other_end is None or effective < other_end)):
                    return _result("UNRESOLVED", "新有效区间与其他事实重叠，暂不自动解决争议",
                                   target=target)
        return _result("TEMPORAL_CHANGE", "明确迁移指定事实，原子关闭旧区间并新增事实",
                       target=target, decision={
                           "action": "TRANSITION", "fact_id": target["fact_id"],
                           "new_version": target["version"] + 1,
                           "new_fact_id": candidate["fact_id"],
                           "effective_at": fields["valid_from"],
                           "object": fields["object"], "sources": sources,
                           "valid_to": fields["valid_to"],
                       })
    if intent == "correct":
        if any(fields[name] != target[name] for name in ("valid_from", "valid_to")):
            return _result("TEMPORAL_CHANGE", "当前 UPDATE 不能修改有效区间", target=target)
        if _object_key(fields) == _object_key(target):
            return _duplicate_result([target], sources, user_id, merge_duplicate_sources,
                                     "目标已经保存相同的值")
        return _result("CORRECTION", "明确纠正指定事实，保留旧版本", target=target, decision={
            "action": "UPDATE", "fact_id": target["fact_id"],
            "new_version": target["version"] + 1,
            "object": fields["object"], "sources": sources,
        })

    if candidate["fact_id"] in latest:
        return _result("UNRESOLVED", "新增事实 ID 已存在；执行重试应复用原 decision",
                       target=latest[candidate["fact_id"]])
    related = [
        fact for fact in latest.values()
        if fact["status"] == "active"
        and all(fields[name] == fact[name] for name in _CONTEXT_FIELDS)
    ]
    add_decision = {
        "action": "ADD", "fact_id": candidate["fact_id"], **fields, "sources": sources,
    }
    if not related:
        return _result("SUPPLEMENT", "没有相关的 active 事实，作为独立补充",
                       decision=add_decision)
    if any(
        fields[name] != fact[name]
        for fact in related for name in ("valid_from", "valid_to")
    ):
        return _result("TEMPORAL_CHANGE", "相关事实的有效区间不同，暂不自动合并或覆盖")

    same_value = [fact for fact in related if _object_key(fields) == _object_key(fact)]
    if same_value and (cardinality == "multi" or len(same_value) == len(related)):
        return _duplicate_result(same_value, sources, user_id, merge_duplicate_sources,
                                 "相同作用域和有效区间内已保存相同的值")
    if cardinality == "multi":
        return _result("MULTIVALUE", "谓词明确允许多值，新增并保留已有事实",
                       decision=add_decision)
    return _result("UNRESOLVED", "存在不同值，缺少明确纠错或多值共存依据")
