"""从真实消息区间生成结构化摘要；模型供应商通过 ModelAdapter 注入。"""

import json

from src.model_adapter import ModelAdapter
from src.summary_store import SummaryStore


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"模型 JSON 包含重复键：{key}")
        result[key] = value
    return result


class SummaryGenerator:
    def __init__(self, summary_store: SummaryStore, adapter: ModelAdapter):
        if not hasattr(adapter, "complete_json"):
            raise ValueError("adapter 必须实现 complete_json")
        self.summary_store = summary_store
        self.message_store = summary_store.message_store
        self.adapter = adapter

    def generate(
        self,
        *,
        user_id: str,
        session_id: str,
        summary_id: str,
        start_seq: int,
        end_seq: int,
    ) -> dict:
        """同一 summary_id 重试直接返回已提交摘要，不重复调用模型。"""
        self.message_store.validate_ids(
            user_id=user_id, session_id=session_id, summary_id=summary_id,
        )
        existing = self.summary_store.get_summary(
            user_id=user_id, session_id=session_id, summary_id=summary_id,
        )
        if existing is not None:
            return existing
        if (type(start_seq) is not int or type(end_seq) is not int
                or start_seq < 1 or end_seq < start_seq):
            raise ValueError("摘要覆盖范围必须为正整数闭区间")
        records = [
            record for record in self.message_store.list_messages(
                user_id=user_id, session_id=session_id, after_seq=start_seq - 1,
            )
            if record["seq"] <= end_seq
        ]
        if [record["seq"] for record in records] != list(range(start_seq, end_seq + 1)):
            raise ValueError("摘要覆盖范围必须包含连续且真实存在的消息")
        response = self.adapter.complete_json(
            task="summarize_history",
            payload={
                "instructions": (
                    "只返回 JSON 对象 {\"summary\":\"...\"}。保留人物、时间、决定、"
                    "未完成事项和不确定性，不添加消息中没有的信息。"
                ),
                "range": {"start_seq": start_seq, "end_seq": end_seq},
                "messages": [record["payload"] for record in records],
            },
        )
        if not isinstance(response, str):
            raise ValueError("模型适配器必须返回 JSON 字符串")
        parsed = json.loads(response, object_pairs_hook=_unique_keys)
        if (not isinstance(parsed, dict) or set(parsed) != {"summary"}
                or not isinstance(parsed["summary"], str) or not parsed["summary"].strip()):
            raise ValueError("模型摘要必须是只含非空 summary 的 JSON 对象")
        return self.summary_store.save_summary(
            user_id=user_id,
            session_id=session_id,
            summary_id=summary_id,
            content=parsed["summary"],
            start_seq=start_seq,
            end_seq=end_seq,
        )
