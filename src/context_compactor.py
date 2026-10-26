"""用已提交摘要压缩队列前缀，不生成摘要，也不删除原始消息。"""

from copy import deepcopy
import json

from src.context_queue import ContextQueue
from src.error import ContextBudgetExceeded
from src.summary_store import SummaryStore


class ContextCompactor:
    def __init__(self, queue: ContextQueue, summary_store: SummaryStore):
        queue.store.validate_ids(user_id=queue.user_id, session_id=queue.session_id)
        if queue.store.conn is not summary_store.conn:
            raise ValueError("队列和摘要必须复用同一个 MessageStore 连接")
        self.queue = queue
        self.summary_store = summary_store

    def build_context(self, *, max_tokens: int, token_counter, keep_recent: int = 2,
                      summary_id: str | None = None) -> list[dict]:
        """预算超限才压缩；调用方先 refresh，先提交覆盖整个旧前缀的摘要。

        新摘要必须继续覆盖先前摘要的全部范围。失败不改变队列，last_seq
        保留原始消息水位；重启后 refresh 原始历史，仍可复用持久化摘要。
        """
        if type(max_tokens) is not int or max_tokens < 0:
            raise ValueError("max_tokens 必须是非负整数")
        if type(keep_recent) is not int or keep_recent < 1:
            raise ValueError("keep_recent 必须是正整数")
        if not callable(token_counter):
            raise ValueError("token_counter 必须可调用")
        if summary_id is not None:
            self.queue.store.validate_ids(summary_id=summary_id)

        def fits(messages):
            count = token_counter(deepcopy(messages))
            if type(count) is not int or count < 0:
                raise ValueError("token_counter 必须返回非负整数")
            return count <= max_tokens

        messages = self.queue.to_messages()
        if fits(messages):
            return messages
        summaries = self.summary_store.list_summaries(
            user_id=self.queue.user_id, session_id=self.queue.session_id,
        )
        if summary_id is not None:
            summaries = [summary for summary in summaries if summary["summary_id"] == summary_id]
        records = self.queue._records
        prefix_start = None if not records else records[0].get("summary_start_seq", records[0]["seq"])
        for summary in summaries:
            if summary["start_seq"] != prefix_start:
                continue
            if any(record["seq"] > summary["end_seq"] and "summary_start_seq" in record
                   for record in records):
                continue
            tail = [record for record in records if record["seq"] > summary["end_seq"]]
            if len(tail) < keep_recent:
                continue
            payload = {
                "role": "user",
                "content": json.dumps({
                    "type": "history_summary", "summary_id": summary["summary_id"],
                    "session_id": summary["session_id"], "start_seq": summary["start_seq"],
                    "end_seq": summary["end_seq"], "content": summary["content"],
                }, ensure_ascii=False, allow_nan=False),
            }
            compacted = [{
                "seq": summary["end_seq"], "summary_start_seq": summary["start_seq"],
                "payload": payload,
            }, *tail]
            candidate = [record["payload"] for record in compacted]
            if fits(candidate):
                # ContextQueue 的水位仍是原始 seq，只替换已持久化覆盖的前缀。
                self.queue._records = deepcopy(compacted)
                return deepcopy(candidate)
        raise ContextBudgetExceeded("预算超限，尚无能覆盖旧前缀且保留最近原文的合适摘要")
