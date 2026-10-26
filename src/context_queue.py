import copy

from src.error import ContextBudgetExceeded
from src.message_store import MessageStore


class ContextQueue:
    def __init__(
        self,
        store: MessageStore,
        *,
        user_id: str,
        session_id: str,
    ):
        self.store = store
        self.user_id = user_id
        self.session_id = session_id

        self._records: list[dict] = []
        self.last_seq = 0

    def refresh(self) -> int:
        """加载新增记录，返回新增数量。"""
        new_records = self.store.list_messages(user_id=self.user_id, session_id=self.session_id,after_seq=self.last_seq)
        if new_records:
            self.last_seq = new_records[-1]["seq"]
            self._records.extend(new_records)
            return len(new_records)
        return 0

    def to_messages(self) -> list[dict]:
        """导出独立的消息列表，仅包含 payload。"""
        messages = []
        for record in self._records:
            messages.append(record["payload"])
        #     深拷贝防止下游调用拿到msg列表之后修改队列record内容
        return copy.deepcopy(messages)

    def build_context(
            self,
            *,
            max_tokens: int,
            token_counter,
    )->list[dict]:
        messages = self.to_messages()
        used_tokens = token_counter(messages)
        if used_tokens > max_tokens:
            raise ContextBudgetExceeded(f"{used_tokens} tokens超过了{max_tokens}最大tokens限额")
        return messages