import json
from src.context_queue import ContextQueue
from src.error import ContextBudgetExceeded
from src.fact_store import FactStore


class ContextBuilder:
    def __init__(self, queue: ContextQueue, fact_store: FactStore):
        self.queue = queue
        self.fact_store = fact_store

    def build_context(
        self,
        *,
        system_prompt: str,
        max_tokens: int,
        token_counter,
        as_of: str | None = None,
    ) -> list[dict]:

        if not isinstance(system_prompt, str):
            raise ValueError("system_prompt 必须是字符串")

        if type(max_tokens) is not int or max_tokens < 0:
            raise ValueError("max_tokens 必须是非负整数")

        if not callable(token_counter):
            raise ValueError("token_counter 必须可调用")
        facts =  self.fact_store.list_facts(
            user_id=self.queue.user_id,
            session_id=self.queue.session_id,
            as_of=as_of,
        )
        messages = self.queue.to_messages()
        context_queue = [
            {"role": "system", "content": system_prompt},
        ]

        if facts:
            context_queue.append({
                "role": "user",
                "content": json.dumps(
                    {"type": "memory_context", "facts": facts},
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            })

        context_queue.extend(messages)

        used_tokens = token_counter(context_queue)

        if type(used_tokens) is not int or used_tokens < 0:
            raise ValueError("token_counter必须返回非负整数")

        if used_tokens > max_tokens:
            raise ContextBudgetExceeded(f"{used_tokens} tokens超过了{max_tokens}最大tokens限额")

        return context_queue
