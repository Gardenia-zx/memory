from src.message_store import MessageStore


class HistoryRetriever:
    def __init__(self, store: MessageStore):
        self.store = store

    def search(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a positive integer")
        records = self.store.list_user_messages(user_id = user_id)
        query_list  = dict.fromkeys(query.casefold().split())
        if len(query_list) == 0 or top_k == 0:
            return []

        messages = []

        for record in records:
            content  = record["payload"].get("content")
            if isinstance(content, str):
                content = content.casefold()
                score = 0
                for query in query_list:
                    if query in content:
                        score += 1
                if score >0:
                    messages.append({
                        "message": record,
                        "score": score,
                    })
        return sorted(messages, key=lambda message: message["score"], reverse=True)[:top_k]

