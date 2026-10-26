from src.history_retriever import HistoryRetriever


def evaluate_retrieval(
    retriever: HistoryRetriever,
    cases: list[dict],
    *,
    top_k: int = 5,
) -> dict:
    """评测固定历史中的来源召回；案例中的正确来源应已由加载阶段校验存在。"""
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k 必须是正整数")

    source_fields = ("user_id", "session_id", "message_id")
    answerable_count = 0
    no_answer_count = 0
    recall_sum = 0.0
    hit_count = 0
    no_answer_empty_count = 0
    details = []

    for case in cases:
        user_id = case["user_id"]
        relevant_sources = [
            {field: source[field] for field in source_fields}
            for source in case["relevant_sources"]
        ]
        if any(source["user_id"] != user_id for source in relevant_sources):
            raise ValueError("正确来源必须属于案例指定的用户")
        expected_keys = {
            tuple(source[field] for field in source_fields)
            for source in relevant_sources
        }

        hits = retriever.search(
            user_id=user_id,
            query=case["query"],
            top_k=top_k,
        )[:top_k]
        retrieved = []
        for hit in hits:
            source = {field: hit["message"][field] for field in source_fields}
            if source["user_id"] != user_id:
                raise ValueError("检索结果包含其他用户的消息")
            retrieved.append({"source": source, "score": hit["score"]})

        # 先取前 k 条，再按完整来源去重；重复结果不能额外得分或补位。
        retrieved_keys = {
            tuple(item["source"][field] for field in source_fields)
            for item in retrieved
        }
        empty_return = not hits
        recall_at_k = None
        hit_at_k = None
        if expected_keys:
            matched_count = len(expected_keys & retrieved_keys)
            recall_at_k = matched_count / len(expected_keys)
            hit_at_k = int(matched_count > 0)
            answerable_count += 1
            recall_sum += recall_at_k
            hit_count += hit_at_k
        else:
            no_answer_count += 1
            no_answer_empty_count += int(empty_return)

        details.append({
            "case_id": case["case_id"],
            "user_id": user_id,
            "query": case["query"],
            "relevant_sources": relevant_sources,
            "retrieved": retrieved,
            "recall_at_k": recall_at_k,
            "hit_at_k": hit_at_k,
            "empty_return": empty_return,
        })

    # 没有该类案例时用 None，避免把“未评测”解释为零分。
    return {
        "top_k": top_k,
        "answerable_count": answerable_count,
        "no_answer_count": no_answer_count,
        "mean_recall_at_k": recall_sum / answerable_count if answerable_count else None,
        "hit_rate_at_k": hit_count / answerable_count if answerable_count else None,
        "no_answer_empty_rate": (
            no_answer_empty_count / no_answer_count if no_answer_count else None
        ),
        "details": details,
    }
