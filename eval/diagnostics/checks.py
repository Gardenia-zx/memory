"""Offline checks V0-V9 for the lexical LoCoMo baseline.

Every check is read-only: it replays ``HybridRetriever`` on an in-memory
snapshot and never calls a model. Each result carries ``status``, raw
``metrics``, ``per_question`` detail and a ``conclusion`` sentence that states
which implementation change the numbers would justify.
"""

from collections import Counter
from dataclasses import dataclass, field

from src.hybrid_retriever import HybridRetriever
from src.message_store import MessageStore

from . import metrics


@dataclass
class DiagnosticContext:
    dataset: dict
    history: dict
    questions: list
    store: MessageStore
    retriever: HybridRetriever
    records: list
    report: dict | None = None
    cache: dict = field(default_factory=dict)

    def search(self, question, *, top_k):
        return self.retriever.search(
            user_id=self.history["user_id"],
            query=question["question"],
            top_k=top_k,
            mode="lexical",
            as_of=question["as_of"],
        )

    def message_texts(self):
        return {
            (record["session_id"], record["message_id"]):
                record["payload"].get("content", "")
            for record in self.records
        }


def _same_recall(left, right):
    if left is None or right is None:
        return left is right
    return abs(left - right) <= 1e-9


def check_v0(context):
    """Replay the reported 20 questions and verify the numbers bit-for-bit."""
    predictions = {}
    if context.report:
        predictions = {
            row["question_id"]: row for row in context.report.get("predictions", [])
        }
    selected_ids = {question["question_id"] for question in context.questions}
    aggregate_comparable = bool(predictions) and selected_ids == set(predictions)
    rows, mismatches = [], []
    for question in context.questions:
        hits = context.search(question, top_k=5)
        expected = metrics.evidence_keys(question)
        source = metrics.recall_at_k(hits, expected)
        session = metrics.session_recall_at_k(hits, question["evidence_session_ids"])
        row = {
            "question_id": question["question_id"],
            "category": question.get("category"),
            "source_recall_at_k": source,
            "session_recall_at_k": session,
            "sources": [metrics.source_key(hit["source"]) for hit in hits],
            "evidence_count": len(expected),
            "evidence_session_count": len(set(question["evidence_session_ids"])),
            "unresolved_evidence": question.get("unresolved_evidence", []),
        }
        reference = predictions.get(question["question_id"])
        if reference is not None:
            row["reported_source_recall"] = reference.get("source_recall_at_k")
            row["reported_session_recall"] = reference.get("session_recall_at_k")
            if not _same_recall(source, reference.get("source_recall_at_k")):
                mismatches.append({
                    "kind": "source_recall", "question_id": question["question_id"],
                    "computed": source, "reported": reference.get("source_recall_at_k"),
                })
            if not _same_recall(session, reference.get("session_recall_at_k")):
                mismatches.append({
                    "kind": "session_recall", "question_id": question["question_id"],
                    "computed": session, "reported": reference.get("session_recall_at_k"),
                })
            reported_sources = [
                metrics.source_key(item) for item in reference.get("retrieved_sources", [])
            ]
            if reported_sources != row["sources"]:
                mismatches.append({
                    "kind": "sources", "question_id": question["question_id"],
                    "computed": row["sources"], "reported": reported_sources,
                })
        rows.append(row)

    aggregate = {
        "question_count": len(rows),
        "source_recall_at_k": metrics.mean(row["source_recall_at_k"] for row in rows),
        "session_recall_at_k": metrics.mean(row["session_recall_at_k"] for row in rows),
    }
    health = {
        "history_id": context.history["history_id"],
        "history_original_id": context.history.get("original_id"),
        "dataset_sha256": context.dataset.get("dataset_sha256"),
        "question_count": len(rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "mean_evidence_turns": metrics.mean(row["evidence_count"] for row in rows),
        "mean_evidence_sessions": metrics.mean(
            row["evidence_session_count"] for row in rows
        ),
        "aggregate_compared": aggregate_comparable,
        "questions_over_top_k": [
            row["question_id"] for row in rows if row["evidence_count"] > 5
        ],
        "unresolved_question_ids": [
            row["question_id"] for row in rows if row["unresolved_evidence"]
        ],
    }

    if context.report:
        overall = context.report.get("overall", {})
        if aggregate_comparable:
            for field_name, computed in (
                ("source_recall_at_k", aggregate["source_recall_at_k"]),
                ("session_recall_at_k", aggregate["session_recall_at_k"]),
            ):
                if not _same_recall(computed, overall.get(field_name)):
                    mismatches.append({
                        "kind": "aggregate", "field": field_name,
                        "computed": computed, "reported": overall.get(field_name),
                    })
        if context.report.get("dataset_sha256") not in (None, health["dataset_sha256"]):
            mismatches.append({
                "kind": "dataset_sha256",
                "computed": health["dataset_sha256"],
                "reported": context.report.get("dataset_sha256"),
            })
        snapshots = context.report.get("snapshot_sha256") or {}
        if snapshots and context.history["history_id"] not in snapshots:
            mismatches.append({
                "kind": "history_id",
                "computed": context.history["history_id"],
                "reported": list(snapshots),
            })

    if not context.report:
        status = "skipped"
        conclusion = "没有报告可比对；只输出重放指标，无法证明与报告一致。"
    elif mismatches:
        status = "fail"
        conclusion = (
            f"发现 {len(mismatches)} 处不一致；先修正快照/检索差异，再相信后续诊断。"
        )
    elif not aggregate_comparable:
        status = "partial"
        conclusion = (
            "逐题来源一致，但本次题集与报告题集不同，聚合指标未比对；"
            "要用报告阈值下结论，请保持相同 --limit。"
        )
    else:
        status = "pass"
        conclusion = "重放结果与报告逐题一致，后续诊断建立在同一快照与同一检索实现上。"
    return {
        "status": status,
        "metrics": {"aggregate": aggregate, "health": health},
        "per_question": rows,
        "mismatches": mismatches,
        "conclusion": conclusion,
    }


def check_v1(context):
    """Can neighbor expansion recover evidence missed by top-5 messages?"""
    positions = metrics.message_positions(context.records)
    rows = []
    missing_total = 0
    within_one = within_two = other_session = 0
    for question in context.questions:
        hits = context.search(question, top_k=5)
        found = metrics.hit_keys(hits)
        missing = []
        for evidence in question["evidence_sources"]:
            if metrics.source_key(evidence) in found:
                continue
            distance = metrics.neighbor_distance(evidence, hits, positions)
            missing.append({
                "session_id": evidence["session_id"],
                "message_id": evidence["message_id"],
                "distance": distance,
                "reachable_within_2": distance is not None and distance <= 2,
            })
            missing_total += 1
            if distance is None:
                other_session += 1
            elif distance <= 1:
                within_one += 1
                within_two += 1
            elif distance <= 2:
                within_two += 1
        rows.append({
            "question_id": question["question_id"],
            "category": question.get("category"),
            "source_recall_at_k": metrics.recall_at_k(
                hits, metrics.evidence_keys(question)
            ),
            "missing": missing,
        })
    coverage = {
        "missing_evidence_turns": missing_total,
        "within_1": within_one / missing_total if missing_total else None,
        "within_2": within_two / missing_total if missing_total else None,
        "different_session": other_session / missing_total if missing_total else None,
    }
    if coverage["within_1"] is None:
        conclusion = "没有缺失证据，邻域扩展无从验证（当前 top-5 已全覆盖）。"
    elif coverage["within_1"] >= 0.5:
        conclusion = "过半缺失证据落在命中轮的 ±1 内，邻域扩展是低成本高收益改动。"
    elif coverage["within_2"] is not None and coverage["within_2"] >= 0.5:
        conclusion = "多数缺失证据在 ±2 内，需要更宽的窗口或会话块，而非单轮邻域。"
    else:
        conclusion = "缺失证据多在同一会话外，邻域扩展无效，应先解决会话/表征。"
    return {
        "status": "info",
        "metrics": coverage,
        "per_question": rows,
        "conclusion": conclusion,
    }


def check_v2(context):
    """Where does gold evidence rank inside its own session under BM25?"""
    max_k = len(context.records) + 1
    rows = []
    overall_buckets = Counter()
    session_buckets = Counter()
    session_only_failures = 0
    for question in context.questions:
        hits = context.search(question, top_k=max_k)
        ranks = metrics.rank_map(hits)
        per_session = {}
        for rank, hit in enumerate(hits, 1):
            per_session.setdefault(hit["source"]["session_id"], []).append(
                (rank, metrics.source_key(hit["source"]))
            )
        evidence_rows = []
        for evidence in question["evidence_sources"]:
            key = metrics.source_key(evidence)
            overall = ranks.get(key)
            session_rank = None
            if overall is not None:
                for position, (_rank, entry_key) in enumerate(
                        per_session.get(evidence["session_id"], []), 1):
                    if entry_key == key:
                        session_rank = position
                        break
            overall_buckets[metrics.rank_bucket(overall)] += 1
            session_buckets[metrics.rank_bucket(session_rank)] += 1
            evidence_rows.append({
                "session_id": evidence["session_id"],
                "message_id": evidence["message_id"],
                "overall_rank": overall,
                "session_rank": session_rank,
                "overall_bucket": metrics.rank_bucket(overall),
                "session_bucket": metrics.rank_bucket(session_rank),
            })
        quick = context.search(question, top_k=5)
        source = metrics.recall_at_k(quick, metrics.evidence_keys(question))
        session = metrics.session_recall_at_k(
            quick, question["evidence_session_ids"]
        )
        if session == 1.0 and source is not None and session is not None and source < 1.0:
            session_only_failures += 1
        rows.append({
            "question_id": question["question_id"],
            "category": question.get("category"),
            "source_recall_at_k": source,
            "session_recall_at_k": session,
            "session_only_failure": session == 1.0 and source is not None and source < 1.0,
            "evidence": evidence_rows,
        })
    ranked = {
        bucket: count for bucket, count in overall_buckets.items()
        if bucket != "unscored"
    }
    total_ranked = sum(ranked.values())
    shares = {
        bucket: (count / total_ranked if total_ranked else None)
        for bucket, count in ranked.items()
    }
    if not total_ranked:
        conclusion = "没有任何证据轮次进入打分结果；问题不在名次，而在完全没匹配。"
    elif (shares.get("6-20") or 0) >= 0.5:
        conclusion = "多数证据排在第 6-20 名，扩大 top_k 或会话内重排即可显著提升召回。"
    elif (shares.get(">50") or 0) >= 0.5:
        conclusion = "多数证据排在 50 名之外，属于词面表征问题；先做查询扩展或向量召回。"
    else:
        conclusion = "证据名次分散，需结合邻域扩展与扩 k 一起评估。"
    return {
        "status": "info",
        "metrics": {
            "overall_buckets": dict(overall_buckets),
            "session_buckets": dict(session_buckets),
            "ranked_share_by_bucket": shares,
            "session_only_failure_questions": session_only_failures,
        },
        "per_question": rows,
        "conclusion": conclusion,
    }


def check_v3(context):
    """Upper bound of session-first retrieval: rank gold sessions directly."""
    variants = {}
    for with_date, name in ((False, "text_only"), (True, "with_date")):
        session_ids, documents = metrics.session_documents(
            context.records, with_date=with_date
        )
        rows = []
        for question in context.questions:
            scores = metrics.bm25_scores(question["question"], documents)
            order = metrics.ranked_indices(scores)
            ranks = {
                session_ids[index]: position
                for position, index in enumerate(order, 1)
            }
            expected = question["evidence_session_ids"]
            available = [ranks[session_id] for session_id in expected
                         if session_id in ranks]
            rows.append({
                "question_id": question["question_id"],
                "category": question.get("category"),
                "best_evidence_session_rank": min(available) if available else None,
                "coverage_at_5": metrics.coverage_at(ranks, 5, expected),
                "coverage_at_8": metrics.coverage_at(ranks, 8, expected),
                "coverage_at_12": metrics.coverage_at(ranks, 12, expected),
            })
        variants[name] = {
            "session_count": len(session_ids),
            "coverage_at_5": metrics.mean(row["coverage_at_5"] for row in rows),
            "coverage_at_8": metrics.mean(row["coverage_at_8"] for row in rows),
            "coverage_at_12": metrics.mean(row["coverage_at_12"] for row in rows),
            "full_coverage_at_8": sum(
                1 for row in rows if row["coverage_at_8"] == 1.0
            ),
            "per_question": rows,
        }
    baseline = metrics.mean(
        metrics.session_recall_at_k(
            context.search(question, top_k=5), question["evidence_session_ids"]
        )
        for question in context.questions
    )
    with_date = variants["with_date"]
    if with_date["coverage_at_8"] is None:
        conclusion = "没有证据会话可评估。"
    elif with_date["coverage_at_8"] >= 0.85:
        conclusion = (
            "会话级 top-8 可覆盖 ≥85% 证据会话，两级检索（先会话后轮次）值得实现。"
        )
    elif baseline is not None and with_date["coverage_at_8"] >= baseline:
        conclusion = (
            "会话级略优于当前消息级的会话命中，但不足以单独支撑两级检索；"
            "需改进会话文本（日期/实体）或与邻域扩展叠加。"
        )
    else:
        conclusion = (
            "会话级检索反而弱于当前消息级；先修会话文本构造，不要直接做两级检索。"
        )
    return {
        "status": "info",
        "metrics": {
            "baseline_message_session_recall": baseline,
            "variants": variants,
        },
        "per_question": variants["with_date"]["per_question"],
        "conclusion": conclusion,
    }


def check_v4(context):
    """Recall curve over top_k plus the structural ceiling of each question."""
    curve = []
    for top_k in (5, 10, 20, 30, 50):
        rows = []
        for question in context.questions:
            hits = context.search(question, top_k=top_k)
            evidence_count = len(question["evidence_sources"])
            rows.append({
                "question_id": question["question_id"],
                "category": question.get("category"),
                "source_recall_at_k": metrics.recall_at_k(
                    hits, metrics.evidence_keys(question)
                ),
                "session_recall_at_k": metrics.session_recall_at_k(
                    hits, question["evidence_session_ids"]
                ),
                "ceiling": metrics.recall_ceiling(top_k, evidence_count),
                "evidence_count": evidence_count,
            })
        categories = {}
        for category in sorted({row["category"] for row in rows}):
            selected = [row for row in rows if row["category"] == category]
            categories[category] = {
                "question_count": len(selected),
                "source_recall_at_k": metrics.mean(
                    row["source_recall_at_k"] for row in selected
                ),
                "session_recall_at_k": metrics.mean(
                    row["session_recall_at_k"] for row in selected
                ),
            }
        curve.append({
            "top_k": top_k,
            "source_recall_at_k": metrics.mean(
                row["source_recall_at_k"] for row in rows
            ),
            "session_recall_at_k": metrics.mean(
                row["session_recall_at_k"] for row in rows
            ),
            "mean_ceiling": metrics.mean(row["ceiling"] for row in rows),
            "by_category": categories,
            "per_question": rows,
        })
    best = max(
        curve, key=lambda item: item["source_recall_at_k"] or 0
    )
    k5 = next(item for item in curve if item["top_k"] == 5)
    k20 = next(item for item in curve if item["top_k"] == 20)
    gain = (k20["source_recall_at_k"] or 0) - (k5["source_recall_at_k"] or 0)
    conclusion = (
        f"k=20 相对 k=5 的 source recall 提升 {gain:.3f}；"
        f"该曲线下最佳 k={best['top_k']}。"
    )
    return {
        "status": "info",
        "metrics": {"curve": curve},
        "per_question": k5["per_question"],
        "conclusion": conclusion,
    }


def check_v5(context):
    """Structural coverage of multi-hop questions under session/turn budgets."""
    budgets = ((5, 2), (8, 2), (5, 4), (19, 10))
    rows = []
    for question in context.questions:
        if question.get("category") != "1":
            continue
        counts = Counter(
            source["session_id"] for source in question["evidence_sources"]
        )
        total = sum(counts.values())
        row = {
            "question_id": question["question_id"],
            "evidence_turns": total,
            "evidence_sessions": len(counts),
            "per_session_evidence": dict(counts),
        }
        for session_budget, turn_budget in budgets:
            selected = sorted(counts.values(), reverse=True)[:session_budget]
            covered = sum(min(count, turn_budget) for count in selected)
            row[f"coverage_{session_budget}sessions_x{turn_budget}turns"] = (
                covered / total if total else None
            )
        rows.append(row)
    metrics_summary = {
        "question_count": len(rows),
        "mean_evidence_turns": metrics.mean(row["evidence_turns"] for row in rows),
        "mean_evidence_sessions": metrics.mean(
            row["evidence_sessions"] for row in rows
        ),
        "max_evidence_sessions": max(
            (row["evidence_sessions"] for row in rows), default=None
        ),
    }
    for session_budget, turn_budget in budgets:
        key = f"coverage_{session_budget}sessions_x{turn_budget}turns"
        metrics_summary[key] = metrics.mean(row[key] for row in rows)
    max_sessions = metrics_summary["max_evidence_sessions"]
    if max_sessions is None:
        conclusion = "没有多跳题可评估。"
    elif max_sessions >= 3:
        conclusion = (
            f"多跳题最多涉及 {max_sessions} 个会话，单会话优先不够；"
            "需要强制多会话配额（每个会话至少 1 轮）。"
        )
    else:
        conclusion = "多跳题证据集中在 1-2 个会话，会话优先 + 会话内多轮即可覆盖。"
    return {
        "status": "info",
        "metrics": metrics_summary,
        "per_question": rows,
        "conclusion": conclusion,
    }


def check_v6(context):
    """Answer-side anomalies: evidence present but abstained, or answered blind."""
    if not context.report:
        return {"status": "skipped", "metrics": {}, "per_question": [],
                "conclusion": "缺少报告，无法分析实际回答。"}
    predictions = {
        row["question_id"]: row for row in context.report.get("predictions", [])
    }
    texts = context.message_texts()
    rows = []
    focus = None
    for question in context.questions:
        prediction = predictions.get(question["question_id"])
        if prediction is None:
            continue
        hits = context.search(question, top_k=5)
        retrieved_text = "\n".join(hit["text"] for hit in hits)
        evidence_text = "\n".join(
            texts.get((source["session_id"], source["message_id"]), "")
            for source in question["evidence_sources"]
        )
        answers = metrics.gold_answers(question)
        retrieved_support = metrics.answer_support(retrieved_text, answers)
        evidence_support = metrics.answer_support(evidence_text, answers)
        source = metrics.recall_at_k(hits, metrics.evidence_keys(question))
        abstained = metrics.is_abstention(prediction["hypothesis"])
        row = {
            "question_id": question["question_id"],
            "category": question.get("category"),
            "source_recall_at_k": source,
            "abstained": abstained,
            "hypothesis": prediction["hypothesis"],
            "gold_answers": answers,
            "retrieved_support": retrieved_support,
            "evidence_support": evidence_support,
            "diagnostic_f1": round(
                metrics.best_token_f1(prediction["hypothesis"], answers), 4
            ),
        }
        rows.append(row)
        if (
            focus is None and source == 1.0 and abstained
            and evidence_support["supported"]
        ):
            focus = {
                **row,
                "evidence_text": evidence_text,
                "retrieved_text": retrieved_text,
            }
    blind_answers = [
        row for row in rows
        if row["source_recall_at_k"] == 0 and not row["abstained"]
    ]
    evidence_present_but_abstained = [
        row for row in rows if row["source_recall_at_k"] == 1.0 and row["abstained"]
    ]
    metrics_summary = {
        "blind_answer_questions": len(blind_answers),
        "evidence_present_but_abstained": len(evidence_present_but_abstained),
        "abstention_count": sum(1 for row in rows if row["abstained"]),
        "focus_question_id": focus["question_id"] if focus else None,
    }
    if focus is not None:
        conclusion = (
            f"{focus['question_id']} 证据齐全且 gold 可在证据中找到，却仍弃答；"
            "这是生成/上下文打包问题，不是检索问题。"
        )
    elif evidence_present_but_abstained:
        conclusion = (
            "存在证据齐全但弃答的题，但 gold 不在证据文本中；"
            "先确认问题是否可推断，再决定是否改答案侧。"
        )
    else:
        conclusion = "没有“证据齐全却弃答”的样本；答案侧异常主要是无证据作答。"
    return {
        "status": "info",
        "metrics": metrics_summary,
        "per_question": rows,
        "focus": focus,
        "conclusion": conclusion,
    }


def check_v7(context):
    """Are the abstentions justified by the retrieved context?"""
    if not context.report:
        return {"status": "skipped", "metrics": {}, "per_question": [],
                "conclusion": "缺少报告，无法判断弃答是否正确。"}
    predictions = {
        row["question_id"]: row for row in context.report.get("predictions", [])
    }
    rows = []
    supported = 0
    for question in context.questions:
        prediction = predictions.get(question["question_id"])
        if prediction is None or not metrics.is_abstention(prediction["hypothesis"]):
            continue
        hits = context.search(question, top_k=5)
        context_text = "\n".join(hit["text"] for hit in hits)
        answers = metrics.gold_answers(question)
        support = metrics.answer_support(context_text, answers)
        source = metrics.recall_at_k(hits, metrics.evidence_keys(question))
        if support["supported"]:
            supported += 1
        rows.append({
            "question_id": question["question_id"],
            "category": question.get("category"),
            "source_recall_at_k": source,
            "gold_answers": answers,
            "retrieved_support": support,
        })
    metrics_summary = {
        "abstention_count": len(rows),
        "gold_supported_by_context": supported,
        "gold_not_supported": len(rows) - supported,
    }
    if supported >= 2:
        conclusion = (
            f"{supported} 个弃答题的 gold 实际出现在上下文中；先修提示词/证据打包，"
            "不要盲目降低弃答率，其余弃答是正确的。"
        )
    else:
        conclusion = "弃答基本都发生在无证据场景；保持当前弃答策略，专注提升召回。"
    return {
        "status": "info",
        "metrics": metrics_summary,
        "per_question": rows,
        "conclusion": conclusion,
    }


def check_v8(context):
    """Token and cost projection for larger top_k and the full benchmark."""
    lengths = [
        len(metrics.text_tokens(record["payload"].get("content", "")))
        for record in context.records
    ]
    average = metrics.mean(lengths)
    measured = None
    full_questions = len([
        question for history in context.dataset["histories"]
        for question in history["questions"]
    ])
    if context.report:
        usage = context.report.get("overall", {}).get("query_usage") or {}
        counted = context.report.get("overall", {}).get("question_count")
        if usage.get("input_tokens") and counted:
            measured = usage["input_tokens"] / counted
        full_questions = (
            context.report.get("run_scope", {}).get("original_question_count")
            or full_questions
        )
    base = None
    if measured is not None and average is not None:
        base = measured - 5 * average
    estimates = {}
    for top_k in (5, 10, 20, 30):
        if base is None:
            estimates[top_k] = None
            continue
        per_question = base + top_k * average
        estimates[top_k] = {
            "input_tokens_per_question": round(per_question, 1),
            "full_benchmark_input_tokens": round(per_question * full_questions, 1),
        }
    metrics_summary = {
        "record_count": len(context.records),
        "average_message_tokens": round(average, 2) if average is not None else None,
        "measured_input_tokens_per_question": (
            round(measured, 1) if measured is not None else None
        ),
        "estimated_prompt_overhead": round(base, 1) if base is not None else None,
        "full_question_count": full_questions,
        "estimates": estimates,
    }
    conclusion = (
        "估算基于当前实测 k=5 的输入 token 与消息平均长度；"
        "扩 k 的边际成本按线性近似，需在真实全量运行前复核。"
    )
    return {
        "status": "info",
        "metrics": metrics_summary,
        "per_question": [],
        "conclusion": conclusion,
    }


def check_v9(context):
    """Is the 20-question smoke slice representative of the full benchmark?"""
    selected_ids = {question["question_id"] for question in context.questions}
    all_questions = [
        question for history in context.dataset["histories"]
        for question in history["questions"]
    ]
    rest_of_history = [
        question for question in context.history["questions"]
        if question["question_id"] not in selected_ids
    ]

    def summary(questions):
        counts = Counter(question.get("category") for question in questions)
        total = len(questions)
        return {
            "question_count": total,
            "category_counts": dict(counts),
            "category_share": {
                category: round(count / total, 4) if total else None
                for category, count in counts.items()
            },
            "mean_evidence_turns": metrics.mean(
                len(question["evidence_sources"]) for question in questions
            ),
            "mean_evidence_sessions": metrics.mean(
                len(set(question["evidence_session_ids"])) for question in questions
            ),
        }

    metrics_summary = {
        "selected": summary(context.questions),
        "same_history_remaining": summary(rest_of_history),
        "full_dataset": summary(all_questions),
    }
    selected_categories = set(metrics_summary["selected"]["category_counts"])
    missing = sorted({"1", "2", "3", "4", "5"} - selected_categories)
    if missing:
        conclusion = (
            f"当前切片缺少类别 {', '.join(missing)}；结论只能当方向验证，"
            "正式判断前应扩到 100-200 题分层抽样。"
        )
    else:
        conclusion = "切片覆盖了全部 5 个类别，可作为小样本验证。"
    return {
        "status": "info",
        "metrics": metrics_summary,
        "per_question": [],
        "conclusion": conclusion,
    }


def _group_sessions(records):
    order, groups = [], {}
    for record in records:
        session_id = record["session_id"]
        if session_id not in groups:
            groups[session_id] = []
            order.append(session_id)
        groups[session_id].append(record)
    return order, groups


def _prompt_tokens(context):
    """Measured prompt overhead from the report, so token estimates stay honest."""
    if not context.report:
        return 0.0
    usage = context.report.get("overall", {}).get("query_usage") or {}
    counted = context.report.get("overall", {}).get("question_count")
    if not usage.get("input_tokens") or not counted:
        return 0.0
    lengths = [
        len(metrics.text_tokens(record["payload"].get("content", "")))
        for record in context.records
    ]
    average = metrics.mean(lengths)
    if average is None:
        return 0.0
    return max(0.0, usage["input_tokens"] / counted - 5 * average)


def _select_turns(groups, chosen_sessions, strategy, scoped_scores, global_by_key):
    selected = []
    for session_id in chosen_sessions:
        records = groups[session_id]
        if strategy["mode"] == "all":
            selected.extend(records)
            continue
        if strategy["mode"] == "scoped":
            scores = scoped_scores[session_id]
        else:
            scores = [
                global_by_key[(session_id, record["message_id"])]
                for record in records
            ]
        ranked = sorted(
            range(len(records)),
            key=lambda index: (-scores[index], records[index]["seq"]),
        )
        picked = [records[index] for index in ranked[:strategy["limit"]]]
        if strategy.get("neighbors"):
            wanted = set()
            for record in picked:
                wanted.update((record["seq"] - 1, record["seq"], record["seq"] + 1))
            picked = [record for record in records if record["seq"] in wanted]
        selected.extend(picked)
    return selected


def check_v10(context):
    """Simulate session-first retrieval and pick a recall/token operating point.

    Messages later than each question's ``as_of`` are excluded, mirroring the
    retriever's time filter. The flat references can therefore be compared with
    V0 on the same question.
    """
    token_by_key = {
        (record["session_id"], record["message_id"]):
            len(metrics.text_tokens(record["payload"].get("content", "")))
        for record in context.records
    }
    prompt = _prompt_tokens(context)
    total_questions = sum(
        len(history["questions"]) for history in context.dataset["histories"]
    )
    if context.report:
        total_questions = (
            context.report.get("run_scope", {}).get("original_question_count")
            or total_questions
        )

    session_variants = ("sum_all", "top2_sum", "max", "concat")
    session_counts = (4, 6, 8, 10, 12)
    turn_strategies = (
        {"name": "all", "mode": "all"},
        {"name": "scoped_top3", "mode": "scoped", "limit": 3},
        {"name": "scoped_top5", "mode": "scoped", "limit": 5},
        {"name": "scoped_top10", "mode": "scoped", "limit": 10},
        {"name": "global_top5", "mode": "global", "limit": 5},
        {"name": "global_top10", "mode": "global", "limit": 10},
        {"name": "scoped_top5_neighbors", "mode": "scoped", "limit": 5,
         "neighbors": True},
    )

    accumulated = {}
    per_question = {}
    references = []
    for question in context.questions:
        query = question["question"]
        evidence = metrics.evidence_keys(question)
        expected_sessions = question["evidence_session_ids"]
        category = question.get("category")
        visible = [
            record for record in context.records
            if record["occurred_at"] <= question["as_of"]
        ]
        if not visible:
            continue
        session_order, groups = _group_sessions(visible)
        session_index = {
            session_id: index for index, session_id in enumerate(session_order)
        }
        contents = [
            record["payload"].get("content", "") for record in visible
        ]

        global_scores = metrics.bm25_scores(query, contents)
        global_by_key = {
            (record["session_id"], record["message_id"]): global_scores[index]
            for index, record in enumerate(visible)
        }
        scoped_scores = {
            session_id: metrics.bm25_scores(
                query,
                [record["payload"].get("content", "") for record in records],
            )
            for session_id, records in groups.items()
        }
        concat_scores = metrics.bm25_scores(
            query,
            [
                " ".join(
                    record["payload"].get("content", "")
                    for record in groups[session_id]
                )
                for session_id in session_order
            ],
        )
        session_scores = {
            "sum_all": {
                session_id: sum(
                    global_by_key[(session_id, record["message_id"])]
                    for record in records
                )
                for session_id, records in groups.items()
            },
            "top2_sum": {
                session_id: sum(sorted(
                    (
                        global_by_key[(session_id, record["message_id"])]
                        for record in records
                    ),
                    reverse=True,
                )[:2])
                for session_id, records in groups.items()
            },
            "max": {
                session_id: max(
                    global_by_key[(session_id, record["message_id"])]
                    for record in records
                )
                for session_id, records in groups.items()
            },
            "concat": {
                session_id: concat_scores[index]
                for index, session_id in enumerate(session_order)
            },
        }
        ranked_sessions = {
            variant: sorted(
                session_order,
                key=lambda session_id: (
                    -session_scores[variant][session_id],
                    session_index[session_id],
                ),
            )
            for variant in session_variants
        }

        ranked_messages = sorted(
            range(len(contents)),
            key=lambda index: (-global_scores[index], index),
        )
        scored_messages = [
            index for index in ranked_messages if global_scores[index] > 0
        ]
        for top_k in (5, 20, 50):
            chosen = scored_messages[:top_k]
            keys = {
                metrics.source_key(visible[index]) for index in chosen
            }
            references.append({
                "top_k": top_k,
                "source_recall_at_k": len(evidence & keys) / len(evidence)
                if evidence else None,
                "session_recall_at_k": len(
                    set(expected_sessions) & {visible[index]["session_id"]
                                              for index in chosen}
                ) / len(set(expected_sessions)) if expected_sessions else None,
                "input_tokens": prompt + sum(
                    token_by_key[
                        (visible[index]["session_id"],
                         visible[index]["message_id"])
                    ]
                    for index in chosen
                ),
            })

        for variant in session_variants:
            for session_count in session_counts:
                chosen_sessions = ranked_sessions[variant][:session_count]
                for strategy in turn_strategies:
                    name = (
                        f"{variant}_sessions{session_count}_{strategy['name']}"
                    )
                    selected = _select_turns(
                        groups, chosen_sessions, strategy,
                        scoped_scores, global_by_key,
                    )
                    keys = {
                        metrics.source_key(record) for record in selected
                    }
                    source = len(evidence & keys) / len(evidence) if evidence else None
                    session = len(set(expected_sessions) & set(chosen_sessions)) / len(
                        set(expected_sessions)
                    ) if expected_sessions else None
                    tokens = prompt + sum(
                        token_by_key[(record["session_id"], record["message_id"])]
                        for record in selected
                    )
                    entry = accumulated.setdefault(name, {
                        "source": [], "session": [], "messages": [],
                        "tokens": [], "by_category": {},
                    })
                    entry["source"].append(source)
                    entry["session"].append(session)
                    entry["messages"].append(len(selected))
                    entry["tokens"].append(tokens)
                    category_entry = entry["by_category"].setdefault(
                        category, {"source": []}
                    )
                    category_entry["source"].append(source)
                    per_question.setdefault(name, []).append({
                        "question_id": question["question_id"],
                        "category": category,
                        "source_recall_at_k": source,
                        "session_recall_at_k": session,
                        "selected_messages": len(selected),
                        "input_tokens": tokens,
                    })

    configs = []
    for name, entry in accumulated.items():
        configs.append({
            "name": name,
            "kind": "session_first",
            "source_recall_at_k": metrics.mean(entry["source"]),
            "session_recall_at_k": metrics.mean(entry["session"]),
            "mean_selected_messages": metrics.mean(entry["messages"]),
            "mean_input_tokens": metrics.mean(entry["tokens"]),
            "full_benchmark_input_tokens": (
                (metrics.mean(entry["tokens"]) or 0) * total_questions
            ),
            "by_category": {
                category: {
                    "source_recall_at_k": metrics.mean(values["source"]),
                }
                for category, values in entry["by_category"].items()
            },
        })
    reference_rows = []
    for top_k in (5, 20, 50):
        rows = [row for row in references if row["top_k"] == top_k]
        reference_rows.append({
            "name": f"flat_top{top_k}",
            "kind": "flat",
            "source_recall_at_k": metrics.mean(
                row["source_recall_at_k"] for row in rows
            ),
            "session_recall_at_k": metrics.mean(
                row["session_recall_at_k"] for row in rows
            ),
            "mean_input_tokens": metrics.mean(row["input_tokens"] for row in rows),
            "full_benchmark_input_tokens": (
                (metrics.mean(row["input_tokens"] for row in rows) or 0)
                * total_questions
            ),
        })

    combined = sorted(
        configs + reference_rows,
        key=lambda item: (item["mean_input_tokens"] or 0, item["name"]),
    )
    pareto, best_recall = [], -1.0
    for item in combined:
        recall = item["source_recall_at_k"]
        if recall is not None and recall > best_recall + 1e-12:
            pareto.append(item["name"])
            best_recall = recall

    best = max(configs, key=lambda item: item["source_recall_at_k"] or 0)
    k50 = next(item for item in reference_rows if item["name"] == "flat_top50")
    baseline = next(item for item in reference_rows if item["name"] == "flat_top5")
    cost_ratio = (
        (best["mean_input_tokens"] or 0) / (baseline["mean_input_tokens"] or 1)
    )
    best_cat1 = max(
        (
            (item["name"], item["by_category"].get("1", {}).get("source_recall_at_k"))
            for item in configs
        ),
        key=lambda pair: pair[1] or 0,
    )
    if (best["source_recall_at_k"] or 0) >= (k50["source_recall_at_k"] or 0) + 0.05:
        conclusion = (
            f"会话优先最优配置 {best['name']} 达到 recall "
            f"{best['source_recall_at_k']:.3f}，高于 flat k=50 的 "
            f"{k50['source_recall_at_k']:.3f}；结构改动优于单纯扩 k。"
        )
    else:
        conclusion = (
            "会话优先没有明显超过 flat k=50；先怀疑会话选择/打分，"
            "或直接转向向量召回，不要急着上两级检索。"
        )
    conclusion += f" 推荐配置约为 flat k=5 的 {cost_ratio:.1f} 倍输入 token。"
    if best_cat1[1] is not None and best_cat1[1] < 0.2:
        conclusion += f" 多跳最优仍只有 {best_cat1[1]:.3f}，语义召回仍是必需项。"
    return {
        "status": "info",
        "metrics": {
            "recommended": best,
            "recommended_cat1": {
                "name": best_cat1[0], "source_recall_at_k": best_cat1[1],
            },
            "references": reference_rows,
            "pareto": pareto,
            "prompt_tokens": round(prompt, 1),
            "total_questions": total_questions,
            "configs": configs,
        },
        "per_question": per_question.get(best["name"], []),
        "conclusion": conclusion,
    }


CHECKS = {
    "V0": check_v0,
    "V1": check_v1,
    "V2": check_v2,
    "V3": check_v3,
    "V4": check_v4,
    "V5": check_v5,
    "V6": check_v6,
    "V7": check_v7,
    "V8": check_v8,
    "V9": check_v9,
    "V10": check_v10,
}
