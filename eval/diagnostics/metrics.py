"""Pure helpers for the offline retrieval diagnostics (no model or network calls)."""

from collections import Counter
from datetime import datetime
import math
import re

from src.hybrid_retriever import tokenize

BM25_K1 = 1.2
BM25_B = 0.75
_WORD = re.compile(r"[\w]+", re.UNICODE)
_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def source_key(source):
    """Stable evidence identity: the same triple used by the benchmark runner."""
    return (source["user_id"], source["session_id"], source["message_id"])


def evidence_keys(question):
    return {source_key(source) for source in question["evidence_sources"]}


def hit_keys(hits):
    return {source_key(hit["source"]) for hit in hits}


def recall_at_k(hits, expected):
    found = hit_keys(hits)
    return len(expected & found) / len(expected) if expected else None


def session_recall_at_k(hits, expected_sessions):
    expected = set(expected_sessions)
    found = {
        hit["source"]["session_id"] for hit in hits
        if hit.get("source", {}).get("type", "message") == "message"
    }
    return len(expected & found) / len(expected) if expected else None


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def bm25_scores(query, documents):
    """Score documents with the same accumulation order as HybridRetriever.

    Keeping k1/b and the loop shape identical means a diagnostic ranking can be
    compared with production retrieval without re-deriving the formula.
    """
    token_counts = [Counter(tokenize(document)) for document in documents]
    if not token_counts:
        return []
    average = sum(sum(count.values()) for count in token_counts) / len(token_counts)
    scores = [0.0] * len(token_counts)
    for word in set(tokenize(query)):
        frequency = sum(word in counts for counts in token_counts)
        inverse = math.log(
            1 + (len(token_counts) - frequency + 0.5) / (frequency + 0.5)
        )
        for index, counts in enumerate(token_counts):
            count = counts[word]
            if count:
                length = sum(counts.values())
                scores[index] += (
                    inverse * count * (BM25_K1 + 1)
                    / (count + BM25_K1 * (1 - BM25_B + BM25_B * length / max(average, 1)))
                )
    return scores


def ranked_indices(scores):
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def rank_map(hits):
    """1-based rank of each hit, matching the retriever's sorted output."""
    return {source_key(hit["source"]): index for index, hit in enumerate(hits, 1)}


def rank_bucket(rank):
    if rank is None:
        return "unscored"
    if rank <= 5:
        return "1-5"
    if rank <= 20:
        return "6-20"
    if rank <= 50:
        return "21-50"
    return ">50"


def message_positions(records):
    return {
        (record["session_id"], record["message_id"]): record["seq"]
        for record in records
    }


def neighbor_distance(evidence, hits, positions):
    """Minimum |seq| distance from an evidence turn to a hit in the same session.

    ``None`` means there is no hit in that session, so neighbor expansion could
    not recover the turn no matter how wide the window is.
    """
    target = positions.get((evidence["session_id"], evidence["message_id"]))
    if target is None:
        return None
    distances = []
    for hit in hits:
        source = hit["source"]
        if source.get("session_id") != evidence["session_id"]:
            continue
        position = positions.get((source.get("session_id"), source.get("message_id")))
        if position is not None:
            distances.append(abs(position - target))
    return min(distances) if distances else None


def human_date(value):
    """Render an ISO timestamp as ``7 May 2023`` for session-level matching."""
    parsed = datetime.fromisoformat(value)
    return f"{parsed.day} {_MONTH_NAMES[parsed.month - 1]} {parsed.year}"


def session_documents(records, *, with_date=False):
    """Group turns into session documents, optionally prefixed by the session date.

    Every turn in one LoCoMo session shares ``occurred_at``, so the first record
    supplies the session date without guessing.
    """
    grouped = {}
    for record in records:
        grouped.setdefault(record["session_id"], []).append(record)
    session_ids, documents = [], []
    for session_id, turns in grouped.items():
        session_ids.append(session_id)
        text = " ".join(turn["payload"].get("content", "") for turn in turns)
        if with_date:
            text = f"{human_date(turns[0]['occurred_at'])} {text}"
        documents.append(text)
    return session_ids, documents


def coverage_at(ranks, cutoff, expected_sessions):
    """Fraction of expected sessions whose rank is within ``cutoff``."""
    expected = set(expected_sessions)
    if not expected:
        return None
    found = sum(
        1 for session_id in expected
        if ranks.get(session_id) is not None and ranks[session_id] <= cutoff
    )
    return found / len(expected)


def normalize_text(text):
    text = str(text).casefold()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def text_tokens(text):
    return _WORD.findall(normalize_text(text))


def gold_answers(question):
    """All acceptable gold strings; LoCoMo answers may be lists or ``a; b``."""
    answer = question.get("answer")
    values = answer if isinstance(answer, list) else [answer]
    result = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if question.get("category") == "3" and ";" in text:
            parts = [part.strip() for part in text.split(";") if part.strip()]
            result.extend(parts)
        else:
            result.append(text)
    return result


def answer_support(context, answers):
    """Heuristic check: does the retrieved text actually contain the gold answer?

    Long answers rarely appear verbatim, so token overlap is reported alongside
    exact containment. This is a diagnostic proxy, not the official metric.
    """
    normalized_context = normalize_text(context)
    context_counts = Counter(text_tokens(context))
    contains, overlap = False, 0.0
    for answer in answers:
        normalized = normalize_text(answer)
        if normalized and normalized in normalized_context:
            contains = True
        gold_counts = Counter(text_tokens(answer))
        total = sum(gold_counts.values())
        if total:
            overlap = max(overlap, sum((gold_counts & context_counts).values()) / total)
    return {
        "contains": contains,
        "token_overlap": round(overlap, 4),
        "supported": contains or overlap >= 0.8,
    }


def token_f1(prediction, gold):
    prediction_counts = Counter(text_tokens(prediction))
    gold_counts = Counter(text_tokens(gold))
    common = sum((prediction_counts & gold_counts).values())
    if not common:
        return 0.0
    precision = common / sum(prediction_counts.values())
    recall = common / sum(gold_counts.values())
    return 2 * precision * recall / (precision + recall)


def best_token_f1(prediction, answers):
    return max((token_f1(prediction, answer) for answer in answers), default=0.0)


def recall_ceiling(top_k, evidence_count):
    """Best possible source recall when only ``top_k`` messages fit."""
    if not evidence_count:
        return None
    return min(top_k, evidence_count) / evidence_count


def is_abstention(hypothesis):
    return "no information available" in str(hypothesis).casefold()
