"""Session-first lexical retrieval: rank sessions, then return their turns.

The measured LoCoMo failure mode was global ranking competition: evidence turns
often rank beyond 50 among all messages, but inside their own session they rank
within the top 20. This retriever therefore scores each session by its best
matching message, keeps the top sessions and returns their turns.

It is deliberately separate from ``HybridRetriever``: the existing lexical /
vector / hybrid contract and its tests stay untouched.
"""

from collections import Counter
from datetime import datetime
import math

from src.external_document_store import _time
from src.hybrid_retriever import tokenize
from src.message_store import MessageStore

BM25_K1 = 1.2
BM25_B = 0.75


def _bm25_scores(query, documents):
    """Same BM25 accumulation as ``HybridRetriever`` so scores stay comparable."""
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


class SessionFirstRetriever:
    """Two-level lexical retrieval with an explicit, reportable operating point.

    ``session_count``        how many sessions are kept after session ranking.
    ``turns_per_session``    ``None`` keeps every turn of a selected session;
                             an integer keeps the best N turns per session.
    ``neighbors``            with ``turns_per_session``, also keep ±1 adjacent
                             turns of each selected turn inside the session.
    """

    def __init__(self, message_store, *, session_count=6,
                 turns_per_session=None, neighbors=False):
        if type(session_count) is not int or session_count < 1:
            raise ValueError("session_count must be a positive integer")
        if turns_per_session is not None and (
            type(turns_per_session) is not int or turns_per_session < 1
        ):
            raise ValueError("turns_per_session must be None or a positive integer")
        if type(neighbors) is not bool:
            raise ValueError("neighbors must be boolean")
        self.message_store = message_store
        self.session_count = session_count
        self.turns_per_session = turns_per_session
        self.neighbors = neighbors

    def search(self, *, user_id, query, top_k=None, session_id=None, as_of=None):
        """Return typed message hits, best session first and turns in seq order.

        ``top_k=None`` returns every selected turn; an integer caps the result
        for callers that still want a flat limit.
        """
        MessageStore.validate_ids(user_id=user_id)
        self.message_store.assert_user_active(user_id=user_id)
        if session_id is not None:
            MessageStore.validate_ids(session_id=session_id)
        if not isinstance(query, str):
            raise ValueError("query must be text")
        if top_k is not None and (type(top_k) is not int or top_k < 0):
            raise ValueError("top_k must be None or a nonnegative integer")
        moment = _time(as_of, "as_of")
        if not tokenize(query) or top_k == 0:
            return []

        records = []
        for record in self.message_store.list_user_messages(user_id=user_id):
            if session_id is not None and record["session_id"] != session_id:
                continue
            if moment is not None and datetime.fromisoformat(
                    record["occurred_at"]) > datetime.fromisoformat(moment):
                continue
            text = record["payload"].get("content")
            if not isinstance(text, str) or not text:
                continue
            records.append(record)
        if not records:
            return []

        groups, session_order = {}, []
        for record in records:
            session = record["session_id"]
            if session not in groups:
                groups[session] = []
                session_order.append(session)
            groups[session].append(record)
        order_index = {
            session: index for index, session in enumerate(session_order)
        }

        scores = _bm25_scores(
            query, [record["payload"]["content"] for record in records]
        )
        index_by_key = {
            (record["session_id"], record["message_id"]): index
            for index, record in enumerate(records)
        }
        session_scores = {
            session: max(
                scores[index_by_key[(session, record["message_id"])]]
                for record in turns
            )
            for session, turns in groups.items()
        }
        ranked_sessions = sorted(
            (session for session in session_order if session_scores[session] > 0),
            key=lambda session: (-session_scores[session], order_index[session]),
        )[:self.session_count]
        if not ranked_sessions:
            return []

        hits = []
        for session in ranked_sessions:
            turns = groups[session]
            selected = turns
            if self.turns_per_session is not None:
                scoped = _bm25_scores(
                    query, [record["payload"]["content"] for record in turns]
                )
                ranked_turns = sorted(
                    range(len(turns)),
                    key=lambda index: (-scoped[index], turns[index]["seq"]),
                )
                selected = [turns[index] for index in ranked_turns[:self.turns_per_session]]
                if self.neighbors:
                    wanted = set()
                    for record in selected:
                        wanted.update(
                            (record["seq"] - 1, record["seq"], record["seq"] + 1)
                        )
                    selected = [record for record in turns if record["seq"] in wanted]
            for record in sorted(selected, key=lambda item: item["seq"]):
                payload = record["payload"]
                hits.append({
                    "type": "message",
                    "text": payload["content"],
                    "source": {
                        "type": "message",
                        "user_id": record["user_id"],
                        "session_id": record["session_id"],
                        "message_id": record["message_id"],
                    },
                    "metadata": {
                        "occurred_at": record["occurred_at"],
                        "role": payload.get("role"),
                    },
                    "score": scores[
                        index_by_key[(session, record["message_id"])]
                    ],
                    "session_score": session_scores[session],
                })
        return hits if top_k is None else hits[:top_k]
