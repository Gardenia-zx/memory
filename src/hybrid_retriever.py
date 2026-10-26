"""Scoped lexical/vector retrieval; hash embeddings are a deterministic test baseline."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import math
import re

from src.external_document_store import _time
from src.message_store import MessageStore


def tokenize(text):
    tokens = []
    for part in re.findall(r"[\u3400-\u9fff]+|[^\W_]+", text.casefold()):
        if all("\u3400" <= char <= "\u9fff" for char in part):
            tokens.extend(part)
            tokens.extend(part[index:index + 2] for index in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


class HashEmbedding:
    """Lexical hashing for offline tests, not a semantic embedding model."""
    def __init__(self, dimensions=128):
        if type(dimensions) is not int or dimensions < 1:
            raise ValueError("dimensions must be a positive integer")
        self.dimensions = dimensions

    def __call__(self, text):
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:8], "big") % self.dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


def _vector(value):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("embedding must return a nonempty numeric vector")
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        raise ValueError("embedding values must be finite numbers")
    return value


class HybridRetriever:
    def __init__(self, message_store, document_store=None, *, embedding=None):
        self.message_store = message_store
        self.document_store = document_store
        if embedding is not None and not callable(embedding):
            raise ValueError("embedding must be callable")
        self.embedding = embedding

    def search(self, *, user_id, query, top_k=5, session_id=None, mode="lexical", as_of=None):
        MessageStore.validate_ids(user_id=user_id)
        self.message_store.assert_user_active(user_id=user_id)
        if session_id is not None:
            MessageStore.validate_ids(session_id=session_id)
        if not isinstance(query, str):
            raise ValueError("query must be text")
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if mode not in ("lexical", "vector", "hybrid"):
            raise ValueError("mode must be lexical, vector or hybrid")
        moment = _time(as_of, "as_of")
        if not tokenize(query) or top_k == 0:
            return []
        if mode != "lexical" and self.embedding is None:
            raise ValueError("vector/hybrid retrieval requires an embedding function")
        candidates = []
        for record in self.message_store.list_user_messages(user_id=user_id):
            if session_id is not None and record["session_id"] != session_id:
                continue
            if moment is not None and datetime.fromisoformat(record["occurred_at"]) > datetime.fromisoformat(moment):
                continue
            text = record["payload"].get("content")
            if not isinstance(text, str) or not text:
                continue
            candidates.append({
                "type": "message", "text": text,
                "source": {"type": "message", **{name: record[name] for name in ("user_id", "session_id", "message_id")}},
                "metadata": {"occurred_at": record["occurred_at"], "role": record["payload"].get("role")},
            })
        if self.document_store is not None:
            for chunk in self.document_store.list_chunks(user_id=user_id, session_id=session_id, as_of=as_of):
                candidates.append({
                    "type": "document_chunk", "text": chunk["text"],
                    "source": {"type": "document_chunk", **{name: chunk[name] for name in ("user_id", "document_id", "version", "chunk_id")}},
                    "metadata": {**chunk["metadata"], "start_offset": chunk["start_offset"],
                                 "end_offset": chunk["end_offset"], "scope": chunk["scope"], "session_id": chunk["session_id"]},
                })
        if not candidates:
            return []
        lexical, vector = {}, {}
        if mode != "vector":
            documents = [Counter(tokenize(item["text"])) for item in candidates]
            average = sum(sum(count.values()) for count in documents) / len(documents)
            for word in set(tokenize(query)):
                frequency = sum(word in doc for doc in documents)
                inverse = math.log(1 + (len(documents) - frequency + 0.5) / (frequency + 0.5))
                for index, document in enumerate(documents):
                    count = document[word]
                    if count:
                        length = sum(document.values())
                        score = inverse * count * 2.2 / (count + 1.2 * (0.25 + 0.75 * length / max(average, 1)))
                        lexical[index] = lexical.get(index, 0) + score
        if mode != "lexical":
            query_vector = _vector(self.embedding(query))
            query_norm = math.sqrt(sum(value * value for value in query_vector))
            for index, candidate in enumerate(candidates):
                embedding = _vector(self.embedding(candidate["text"]))
                if len(embedding) != len(query_vector):
                    raise ValueError("query and document embedding dimensions differ")
                norm = math.sqrt(sum(value * value for value in embedding)) * query_norm
                score = sum(left * right for left, right in zip(query_vector, embedding)) / norm if norm else 0.0
                if score > 0:
                    vector[index] = score
        scores = lexical if mode == "lexical" else vector
        if mode == "hybrid":
            scores = {}
            for ranking in (lexical, vector):
                for rank, index in enumerate(sorted(ranking, key=lambda key: (-ranking[key], key)), 1):
                    scores[index] = scores.get(index, 0) + 1 / (60 + rank)
        return [{**candidates[index], "score": scores[index],
                 "lexical_score": lexical.get(index), "vector_score": vector.get(index)}
                for index in sorted(scores, key=lambda key: (-scores[key], key))[:top_k]]
