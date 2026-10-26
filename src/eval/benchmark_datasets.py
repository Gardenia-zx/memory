"""Official V1 LongMemEval/LoCoMo JSON adapters; labels never enter messages.

Schemas: github.com/xiaowu0162/LongMemEval#dataset-format and
github.com/snap-research/locomo#data (checked 2026-10-25).
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


ADAPTER_VERSION = "official-v1-json-2"
_MONTHS = {name: index for index, name in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"), 1)}
_LOCOMO_EVIDENCE_ID = re.compile(r"(?<![A-Za-z0-9])D:?(\d+):(\d+)(?!\d)")


def _identifier(value, field):
    if type(value) is int:
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty identifier")
    return value


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value


def timestamp(value, assumed_timezone="+00:00"):
    """Naive official dates use an explicit recorded offset, never the host timezone."""
    _text(value, "timestamp")
    if not isinstance(assumed_timezone, str) or not re.fullmatch(r"[+-]\d\d:\d\d", assumed_timezone):
        raise ValueError("assumed_timezone must be an explicit offset such as +00:00")
    assumed = datetime.fromisoformat("2000-01-01T00:00:00" + assumed_timezone).tzinfo
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        cleaned = re.sub(r"\s*\([A-Za-z]+\)", "", value).strip()
        parsed = None
        for pattern in ("%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d"):
            try:
                parsed = datetime.strptime(cleaned, pattern)
                break
            except ValueError:
                pass
        if parsed is None:
            match = re.fullmatch(r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", value.strip(), re.IGNORECASE)
            if match is None or match[5].casefold() not in _MONTHS:
                raise ValueError(f"unsupported benchmark timestamp: {value!r}")
            hour = int(match[1])
            if not 1 <= hour <= 12:
                raise ValueError("12-hour timestamps require an hour between 1 and 12")
            parsed = datetime(int(match[6]), _MONTHS[match[5].casefold()], int(match[4]),
                              hour % 12 + (12 if match[3].casefold() == "pm" else 0), int(match[2]))
    return parsed.replace(tzinfo=assumed).astimezone(timezone.utc).isoformat() if parsed.utcoffset() is None else parsed.astimezone(timezone.utc).isoformat()


def _list(value, field):
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _locomo_evidence_ids(value):
    """Extract released dialog IDs, including its compound and typo forms."""
    ids = [f"D{int(match[1])}:{int(match[2])}"
           for match in _LOCOMO_EVIDENCE_ID.finditer(value)]
    remainder = _LOCOMO_EVIDENCE_ID.sub("", value).strip(" \t\r\n;,()")
    return ids, ([value.strip()] if remainder else [])


def _history(benchmark, messages, *, original_id=None):
    if not messages:
        raise ValueError("benchmark histories must contain at least one turn")
    serialized = json.dumps(messages, sort_keys=True, ensure_ascii=False, allow_nan=False)
    history_id = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    user_id = f"benchmark:{benchmark}:{history_id}"
    return {"benchmark": benchmark, "history_id": history_id, "user_id": user_id,
            "original_id": original_id,
            "messages": [{"user_id": user_id, **message} for message in messages], "questions": []}


def adapt_longmemeval(data, *, assumed_timezone="+00:00"):
    grouped, question_ids = {}, set()
    for entry in _list(data, "LongMemEval dataset"):
        if not isinstance(entry, dict):
            raise ValueError("LongMemEval entries must be objects")
        question_id = _identifier(entry.get("question_id"), "question_id")
        if question_id in question_ids:
            raise ValueError(f"duplicate question_id: {question_id}")
        question_ids.add(question_id)
        sessions = _list(entry.get("haystack_sessions"), "haystack_sessions")
        session_ids = _list(entry.get("haystack_session_ids"), "haystack_session_ids")
        dates = _list(entry.get("haystack_dates"), "haystack_dates")
        if not sessions or len(sessions) != len(session_ids) or len(sessions) != len(dates):
            raise ValueError("haystack sessions, IDs and dates must have matching nonzero lengths")
        raw_session_ids = [_identifier(value, "session_id") for value in session_ids]
        raw_id_set = set(raw_session_ids)
        session_ids_by_raw, normalized_ids, used_ids = {}, [], set()
        for index, raw_id in enumerate(raw_session_ids):
            session_id = raw_id
            if session_id in used_ids:
                session_id = f"{raw_id}#occurrence-{index + 1}"
                while session_id in used_ids or session_id in raw_id_set:
                    session_id += "#"
            used_ids.add(session_id)
            normalized_ids.append(session_id)
            session_ids_by_raw.setdefault(raw_id, []).append(session_id)
        dated = [(timestamp(date, assumed_timezone), index) for index, date in enumerate(dates)]
        messages, evidence, marked_evidence_sessions = [], [], set()
        for occurred, index in sorted(dated, key=lambda item: (item[0], item[1])):
            session_id = normalized_ids[index]
            for position, turn in enumerate(_list(sessions[index], "haystack session"), 1):
                if not isinstance(turn, dict) or turn.get("role") not in ("user", "assistant") or not isinstance(turn.get("content"), str):
                    raise ValueError("history turns require user/assistant role and text content")
                if "has_answer" in turn and type(turn["has_answer"]) is not bool:
                    raise ValueError("has_answer must be boolean when present")
                message_id = f"turn:{position}"
                messages.append({"session_id": session_id, "message_id": message_id,
                                 "occurred_at": occurred,
                                 "payload": {"role": turn["role"], "content": turn["content"]}})
                if turn.get("has_answer", False):
                    evidence.append({"session_id": session_id, "message_id": message_id})
                    marked_evidence_sessions.add(session_id)
        history = _history("longmemeval", messages)
        history = grouped.setdefault(history["history_id"], history)
        answer_sessions = [_identifier(value, "answer_session_id") for value in _list(entry.get("answer_session_ids", []), "answer_session_ids")]
        if any(value not in session_ids_by_raw for value in answer_sessions):
            raise ValueError("answer_session_ids must refer to history sessions")
        evidence_session_ids = []
        for raw_id in answer_sessions:
            candidates = session_ids_by_raw[raw_id]
            marked = [session_id for session_id in candidates
                      if session_id in marked_evidence_sessions]
            evidence_session_ids.extend(marked or candidates)
        if "answer" not in entry:
            raise ValueError("answer is required in the scoring layer")
        answer = json.loads(json.dumps(entry["answer"], allow_nan=False))
        history["questions"].append({
            "question_id": question_id, "question": _text(entry.get("question"), "question"),
            "category": _text(entry.get("question_type"), "question_type"),
            "as_of": timestamp(entry.get("question_date"), assumed_timezone),
            "answer": answer,
            "evidence_sources": [{"type": "message", "user_id": history["user_id"], **ref} for ref in evidence],
            "evidence_session_ids": list(dict.fromkeys(evidence_session_ids)),
            "answerable": not question_id.endswith("_abs"),
            "official": {"question_id": question_id, "question_type": entry["question_type"],
                         "question": entry["question"], "answer": answer},
        })
    return list(grouped.values())


def adapt_locomo(data, *, assumed_timezone="+00:00"):
    histories, sample_ids = [], set()
    for sample in _list(data, "LoCoMo dataset"):
        if not isinstance(sample, dict) or not isinstance(sample.get("conversation"), dict):
            raise ValueError("LoCoMo conversation must be an object keyed by session_N")
        sample_id = _identifier(sample.get("sample_id"), "sample_id")
        if sample_id in sample_ids:
            raise ValueError("duplicate LoCoMo sample_id")
        sample_ids.add(sample_id)
        conversation = sample["conversation"]
        _text(conversation.get("speaker_a"), "speaker_a")
        _text(conversation.get("speaker_b"), "speaker_b")
        session_keys = sorted((key for key in conversation if re.fullmatch(r"session_\d+", key)), key=lambda key: int(key.split("_")[1]))
        if not session_keys:
            raise ValueError("LoCoMo conversation has no session_N entries")
        messages, refs = [], {}
        for session_id in session_keys:
            occurred = timestamp(conversation.get(session_id + "_date_time"), assumed_timezone)
            for turn in _list(conversation[session_id], session_id):
                if not isinstance(turn, dict):
                    raise ValueError("LoCoMo turns must be objects")
                message_id = _identifier(turn.get("dia_id"), "dia_id")
                if message_id in refs:
                    raise ValueError("dia_id must be unique within a conversation")
                speaker = _text(turn.get("speaker"), "speaker")
                if speaker not in (conversation["speaker_a"], conversation["speaker_b"]):
                    raise ValueError("turn speaker must be one of the two conversation speakers")
                if not isinstance(turn.get("text"), str):
                    raise ValueError("LoCoMo text must be a string")
                # Both speakers are historical people, not this assistant's own utterances.
                payload = {"role": "user", "content": speaker + ": " + turn["text"], "speaker": speaker}
                for field in ("img_url", "blip_caption", "query"):
                    if field in turn:
                        payload[field] = deepcopy(turn[field])
                if isinstance(turn.get("blip_caption"), str):
                    payload["content"] += "\n[Image caption] " + turn["blip_caption"]
                messages.append({"session_id": session_id, "message_id": message_id,
                                 "occurred_at": occurred, "payload": payload})
                refs[message_id] = {"session_id": session_id, "message_id": message_id}
        history = _history("locomo", messages, original_id=sample["sample_id"])
        for index, qa in enumerate(_list(sample.get("qa"), "qa")):
            if not isinstance(qa, dict) or type(qa.get("category")) is not int or qa["category"] not in (1, 2, 3, 4, 5):
                raise ValueError("LoCoMo QA category must be an integer from 1 to 5")
            evidence, unresolved_evidence = [], []
            for raw_ref in _list(qa.get("evidence", []), "evidence"):
                if not isinstance(raw_ref, str):
                    raise ValueError("evidence IDs must be strings")
                keys, malformed = _locomo_evidence_ids(raw_ref)
                unresolved_evidence.extend(malformed)
                for key in keys:
                    if key not in refs:
                        unresolved_evidence.append(key)
                        continue
                    evidence.append({"type": "message", "user_id": history["user_id"], **refs[key]})
            if "answer" not in qa and not (qa["category"] == 5 and "adversarial_answer" in qa):
                raise ValueError("QA requires answer, or adversarial_answer for category 5")
            official = json.loads(json.dumps(qa, ensure_ascii=False, allow_nan=False))
            history["questions"].append({
                "question_id": f"locomo:{sample_id}:{index}", "question": _text(qa.get("question"), "question"),
                "category": str(qa["category"]), "as_of": max(message["occurred_at"] for message in messages),
                # adversarial_answer is a distractor, never the expected answer or model input.
                "answer": deepcopy(qa.get("answer")), "answerable": qa["category"] != 5,
                "evidence_sources": evidence,
                "evidence_session_ids": list(dict.fromkeys(ref["session_id"] for ref in evidence)),
                # The release contains a few dangling labels; keep them visible instead of
                # inventing message sources or rejecting the otherwise usable conversation.
                "unresolved_evidence": list(dict.fromkeys(unresolved_evidence)),
                "official": official, "official_index": index,
            })
        histories.append(history)
    return histories


def load_benchmark_dataset(path, *, benchmark, assumed_timezone="+00:00"):
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("dataset_path is empty; provide an official local JSON file")
    raw = Path(path).read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    adapter = {"longmemeval": adapt_longmemeval, "locomo": adapt_locomo}.get(benchmark)
    if adapter is None:
        raise ValueError("benchmark must be longmemeval or locomo")
    histories = adapter(data, assumed_timezone=assumed_timezone)
    if not histories or not any(history["questions"] for history in histories):
        raise ValueError("benchmark dataset contains no questions")
    return {"benchmark": benchmark, "adapter_version": ADAPTER_VERSION,
            "dataset_sha256": hashlib.sha256(raw).hexdigest(),
            "assumed_timezone": assumed_timezone, "histories": histories}
