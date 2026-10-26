"""Label-isolated benchmark orchestration and exports; no built-in accuracy claim."""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import time

from src.message_store import MessageStore


def _usage(value):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("usage must be an object")
    result = {}
    for name in ("input_tokens", "output_tokens", "cost_usd"):
        number = value.get(name)
        if number is not None:
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise ValueError("usage values must be finite nonnegative numbers")
            if name != "cost_usd" and type(number) is not int:
                raise ValueError("token counts must be integers")
            result[name] = number
    return result


def _sum_usage(values):
    return {key: sum(value[key] for value in values if key in value)
            if values and all(key in value for value in values) else None
            for key in ("input_tokens", "output_tokens", "cost_usd")}


def _source_key(source):
    if not isinstance(source, dict):
        raise ValueError("retrieved sources must be objects")
    kind = source.get("type", "message")
    if kind == "message":
        names = ("user_id", "session_id", "message_id")
    elif kind == "document_chunk":
        names = ("user_id", "document_id", "version", "chunk_id")
    else:
        raise ValueError("unknown retrieved source type")
    if any(name not in source for name in names):
        raise ValueError("retrieved source is incomplete")
    return (kind, *(source[name] for name in names))


def _summary(rows):
    def mean(name):
        values = [row[name] for row in rows if row[name] is not None]
        return sum(values) / len(values) if values else None
    return {"question_count": len(rows), "scored_answer_count": sum(row["answer_score"] is not None for row in rows),
            "answer_score": mean("answer_score"),
            "source_recall_at_k": mean("source_recall_at_k"),
            "session_recall_at_k": mean("session_recall_at_k"),
            "query_usage": _sum_usage([row["usage"] for row in rows])}


def _transient_error(error):
    """Walk the cause chain; adapter wraps socket timeouts in ModelRequestError."""
    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, (TimeoutError, ConnectionError, OSError)):
            return True
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    return False


def _call_with_retries(call, attempts, delay):
    for attempt in range(attempts):
        try:
            return call()
        except Exception as error:
            if attempt + 1 >= attempts or not _transient_error(error):
                raise
            time.sleep(delay * (2 ** attempt))


def _checkpoint_fingerprint(dataset, top_k, source_limit, config):
    payload = {
        "benchmark": dataset.get("benchmark"),
        "dataset_sha256": dataset.get("dataset_sha256"),
        "top_k": top_k,
        "source_limit": source_limit,
        "run_config": config,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_checkpoint(path, fingerprint):
    """Completed outputs keyed by question_id; ignores stale or partial rows."""
    completed = {}
    target = Path(path)
    if not target.is_file():
        return completed
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("fingerprint") != fingerprint:
            continue
        question_id, output = row.get("question_id"), row.get("output")
        if isinstance(question_id, str) and isinstance(output, dict):
            completed[question_id] = output
    return completed


def _append_checkpoint(path, fingerprint, question_id, output):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "fingerprint": fingerprint,
        "question_id": question_id,
        "output": output,
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_benchmark(dataset, *, answer_question, build_history=None, score_answer=None,
                  top_k=5, run_config=None, source_limit=None,
                  answer_attempts=1, retry_delay_seconds=2.0,
                  checkpoint_path=None):
    """Build each history once; each query receives a fresh read-only connection.

    build_history(store, messages) must finish pending jobs before it returns and may
    return build usage. answer_question(connection, public_question) returns text or
    {answer, retrieved_sources, usage}. Only score_answer(labelled_question, answer)
    receives labels and may return a numeric score. No query or grader is ingested.

    ``source_limit`` caps retrieved sources for scoring; ``None`` trusts the answer
    callback, which is required by session-first retrieval that returns many turns.

    ``checkpoint_path`` appends one validated answer per line; a rerun with the same
    dataset/config reuses completed questions instead of paying for them again.
    """
    if not isinstance(dataset, dict) or not isinstance(dataset.get("histories"), list):
        raise ValueError("dataset must be produced by a benchmark adapter")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if source_limit is not None and (
            type(source_limit) is not int or source_limit < 1):
        raise ValueError("source_limit must be None or a positive integer")
    if type(answer_attempts) is not int or answer_attempts < 1:
        raise ValueError("answer_attempts must be a positive integer")
    if type(retry_delay_seconds) not in (int, float) or not math.isfinite(
            retry_delay_seconds) or retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be a nonnegative number")
    if checkpoint_path is not None and not isinstance(checkpoint_path, (str, Path)):
        raise ValueError("checkpoint_path must be None or a filesystem path")
    if not callable(answer_question) or (build_history is not None and not callable(build_history)):
        raise ValueError("answer_question/build_history must be callable")
    if score_answer is not None and not callable(score_answer):
        raise ValueError("score_answer must be callable")
    config = {} if run_config is None else deepcopy(run_config)
    if not isinstance(config, dict):
        raise ValueError("run_config must be an object")
    # Configuration artifacts must not persist credentials when users fill the key later.
    def is_sensitive_key(key):
        normalized = str(key).casefold().replace("-", "_")
        return (
            any(marker in normalized for marker in ("api_key", "authorization", "secret"))
            or normalized == "token"
            or normalized.endswith("_token")
        )

    def redact(value):
        if isinstance(value, dict):
            return {key: "[REDACTED]" if is_sensitive_key(key) and item else redact(item)
                    for key, item in value.items()}
        return [redact(item) for item in value] if isinstance(value, list) else value
    config = redact(config)
    json.dumps(config, allow_nan=False)
    fingerprint = _checkpoint_fingerprint(dataset, top_k, source_limit, config)
    cached = (
        _load_checkpoint(checkpoint_path, fingerprint)
        if checkpoint_path is not None else {}
    )
    reused_answers = 0
    rows, build_costs, snapshot_hashes, seen_questions = [], [], {}, set()
    with TemporaryDirectory(prefix="memory-benchmark-") as folder:
        for index, history in enumerate(dataset["histories"]):
            path = Path(folder) / f"history-{index}.sqlite"
            store = MessageStore(path)
            try:
                messages = deepcopy(history["messages"])
                if any(message.get("user_id") != history["user_id"] for message in messages):
                    raise ValueError("benchmark history contains another user's messages")
                if build_history is None:
                    for message in messages:
                        store.append_message(**message)
                    build_costs.append({})
                else:
                    build_costs.append(_usage(build_history(store, messages)))
                store.conn.commit()
            finally:
                store.close()
            snapshot_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            snapshot_hashes[history["history_id"]] = snapshot_hash
            for question in history["questions"]:
                qid = question["question_id"]
                if qid in seen_questions:
                    raise ValueError("benchmark question IDs must be globally unique")
                seen_questions.add(qid)
                public = {"request_id": hashlib.sha256(qid.encode("utf-8")).hexdigest(), "question": question["question"],
                          "user_id": history["user_id"], "as_of": question["as_of"]}
                output = cached.get(qid)
                reused = output is not None
                if reused:
                    reused_answers += 1
                else:
                    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
                    connection.row_factory = sqlite3.Row
                    try:
                        connection.execute("PRAGMA query_only=ON")
                        output = _call_with_retries(
                            lambda: answer_question(connection, deepcopy(public)),
                            answer_attempts, retry_delay_seconds,
                        )
                    finally:
                        connection.close()
                if hashlib.sha256(path.read_bytes()).hexdigest() != snapshot_hash:
                    raise ValueError("answering modified the fixed history snapshot")
                if isinstance(output, str):
                    output = {"answer": output}
                if not isinstance(output, dict) or not isinstance(output.get("answer"), str):
                    raise ValueError("answer callback must return text or an object with answer text")
                sources = output.get("retrieved_sources", [])
                if not isinstance(sources, list):
                    raise ValueError("retrieved_sources must be a list")
                sources = deepcopy(sources if source_limit is None else sources[:source_limit])
                for source in sources:
                    _source_key(source)
                    if source.get("user_id") != history["user_id"]:
                        raise ValueError("retrieval crossed benchmark users")
                retrieved = {_source_key(source) for source in sources}
                expected = {_source_key(source) for source in question["evidence_sources"]}
                expected_sessions = set(question["evidence_session_ids"])
                found_sessions = {source["session_id"] for source in sources if source.get("type", "message") == "message"}
                score = None if score_answer is None else score_answer(deepcopy(question), output["answer"])
                if score is not None and (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError("answer score must be None or a finite number in [0,1]")
                rows.append({"benchmark": history["benchmark"], "history_id": history["history_id"],
                             "question_id": qid, "category": question["category"],
                             "hypothesis": output["answer"], "retrieved_sources": sources,
                             "source_recall_at_k": len(expected & retrieved) / len(expected) if expected else None,
                             "session_recall_at_k": len(expected_sessions & found_sessions) / len(expected_sessions) if expected_sessions else None,
                             "answer_score": score, "usage": _usage(output.get("usage"))})
                if not reused and checkpoint_path is not None:
                    _append_checkpoint(
                        checkpoint_path, fingerprint, qid,
                        {
                            "answer": output["answer"],
                            "retrieved_sources": sources,
                            "usage": _usage(output.get("usage")),
                        },
                    )
    categories = sorted({row["category"] for row in rows})
    return {"benchmark": dataset.get("benchmark"), "adapter_version": dataset.get("adapter_version"),
            "dataset_sha256": dataset.get("dataset_sha256"), "assumed_timezone": dataset.get("assumed_timezone"),
            "top_k": top_k, "run_config": config, "snapshot_sha256": snapshot_hashes,
            "answer_metric": "external_callback" if score_answer else "not_run",
            "ingestion_mode": "custom_memory_pipeline" if build_history else "raw_history_baseline",
            "overall": _summary(rows),
            "by_category": {category: _summary([row for row in rows if row["category"] == category]) for category in categories},
            "build_usage": _sum_usage(build_costs), "predictions": rows,
            "answer_attempts": answer_attempts,
            "checkpoint_path": None if checkpoint_path is None else str(Path(checkpoint_path)),
            "reused_answers": reused_answers}


def _save_json(path, value, *, lines=False):
    encoded = "\n".join(json.dumps(item, ensure_ascii=False, allow_nan=False) for item in value) + "\n" if lines else json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(target)


def export_predictions_jsonl(report, path):
    _save_json(path, [{"question_id": row["question_id"], "hypothesis": row["hypothesis"]}
                     for row in report["predictions"]], lines=True)


def save_benchmark_report(report, path):
    """保存完整配置、分类指标、证据召回与成本；严格 JSON，失败前不改目标。"""
    _save_json(path, report)


def export_locomo_predictions(dataset, report, path, *, prediction_key="memory_prediction"):
    if dataset.get("benchmark") != "locomo" or not isinstance(prediction_key, str) or not prediction_key.endswith("_prediction"):
        raise ValueError("LoCoMo export requires a locomo dataset and a *_prediction key")
    rows = {row["question_id"]: row for row in report["predictions"]}
    if len(rows) != len(report["predictions"]):
        raise ValueError("duplicate prediction IDs")
    samples = []
    for history in dataset["histories"]:
        qas = []
        for question in history["questions"]:
            if question["question_id"] not in rows:
                raise ValueError("missing LoCoMo prediction")
            row = rows.pop(question["question_id"])
            qa = deepcopy(question["official"])
            qa.setdefault("answer", "")
            qa.setdefault("evidence", [])
            qa[prediction_key] = row["hypothesis"]
            qa[prediction_key + "_context"] = [source["message_id"] for source in row["retrieved_sources"]
                                              if source.get("type", "message") == "message"]
            qas.append(qa)
        samples.append({"sample_id": history["original_id"], "qa": qas})
    if rows:
        raise ValueError("predictions do not belong to the LoCoMo dataset")
    _save_json(path, samples)


def official_score_command(*, benchmark, repository, predictions_path, dataset_path,
                           python_executable=None, judge_model="gpt-4o", prediction_key="memory_prediction"):
    """Prepare argv only. Never execute a paid or external scorer implicitly."""
    for name, value in (("repository", repository), ("predictions_path", predictions_path), ("dataset_path", dataset_path)):
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"{name} is empty")
        if not Path(value).exists():
            raise FileNotFoundError(f"{name} does not exist: {value}")
    python_executable = str(python_executable or sys.executable)
    repository = Path(repository).resolve()
    if benchmark == "longmemeval":
        script = repository / "src" / "evaluation" / "evaluate_qa.py"
        argv = [python_executable, str(script), judge_model, str(Path(predictions_path).resolve()), str(Path(dataset_path).resolve())]
    elif benchmark == "locomo":
        script = repository / "task_eval" / "evaluation.py"
        bridge = Path(__file__).resolve().parents[2] / "eval" / "score_locomo_official.py"
        argv = [python_executable, str(bridge), "--repository", str(repository),
                "--predictions", str(Path(predictions_path).resolve()), "--prediction-key", prediction_key,
                "--dataset", str(Path(dataset_path).resolve())]
    else:
        raise ValueError("benchmark must be longmemeval or locomo")
    if not script.is_file():
        raise FileNotFoundError(f"official scoring script not found: {script}")
    return {"argv": argv, "cwd": str(repository), "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "requires_api_key": benchmark == "longmemeval", "executed": False}
