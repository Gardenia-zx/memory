"""Run a real raw-history lexical benchmark and export reproducible artifacts."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

from src.eval.benchmark_datasets import load_benchmark_dataset
from src.eval.benchmark_runner import (
    export_locomo_predictions,
    export_predictions_jsonl,
    official_score_command,
    run_benchmark,
    save_benchmark_report,
)
from src.hybrid_retriever import HybridRetriever
from src.session_retriever import SessionFirstRetriever
from src.message_store import MessageStore
from src.model_adapter import (
    ModelConfig,
    ModelNotConfiguredError,
    OpenAIResponsesAdapter,
)


ANSWER_PROMPT_VERSION = "raw-lexical-json-v2"


def _reject_constant(value):
    raise ValueError(f"JSON does not allow {value}")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_config(path):
    path = Path(path).resolve()
    raw = path.read_text(encoding="utf-8-sig")
    config = json.loads(
        raw, parse_constant=_reject_constant, object_pairs_hook=_unique_keys,
    )
    if not isinstance(config, dict):
        raise ValueError("benchmark config must be a JSON object")
    json.dumps(config, ensure_ascii=False, allow_nan=False)
    return config, path


def _positive_limit(value, name):
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_path(value, *, base_dir, field, default=None):
    value = default if value in (None, "") else value
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{field} is empty")
    path = Path(value)
    return (path if path.is_absolute() else base_dir / path).resolve()


def _executable(value, *, base_dir):
    if value in (None, ""):
        return sys.executable
    if not isinstance(value, str) or not value.strip():
        raise ValueError("official_python must be nonempty text")
    if "/" not in value and "\\" not in value:
        return value
    path = _resolve_path(value, base_dir=base_dir, field="official_python")
    if not path.is_file():
        raise FileNotFoundError(f"official Python not found: {path}")
    return str(path)


def _select_dataset(dataset, *, limit_histories=None, limit_questions=None):
    limit_histories = _positive_limit(limit_histories, "limit_histories")
    limit_questions = _positive_limit(limit_questions, "limit_questions")
    original_histories = dataset["histories"]
    histories = original_histories[:limit_histories] if limit_histories else original_histories
    selected, remaining = [], limit_questions
    for original in histories:
        if remaining == 0:
            break
        history = deepcopy(original)
        if remaining is not None:
            history["questions"] = history["questions"][:remaining]
            remaining -= len(history["questions"])
        if history["questions"] or limit_questions is None:
            selected.append(history)
    if not selected:
        raise ValueError("benchmark limits selected no questions")
    result = {**dataset, "histories": selected}
    original_ids = [
        question["question_id"]
        for history in original_histories for question in history["questions"]
    ]
    selected_ids = [
        question["question_id"]
        for history in selected for question in history["questions"]
    ]
    scope = {
        "full": (selected_ids == original_ids
                 and [history["history_id"] for history in selected]
                 == [history["history_id"] for history in original_histories]),
        "original_history_count": len(original_histories),
        "selected_history_count": len(selected),
        "original_question_count": len(original_ids),
        "selected_question_count": len(selected_ids),
        "limit_histories": limit_histories,
        "limit_questions": limit_questions,
    }
    return result, scope


class _ReadOnlyMessages:
    """The small MessageStore surface HybridRetriever needs on a query snapshot."""

    def __init__(self, connection):
        self.conn = connection

    def assert_user_active(self, *, user_id):
        MessageStore.validate_ids(user_id=user_id)
        if self.conn.execute(
            "SELECT 1 FROM memory_forget_tombstones WHERE user_id = ?", (user_id,),
        ).fetchone() is not None:
            raise ValueError("benchmark user was forgotten")

    def list_user_messages(self, *, user_id):
        MessageStore.validate_ids(user_id=user_id)
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE user_id = ? ORDER BY session_id, seq",
            (user_id,),
        ).fetchall()
        result = []
        for row in rows:
            message = dict(row)
            message["payload"] = json.loads(message.pop("payload_json"))
            result.append(message)
        return result


class LexicalAnswerer:
    def __init__(self, adapter, *, top_k, max_context_characters,
                 retriever_name="lexical", session_count=6,
                 turns_per_session=None, neighbors=False):
        if not hasattr(adapter, "complete_json"):
            raise ValueError("adapter must implement complete_json")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if type(max_context_characters) is not int or max_context_characters < 1:
            raise ValueError("max_context_characters must be a positive integer")
        if retriever_name not in ("lexical", "session_first"):
            raise ValueError("retriever must be lexical or session_first")
        if type(session_count) is not int or session_count < 1:
            raise ValueError("session_count must be a positive integer")
        if turns_per_session is not None and (
                type(turns_per_session) is not int or turns_per_session < 1):
            raise ValueError("turns_per_session must be None or a positive integer")
        if type(neighbors) is not bool:
            raise ValueError("neighbors must be boolean")
        self.adapter = adapter
        self.top_k = top_k
        self.max_context_characters = max_context_characters
        self.retriever_name = retriever_name
        self.session_count = session_count
        self.turns_per_session = turns_per_session
        self.neighbors = neighbors

    def __call__(self, connection, question):
        store = _ReadOnlyMessages(connection)
        if self.retriever_name == "session_first":
            retrieved = SessionFirstRetriever(
                store, session_count=self.session_count,
                turns_per_session=self.turns_per_session,
                neighbors=self.neighbors,
            ).search(
                user_id=question["user_id"], query=question["question"],
                as_of=question["as_of"],
            )
        else:
            retrieved = HybridRetriever(store).search(
                user_id=question["user_id"], query=question["question"],
                top_k=self.top_k, mode="lexical", as_of=question["as_of"],
            )
        context, selected, remaining = [], [], self.max_context_characters
        for item in retrieved:
            if remaining == 0:
                break
            content = item["text"][:remaining]
            if not content:
                continue
            context.append({
                "source": item["source"], "occurred_at": item["metadata"]["occurred_at"],
                "role": item["metadata"]["role"], "content": content,
            })
            selected.append(item)
            remaining -= len(content)
        response = self.adapter.complete_json(
            task="benchmark_answer",
            payload={
                "instructions": (
                    "Return exactly {\"answer\":\"...\"} and no other fields. Answer only "
                    "from retrieved_context. Use each item's occurred_at, not the question's "
                    "as_of, as the anchor for relative times: resolve yesterday to the preceding "
                    "calendar day, last week to 'the week before <anchor date>', and next month "
                    "or this month to the corresponding named month and year. Preserve the "
                    "evidence's time granularity: format an exact day as '5 March 2024' and a "
                    "month as 'March 2024'; never invent a more precise date. When asked for a "
                    "duration, return only the supported duration without inventing extra "
                    "precision. Keep the answer concise and close to the evidence wording. "
                    "Do not use ISO date format. If the context does not support an answer, return "
                    "{\"answer\":\"No information available\"}."
                ),
                "question": question["question"],
                "as_of": question["as_of"],
                "retrieved_context": context,
            },
        )
        if not isinstance(response, str):
            raise ValueError("model adapter must return JSON text")
        parsed = json.loads(
            response, parse_constant=_reject_constant, object_pairs_hook=_unique_keys,
        )
        if (not isinstance(parsed, dict) or set(parsed) != {"answer"}
                or not isinstance(parsed["answer"], str) or not parsed["answer"].strip()):
            raise ValueError("benchmark answer must be a nonempty JSON answer string")
        result = {
            "answer": parsed["answer"],
            # Sources come only from local retrieval, never from model output.
            "retrieved_sources": [item["source"] for item in selected],
        }
        usage = getattr(self.adapter, "last_usage", None)
        if usage:
            result["usage"] = deepcopy(usage)
        return result


def _model_adapter(config):
    model = config.get("model")
    env_name = config.get("api_key_env", "OPENAI_API_KEY")
    if not isinstance(env_name, str) or not env_name.strip():
        raise ValueError("api_key_env must be a nonempty string")
    api_key = os.environ.get(env_name) or config.get("api_key")
    if not isinstance(model, str) or not model.strip():
        raise ModelNotConfiguredError("benchmark model is empty")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ModelNotConfiguredError(f"benchmark API key is missing; set {env_name}")
    return OpenAIResponsesAdapter(ModelConfig(
        model=model,
        api_key=api_key,
        base_url=config.get("base_url", "https://api.openai.com/v1"),
        timeout_seconds=config.get("timeout_seconds", 60.0),
        api_mode=config.get("api_mode", "responses"),
        max_output_tokens=config.get("max_output_tokens", 2048),
    ))


def _validate_output_paths(paths, protected):
    used = [path for path in paths if path is not None]
    if len(set(used)) != len(used):
        raise ValueError("benchmark output paths must be different")
    if any(path in protected for path in used):
        raise ValueError("benchmark output must not overwrite config or dataset input")


def _require_suffix(path, suffix, field):
    if path.suffix.casefold() != suffix:
        raise ValueError(f"{field} must end with {suffix}")


def run_from_config(
    config, *, adapter=None, config_dir=None, config_path=None,
    limit_histories=None, limit_questions=None, report_path=None,
    predictions_path=None, allow_full=False,
):
    if not isinstance(config, dict):
        raise ValueError("benchmark config must be an object")
    if type(allow_full) is not bool:
        raise ValueError("allow_full must be boolean")
    if not allow_full and limit_questions is None:
        raise ValueError("smoke runs require limit_questions")
    benchmark = config.get("benchmark")
    if benchmark not in {"longmemeval", "locomo"}:
        raise ValueError("benchmark must be longmemeval or locomo")
    top_k = config.get("top_k", 5)
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    max_context_characters = config.get("max_context_characters", 60000)
    if type(max_context_characters) is not int or max_context_characters < 1:
        raise ValueError("max_context_characters must be a positive integer")
    run_config = deepcopy(config.get("run_config", {}))
    if not isinstance(run_config, dict):
        raise ValueError("run_config must be an object")
    retriever_name = run_config.get("retriever", "lexical")
    if retriever_name not in ("lexical", "session_first"):
        raise ValueError("retriever must be lexical or session_first")
    if run_config.get("graph_enabled", False) is not False:
        raise ValueError("this raw-history baseline does not enable graph retrieval")
    session_count = run_config.get("session_count", 6)
    if type(session_count) is not int or session_count < 1:
        raise ValueError("session_count must be a positive integer")
    turns_per_session = run_config.get("turns_per_session")
    if turns_per_session is not None and (
            type(turns_per_session) is not int or turns_per_session < 1):
        raise ValueError("turns_per_session must be None or a positive integer")
    neighbors = run_config.get("neighbors", False)
    if type(neighbors) is not bool:
        raise ValueError("neighbors must be boolean")
    if "context_budget" in run_config:
        raise ValueError(
            "run_config.context_budget is not used by this baseline; use "
            "max_context_characters instead"
        )
    answer_attempts = config.get("answer_attempts", 3)
    if type(answer_attempts) is not int or answer_attempts < 1:
        raise ValueError("answer_attempts must be a positive integer")
    retry_delay_seconds = config.get("retry_delay_seconds", 2.0)
    if type(retry_delay_seconds) not in (int, float) or retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be a nonnegative number")
    base_dir = Path(config_dir or Path.cwd()).resolve()
    dataset_path = _resolve_path(
        config.get("dataset_path"), base_dir=base_dir, field="dataset_path",
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"benchmark dataset not found: {dataset_path}")
    adapter = _model_adapter(config) if adapter is None else adapter

    full_dataset = load_benchmark_dataset(
        dataset_path, benchmark=benchmark,
        assumed_timezone=config.get("assumed_timezone", "+00:00"),
    )
    dataset, scope = _select_dataset(
        full_dataset, limit_histories=limit_histories,
        limit_questions=limit_questions,
    )
    if scope["full"] and not allow_full:
        raise ValueError("selected scope is the full benchmark; explicitly allow_full")
    if not scope["full"] and predictions_path is not None:
        raise ValueError("--predictions is available only for a full benchmark run")
    suffix = "" if scope["full"] else "-smoke"
    output_directory = _resolve_path(
        config.get("output_directory"), base_dir=base_dir,
        field="output_directory", default="reports",
    )
    report_path = _resolve_path(
        report_path or config.get("report_path"), base_dir=base_dir,
        field="report_path",
        default=output_directory / f"{benchmark}{suffix}-report.json",
    )
    _require_suffix(report_path, ".json", "report_path")
    checkpoint_value = config.get("checkpoint_path")
    if checkpoint_value in (None, ""):
        checkpoint_path = report_path.with_name(
            report_path.name + ".checkpoint.jsonl"
        )
    else:
        checkpoint_path = _resolve_path(
            checkpoint_value, base_dir=base_dir, field="checkpoint_path",
        )
    prediction_extension = "jsonl" if benchmark == "longmemeval" else "json"
    prediction_path = None
    if scope["full"]:
        prediction_path = _resolve_path(
            predictions_path or config.get("predictions_path"), base_dir=base_dir,
            field="predictions_path",
            default=output_directory / f"{benchmark}-predictions.{prediction_extension}",
        )
        _require_suffix(
            prediction_path, ".jsonl" if benchmark == "longmemeval" else ".json",
            "predictions_path",
        )
    config_path = Path(config_path).resolve() if config_path is not None else None
    bridge_path = Path(__file__).resolve().with_name("score_locomo_official.py")
    protected = {
        dataset_path, Path(__file__).resolve(), bridge_path,
        *([] if config_path is None else [config_path]),
    }
    repository = None
    command_path = None
    official_scores_path = None
    official_python = None
    if scope["full"] and config.get("official_repository") not in (None, ""):
        repository = _resolve_path(
            config["official_repository"], base_dir=base_dir,
            field="official_repository",
        )
        official_script = repository / (
            "src/evaluation/evaluate_qa.py" if benchmark == "longmemeval"
            else "task_eval/evaluation.py"
        )
        if not official_script.is_file():
            raise FileNotFoundError(f"official scoring script not found: {official_script}")
        protected.add(official_script.resolve())
        official_python = _executable(
            config.get("official_python"), base_dir=base_dir,
        )
        judge_model = config.get("judge_model", "gpt-4o")
        if not isinstance(judge_model, str) or not judge_model.strip():
            raise ValueError("judge_model must be nonempty text")
        command_path = _resolve_path(
            config.get("official_command_path"), base_dir=base_dir,
            field="official_command_path",
            default=output_directory / f"{benchmark}-official-command.json",
        )
        _require_suffix(command_path, ".json", "official_command_path")
        if benchmark == "locomo":
            official_scores_path = _resolve_path(
                config.get("official_scores_path"), base_dir=base_dir,
                field="official_scores_path",
                default=output_directory / "locomo-official-scores.json",
            )
            _require_suffix(official_scores_path, ".json", "official_scores_path")
    _validate_output_paths(
        [report_path, prediction_path, command_path, official_scores_path,
         checkpoint_path],
        protected,
    )

    run_config.update({
        "model": config.get("model") or type(adapter).__name__,
        "api_mode": config.get("api_mode", "responses"),
        "base_url": config.get("base_url", "https://api.openai.com/v1"),
        "max_output_tokens": config.get("max_output_tokens", 2048),
        "dataset_variant": "locomo10" if benchmark == "locomo"
                           else run_config.get("dataset_variant", "cleaned_s"),
        "retriever": retriever_name,
        "session_count": session_count,
        "turns_per_session": turns_per_session,
        "neighbors": neighbors,
        "graph_enabled": False,
        "ingestion_mode": "raw_history_baseline",
        "answer_prompt_version": ANSWER_PROMPT_VERSION,
        "max_context_characters": max_context_characters,
        "run_scope": deepcopy(scope),
    })
    # Answers are checkpointed per question; a rerun reuses completed questions.
    report = run_benchmark(
        dataset, answer_question=LexicalAnswerer(
            adapter, top_k=top_k,
            max_context_characters=max_context_characters,
            retriever_name=retriever_name, session_count=session_count,
            turns_per_session=turns_per_session, neighbors=neighbors,
        ),
        top_k=top_k, run_config=run_config,
        source_limit=top_k if retriever_name == "lexical" else None,
        answer_attempts=answer_attempts,
        retry_delay_seconds=retry_delay_seconds,
        checkpoint_path=checkpoint_path,
    )
    report["run_scope"] = scope
    official = {"eligible": scope["full"], "command_generated": False}
    if not scope["full"]:
        official["reason"] = "subset smoke runs are not eligible for official scoring"
    else:
        if benchmark == "longmemeval":
            export_predictions_jsonl(report, prediction_path)
        else:
            export_locomo_predictions(dataset, report, prediction_path)
        if repository is not None:
            command = official_score_command(
                benchmark=benchmark, repository=repository,
                predictions_path=prediction_path, dataset_path=dataset_path,
                python_executable=official_python,
                judge_model=judge_model,
            )
            if official_scores_path is not None:
                command["argv"].extend(["--output", str(official_scores_path)])
                command["expected_output_path"] = str(official_scores_path)
            entrypoint = Path(command["argv"][1])
            command.update({
                "benchmark": benchmark,
                "dataset_sha256": report["dataset_sha256"],
                "predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
                "entrypoint_sha256": hashlib.sha256(entrypoint.read_bytes()).hexdigest(),
            })
            save_benchmark_report(command, command_path)
            official["command_generated"] = True
            official["command_path"] = str(command_path)
        else:
            official["reason"] = "official_repository is not configured"
    report["official_scoring"] = official
    save_benchmark_report(report, report_path)
    return {
        "report": report,
        "report_path": str(report_path),
        "predictions_path": None if prediction_path is None else str(prediction_path),
        "official_command_path": None if command_path is None else str(command_path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-histories", type=int)
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument("--full", action="store_true",
                        help="explicitly allow every question and its model cost")
    parser.add_argument("--report", help="path relative to the config directory")
    parser.add_argument("--predictions", help="full runs only; relative to the config directory")
    parser.add_argument("--retriever", choices=("lexical", "session_first"),
                        default=None)
    parser.add_argument("--session-count", type=int, default=None)
    parser.add_argument("--turns-per-session", type=int, default=None)
    args = parser.parse_args(argv)
    if args.full and (args.limit_histories is not None or args.limit_questions is not None):
        parser.error("--full cannot be combined with limits")
    if not args.full and args.limit_questions is None:
        parser.error("smoke tests require --limit-questions; otherwise pass --full")
    config, config_path = load_config(args.config)
    overrides = {
        "retriever": args.retriever,
        "session_count": args.session_count,
        "turns_per_session": args.turns_per_session,
    }
    if any(value is not None for value in overrides.values()):
        run_config = config.get("run_config")
        if not isinstance(run_config, dict):
            parser.error("run_config must be an object to override retrieval")
        for key, value in overrides.items():
            if value is not None:
                run_config[key] = value
    result = run_from_config(
        config, config_dir=config_path.parent, config_path=config_path,
        limit_histories=args.limit_histories,
        limit_questions=args.limit_questions,
        report_path=args.report, predictions_path=args.predictions,
        allow_full=args.full,
    )
    print(json.dumps({
        "question_count": result["report"]["overall"]["question_count"],
        "source_recall_at_k": result["report"]["overall"]["source_recall_at_k"],
        "report_path": result["report_path"],
        "predictions_path": result["predictions_path"],
        "official_command_path": result["official_command_path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
