"""CLI: replay the LoCoMo lexical baseline and emit offline diagnostics.

Example:
    python -X utf8 -m eval.diagnostics.run_diagnostics --limit 20

The command only reads the dataset and the existing report. It writes one JSON
artifact under ``data/benchmark_runs/diagnostics/`` and never calls a model.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from src.eval.benchmark_datasets import load_benchmark_dataset
from src.hybrid_retriever import HybridRetriever
from src.message_store import MessageStore

from . import checks, metrics

DEFAULT_DATASET = "external/locomo/data/locomo10.json"
DEFAULT_REPORT = "data/benchmark_runs/locomo-smoke-v2-20q-report.json"
DEFAULT_OUTPUT = "data/benchmark_runs/diagnostics/locomo-conv-26-20q.json"
DEFAULT_HISTORY = "conv-26"
DEFAULT_LIMIT = 20


def prepare(dataset, *, history=None, history_index=None, limit=None, report=None):
    """Build the in-memory snapshot for one history and select its questions."""
    histories = dataset["histories"]
    if history_index is not None:
        if type(history_index) is not int or not 0 <= history_index < len(histories):
            raise ValueError("history_index must point at an existing history")
        selected = histories[history_index]
    else:
        matches = [item for item in histories if item.get("original_id") == history]
        if not matches:
            available = sorted(
                str(item.get("original_id")) for item in histories
            )
            raise ValueError(
                f"history {history!r} not found; available: {', '.join(available)}"
            )
        selected = matches[0]
    if limit is not None:
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        questions = list(selected["questions"][:limit]) if limit else list(
            selected["questions"]
        )
    else:
        questions = list(selected["questions"])
    if not questions:
        raise ValueError("selected history has no questions")

    store = MessageStore(":memory:")
    try:
        for message in selected["messages"]:
            store.append_message(**deepcopy(message))
        retriever = HybridRetriever(store)
        records = store.list_user_messages(user_id=selected["user_id"])
        return checks.DiagnosticContext(
            dataset=dataset, history=selected, questions=questions,
            store=store, retriever=retriever, records=records, report=report,
        )
    except Exception:
        store.close()
        raise


def evaluate(dataset, *, history=None, history_index=None, limit=None,
             report=None, checks_names=None):
    context = prepare(
        dataset, history=history, history_index=history_index,
        limit=limit, report=report,
    )
    names = list(checks_names) if checks_names else list(checks.CHECKS)
    unknown = [name for name in names if name not in checks.CHECKS]
    if unknown:
        context.store.close()
        raise ValueError(f"unknown checks: {', '.join(unknown)}")
    try:
        results = {name: checks.CHECKS[name](context) for name in names}
    finally:
        context.store.close()
    return {
        "dataset": {
            "benchmark": dataset.get("benchmark"),
            "adapter_version": dataset.get("adapter_version"),
            "dataset_sha256": dataset.get("dataset_sha256"),
            "assumed_timezone": dataset.get("assumed_timezone"),
            "history_id": context.history["history_id"],
            "history_original_id": context.history.get("original_id"),
            "question_ids": [question["question_id"] for question in context.questions],
        },
        "checks": results,
        "summary": _summarize(results),
    }


def _summarize(results):
    return {
        "statuses": {name: result["status"] for name, result in results.items()},
        "conclusions": {
            name: result["conclusion"] for name, result in results.items()
        },
        "reproduction_ok": results.get("V0", {}).get("status") == "pass",
    }


def _headline(name, result):
    numbers = result.get("metrics", {})
    if name == "V0" and "aggregate" in numbers:
        aggregate = numbers["aggregate"]
        return (
            f"source={aggregate.get('source_recall_at_k')}, "
            f"session={aggregate.get('session_recall_at_k')}"
        )
    if name == "V1":
        return f"missing={numbers.get('missing_evidence_turns')}, ±1={numbers.get('within_1')}"
    if name == "V2":
        return f"overall_buckets={numbers.get('overall_buckets')}"
    if name == "V3":
        variant = (numbers.get("variants") or {}).get("with_date", {})
        return f"session_top8={variant.get('coverage_at_8')}"
    if name == "V4":
        curve = numbers.get("curve") or []
        return ", ".join(
            f"k={item['top_k']}:{item['source_recall_at_k']:.3f}"
            for item in curve if item["source_recall_at_k"] is not None
        )
    if name == "V5":
        return (
            f"cat1={numbers.get('question_count')}, "
            f"max_sessions={numbers.get('max_evidence_sessions')}"
        )
    if name == "V6":
        return (
            f"blind_answers={numbers.get('blind_answer_questions')}, "
            f"evidence_but_abstained={numbers.get('evidence_present_but_abstained')}"
        )
    if name == "V7":
        return (
            f"abstained={numbers.get('abstention_count')}, "
            f"gold_supported={numbers.get('gold_supported_by_context')}"
        )
    if name == "V8":
        estimates = numbers.get("estimates") or {}
        value = (estimates.get(20) or {}).get("full_benchmark_input_tokens")
        return f"k20_full_input≈{value}"
    if name == "V9":
        selected = (numbers.get("selected") or {}).get("category_share")
        return f"selected_categories={sorted((selected or {}))}"
    if name == "V10":
        recommended = numbers.get("recommended") or {}
        return (
            f"{recommended.get('name')}: recall={recommended.get('source_recall_at_k')}, "
            f"tokens={recommended.get('mean_input_tokens')}"
        )
    return ""


def _print_summary(result):
    dataset = result["dataset"]
    print(
        f"LoCoMo diagnostics: history={dataset['history_original_id']} "
        f"({dataset['history_id'][:12]}...), "
        f"questions={len(dataset['question_ids'])}"
    )
    for name, check in result["checks"].items():
        headline = _headline(name, check)
        print(f"  {name} [{check['status']}] {headline}")
        print(f"      {check['conclusion']}")
    print(f"reproduction_ok={result['summary']['reproduction_ok']}")


def _write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--benchmark", default="locomo", choices=("locomo", "longmemeval"))
    parser.add_argument("--assumed-timezone", default="+00:00")
    parser.add_argument("--history", default=DEFAULT_HISTORY,
                        help="LoCoMo sample_id, e.g. conv-26")
    parser.add_argument("--history-index", type=int, default=None,
                        help="select by position instead of sample_id")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help="first N questions; 0 means all questions")
    parser.add_argument("--report", default=DEFAULT_REPORT,
                        help='report to compare against; pass "" to disable')
    parser.add_argument("--output", default=None)
    parser.add_argument("--checks", default="all",
                        help="comma separated subset, e.g. V0,V1,V4")
    parser.add_argument("--allow-mismatch", action="store_true",
                        help="do not exit 2 when V0 disagrees with the report")
    args = parser.parse_args(argv)

    names = None if args.checks.strip().casefold() == "all" else [
        name.strip().upper() for name in args.checks.split(",") if name.strip()
    ]
    report = None
    if args.report:
        report_path = Path(args.report)
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            print(f"warning: report not found: {report_path}", file=sys.stderr)

    output = args.output
    if output is None:
        output = f"data/benchmark_runs/diagnostics/{args.history}-{args.limit}q.json"
    try:
        dataset = load_benchmark_dataset(
            args.dataset, benchmark=args.benchmark,
            assumed_timezone=args.assumed_timezone,
        )
        result = evaluate(
            dataset, history=args.history, history_index=args.history_index,
            limit=args.limit, report=report, checks_names=names,
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"diagnostics failed: {error}", file=sys.stderr)
        return 1
    _write_json(output, result)
    _print_summary(result)
    print(f"report written: {output}")
    if (result["checks"].get("V0", {}).get("status") == "fail"
            and not args.allow_mismatch):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
