"""Run an installed official LoCoMo scorer on frozen predictions, never regenerate answers."""

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def score_predictions(*, repository, predictions, dataset, prediction_key="memory_prediction", evaluator=None):
    script = Path(repository) / "task_eval" / "evaluation.py"
    if not script.is_file():
        raise FileNotFoundError(f"official LoCoMo scorer not found: {script}")
    injected = evaluator is not None
    if evaluator is None:
        spec = importlib.util.spec_from_file_location("locomo_official_evaluation", script)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ImportError as error:
            raise RuntimeError("Install the official LoCoMo repository dependencies in the configured scoring interpreter") from error
        evaluator = module.eval_question_answering
    references = json.loads(Path(dataset).read_text(encoding="utf-8-sig"))
    outputs = json.loads(Path(predictions).read_text(encoding="utf-8-sig"))
    if not isinstance(references, list) or not isinstance(outputs, list):
        raise ValueError("official references and exported predictions must be JSON lists")
    output_ids = [str(sample["sample_id"]) for sample in outputs]
    reference_ids = [str(sample["sample_id"]) for sample in references]
    if len(set(output_ids)) != len(output_ids) or len(set(reference_ids)) != len(reference_ids) or set(output_ids) != set(reference_ids):
        raise ValueError("predictions must match official sample IDs exactly")
    by_id = {str(sample["sample_id"]): sample for sample in outputs}
    rows = []
    for sample in references:
        predicted = by_id[str(sample["sample_id"])]
        if len(predicted["qa"]) != len(sample["qa"]):
            raise ValueError("prediction QA count differs from official reference")
        qas = []
        for reference, output in zip(sample["qa"], predicted["qa"]):
            if not isinstance(output.get(prediction_key), str) or output.get("question") != reference["question"]:
                raise ValueError("prediction questions must remain aligned with official references")
            qa = deepcopy(reference)
            # The released metric reads answer even for category 5, then scores abstention.
            qa.setdefault("answer", "")
            qa.setdefault("evidence", [])
            qa[prediction_key] = output[prediction_key]
            if prediction_key + "_context" in output:
                qa[prediction_key + "_context"] = deepcopy(output[prediction_key + "_context"])
            qas.append(qa)
        scores, _, recall = evaluator(qas, eval_key=prediction_key)
        if len(scores) != len(qas) or len(recall) != len(qas):
            raise ValueError("official scorer returned the wrong number of scores or recalls")
        for index, (qa, score) in enumerate(zip(qas, scores)):
            score = float(score)
            recall_score = float(recall[index])
            if (not math.isfinite(score) or not 0 <= score <= 1
                    or not math.isfinite(recall_score) or not 0 <= recall_score <= 1):
                raise ValueError("official scorer returned an invalid score or recall")
            rows.append({"sample_id": sample["sample_id"], "question_index": index,
                         "category": qa["category"], "score": score, "retrieval_recall": recall_score})
    categories = sorted({row["category"] for row in rows})
    return {"metric": "injected_test_scorer" if injected else "official_locomo_f1",
            "official_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "question_count": len(rows),
            "overall": sum(row["score"] for row in rows) / len(rows) if rows else None,
            "retrieval_recall": sum(row["retrieval_recall"] for row in rows) / len(rows) if rows else None,
            "by_category": {str(category): {
                "count": sum(row["category"] == category for row in rows),
                "score": sum(row["score"] for row in rows if row["category"] == category) / sum(row["category"] == category for row in rows),
                "retrieval_recall": sum(row["retrieval_recall"] for row in rows if row["category"] == category) / sum(row["category"] == category for row in rows)
            } for category in categories}, "details": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prediction-key", default="memory_prediction")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = score_predictions(repository=args.repository, predictions=args.predictions,
                               dataset=args.dataset, prediction_key=args.prediction_key)
    path = Path(args.output or args.predictions + ".official-scores.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
