"""加载固定开发数据、执行检索评测，并保存 JSON 基线报告。"""

import json
from pathlib import Path

from src.eval.retrieval_evaluator import evaluate_retrieval
from src.history_retriever import HistoryRetriever
from src.message_store import MessageStore


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "data" / "retrieval_dev.json"
DEFAULT_REPORT_PATH = Path(__file__).resolve().parent / "reports" / "retrieval_baseline.json"
_SOURCE_FIELDS = ("user_id", "session_id", "message_id")


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value


def _source_ref(value: object, context: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{context} 必须是对象")
    return {
        field: _require_string(value.get(field), f"{context}.{field}")
        for field in _SOURCE_FIELDS
    }


def run_evaluation(dataset_path: str | Path, *, top_k: int = 5) -> dict:
    """在独立内存库中评测数据集，返回带版本信息的结果字典。"""
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k 必须是正整数")

    with Path(dataset_path).open(encoding="utf-8") as dataset_file:
        dataset = json.load(dataset_file)
    if not isinstance(dataset, dict):
        raise ValueError("数据集顶层必须是对象")
    dataset_id = _require_string(dataset.get("dataset_id"), "dataset_id")
    history = dataset.get("history")
    cases = dataset.get("cases")
    if not isinstance(history, list) or not isinstance(cases, list):
        raise ValueError("history 和 cases 必须是列表")

    store = MessageStore(":memory:")
    try:
        for index, message in enumerate(history):
            source = _source_ref(message, f"history[{index}]")
            # 数据集重复记录应报错，避免存储层的幂等返回掩盖标注错误。
            if store.get_message(**source) is not None:
                raise ValueError(f"history[{index}] 的消息来源重复：{source}")
            store.append_message(**message)

        case_ids = set()
        for index, case in enumerate(cases):
            if not isinstance(case, dict):
                raise ValueError(f"cases[{index}] 必须是对象")
            case_id = _require_string(case.get("case_id"), f"cases[{index}].case_id")
            if case_id in case_ids:
                raise ValueError(f"案例 {case_id} 的 case_id 重复")
            case_ids.add(case_id)
            user_id = _require_string(case.get("user_id"), f"案例 {case_id} 的 user_id")
            if not isinstance(case.get("query"), str):
                raise ValueError(f"案例 {case_id} 的 query 必须是字符串")
            sources = case.get("relevant_sources")
            if not isinstance(sources, list):
                raise ValueError(f"案例 {case_id} 的 relevant_sources 必须是列表")
            for reference in sources:
                source = _source_ref(reference, f"案例 {case_id} 的正确来源")
                if source["user_id"] != user_id:
                    raise ValueError(f"案例 {case_id} 的正确来源属于其他用户")
                if store.get_message(**source) is None:
                    raise ValueError(f"案例 {case_id} 的正确来源不存在：{source}")

        # 只有历史消息入库，查询与来源标签留在评测层。
        report = evaluate_retrieval(HistoryRetriever(store), cases, top_k=top_k)
        report.update(dataset_id=dataset_id, retriever="keyword_v1")
        return report
    finally:
        store.close()


def save_report(report: dict, report_path: str | Path) -> Path:
    """以 UTF-8 保存完整报告；序列化成功后才创建或覆盖输出文件。"""
    report_json = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_json + "\n", encoding="utf-8")
    return path


def main() -> None:
    report = run_evaluation(DEFAULT_DATASET_PATH, top_k=2)
    report_path = save_report(report, DEFAULT_REPORT_PATH)
    top_k = report["top_k"]
    print(f"数据集：{report['dataset_id']}")
    print(f"检索器：{report['retriever']}")
    print(f"top_k：{top_k}")
    print(
        f"案例数：{len(report['details'])}"
        f"（有答案 {report['answerable_count']}，无答案 {report['no_answer_count']}）"
    )
    for label, field in (
        (f"Recall@{top_k}", "mean_recall_at_k"),
        (f"Hit@{top_k}", "hit_rate_at_k"),
        ("无答案返回空比例", "no_answer_empty_rate"),
    ):
        value = report[field]
        display = "N/A" if value is None else f"{value:.2%}"
        print(f"{label}：{display}")
    print(f"报告已保存：{report_path.resolve()}")


if __name__ == "__main__":
    main()
