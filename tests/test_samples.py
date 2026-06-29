"""Tests for unified seed input schema and normalization."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracesynth.io import (
    SeedRecordError,
    InputConfig,
    normalize_seed_record,
    load_seed_records,
    extract_predicted_answer,
    check_label_match,
    validate_seed_info,
)


def _expect_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def test_normalize_question_alias():
    record = normalize_seed_record({
        "id": "t1",
        "query": "什么是 GRPO？",
        "label": "群体相对策略优化",
    })
    assert record.question == "什么是 GRPO？"
    assert record.label == "群体相对策略优化"


def test_missing_label_raises():
    _expect_raises(
        SeedRecordError,
        lambda: normalize_seed_record({"id": "t1", "question": "hello"}),
    )


def test_context_list_joined():
    record = normalize_seed_record({
        "question": "q",
        "label": "a",
        "context": ["doc1", "doc2"],
    })
    assert record.context == "doc1\n\ndoc2"


def test_auto_id_when_missing():
    record = normalize_seed_record({
        "question": "same q",
        "label": "same label",
    })
    assert record.id.startswith("seed-")


def test_validate_seed_info_builds_background():
    seed_info = validate_seed_info({
        "id": "rag-001",
        "question": "问题",
        "label": "答案",
        "context": "上下文",
    })
    assert seed_info["question"] == "问题"
    assert "标准答案" in seed_info["background"]
    assert "上下文" in seed_info["background"]


def test_load_seed_records_from_fixture():
    fixture = PROJECT_ROOT / "configs" / "seed_qa_sample.jsonl"
    records = load_seed_records(fixture, input_config=InputConfig())
    assert len(records) == 3
    assert all(record.label for record in records)


def test_extract_predicted_answer():
    history = [
        {"role": "assistant", "content": "thinking"},
        {"role": "assistant", "content": "done <answer>最终答案</answer>"},
    ]
    assert extract_predicted_answer(history) == "最终答案"


def test_check_label_match():
    result = check_label_match("最终答案是 GRPO 相对优化", "GRPO 相对优化")
    assert result["label_match_status"] == "match"


def test_check_label_mismatch():
    result = check_label_match("完全不相关", "标准答案内容")
    assert result["label_match_status"] == "mismatch"


if __name__ == "__main__":
    test_normalize_question_alias()
    test_missing_label_raises()
    test_context_list_joined()
    test_auto_id_when_missing()
    test_validate_seed_info_builds_background()
    test_load_seed_records_from_fixture()
    test_extract_predicted_answer()
    test_check_label_match()
    test_check_label_mismatch()
    print("All sample schema tests passed")
