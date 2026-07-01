import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tool_use_data_gen import get_ids_to_skip, write_exception_failure
from tracesynth.io import read_failed_ids, read_failure_attempt_count, write_failure_record


def _read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_write_failure_record_schema_and_attempt_index(tmp_path):
    failed_path = tmp_path / "virtual_tool_use.failed.jsonl"
    seed_info = {
        "id": "rag-001",
        "question": "q",
        "label": "a",
        "context": "ctx",
    }
    final_state = {
        "fuzzy_task": "q",
        "checked_tools": [{"name": "Search", "parameters": {}}],
    }

    first = write_failure_record(
        failed_path,
        seed_info=seed_info,
        final_state=final_state,
        stage="graph",
        failure_type="generation_failed",
        failure_reason="bad tool schema",
    )
    second = write_failure_record(
        failed_path,
        seed_info=seed_info,
        final_state=final_state,
        stage="graph",
        failure_type="generation_failed",
        failure_reason="bad tool schema again",
    )

    rows = _read_jsonl(failed_path)
    assert first["attempt_index"] == 1
    assert second["attempt_index"] == 2
    assert rows[0]["stage"] == "graph"
    assert rows[0]["failure_type"] == "generation_failed"
    assert rows[0]["context_present"] is True
    assert rows[0]["fuzzy_task"] == "q"
    assert rows[0]["checked_tools"][0]["name"] == "Search"
    assert rows[0]["label_match_status"] is None
    assert "timestamp" in rows[0]
    assert read_failure_attempt_count(failed_path, "rag-001") == 2
    assert read_failed_ids(failed_path) == {"rag-001"}


def test_write_exception_failure_records_entrypoint_exception(tmp_path):
    failed_path = tmp_path / "failed.jsonl"
    run_config = {
        "logging": {
            "task_file_path": str(tmp_path / "success.jsonl"),
            "failed_task_file_path": str(failed_path),
        }
    }

    record = write_exception_failure(
        {"id": "rag-err", "question": "q", "label": "a"},
        run_config,
        RuntimeError("upstream unavailable"),
    )

    rows = _read_jsonl(failed_path)
    assert record["stage"] == "entrypoint"
    assert rows[0]["failure_type"] == "exception"
    assert rows[0]["exception_class"] == "RuntimeError"
    assert rows[0]["failure_reason"] == "upstream unavailable"


def test_get_ids_to_skip_can_include_failed_ids(tmp_path):
    success_path = tmp_path / "success.jsonl"
    failed_path = tmp_path / "failed.jsonl"
    success_path.write_text('{"id": "done"}\n', encoding="utf-8")
    write_failure_record(
        failed_path,
        seed_info={"id": "failed", "question": "q", "label": "a"},
        final_state={},
        stage="graph",
        failure_type="generation_failed",
        failure_reason="failed before",
    )

    run_config = {
        "logging": {
            "task_file_path": str(success_path),
            "failed_task_file_path": str(failed_path),
        },
        "processing": {"retry_failed_tasks": False},
    }

    assert get_ids_to_skip(run_config) == {"done", "failed"}
    run_config["processing"]["retry_failed_tasks"] = True
    assert get_ids_to_skip(run_config) == {"done"}


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        test_write_failure_record_schema_and_attempt_index(tmp_path)
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        test_write_exception_failure_records_entrypoint_exception(tmp_path)
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        test_get_ids_to_skip_can_include_failed_ids(tmp_path)
    print("All failure logging tests passed")
