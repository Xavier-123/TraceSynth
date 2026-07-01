import json
from pathlib import Path
import sys
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracesynth.graph.graph_virtual_tools import run_agent


def _config(tmp_path: Path):
    return {
        "logging": {
            "task_file_path": str(tmp_path / "virtual_tool_use.jsonl"),
            "failed_task_file_path": str(tmp_path / "virtual_tool_use.failed.jsonl"),
            "solve_path": str(tmp_path / "solve_tool_use"),
        },
        "evaluation": {"skip_label_match": False, "use_label_as_answer": False},
        "synthesis": {
            "task_complexity": {
                "num_tools": "4",
                "num_custom_tools": "1",
                "distractor_tools": "1",
            },
            "iteration_complexity": {"max_iterations": "1"},
        },
    }


def _seed():
    return {
        "id": "rag-fail",
        "question": "q",
        "label": "gold",
        "context": "ctx",
    }


def _read_failed_rows(tmp_path: Path):
    failed_path = tmp_path / "virtual_tool_use.failed.jsonl"
    with open(failed_path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_run_agent_graph_failure_writes_structured_failure_and_state(tmp_path):
    final_state = {
        "seed_info": _seed(),
        "breaked": True,
        "task_finished": "Terminated",
        "failure_reason": "ToolCheckAgent returned invalid JSON",
        "fuzzy_task": "q",
        "checked_tools": None,
        "solve_history": [],
    }

    with patch("tracesynth.graph.graph_virtual_tools.graph.invoke", return_value=final_state):
        result = run_agent(_seed(), _config(tmp_path))

    rows = _read_failed_rows(tmp_path)
    failed_state_path = tmp_path / "solve_tool_use" / "rag-fail" / "failed_state.json"
    assert result is final_state
    assert rows[0]["stage"] == "graph"
    assert rows[0]["failure_type"] == "generation_failed"
    assert rows[0]["failure_reason"] == "ToolCheckAgent returned invalid JSON"
    assert rows[0]["attempt_index"] == 1
    assert failed_state_path.exists()


def test_run_agent_graph_failure_writes_partial_trajectory(tmp_path):
    final_state = {
        "seed_info": _seed(),
        "breaked": True,
        "task_finished": "Terminated",
        "failure_reason": "SolveAgent exceeded max_solver_turns=1 without producing <answer>",
        "fuzzy_task": "q",
        "checked_tools": [{"name": "Search", "parameters": {}}],
        "solve_history": [{"role": "assistant", "content": "partial reasoning"}],
        "tool_call_history": ["Query: Search, Response: partial"],
    }

    with patch("tracesynth.graph.graph_virtual_tools.graph.invoke", return_value=final_state):
        run_agent(_seed(), _config(tmp_path))

    artifact_dir = tmp_path / "solve_tool_use" / "rag-fail"
    failed_solution_path = artifact_dir / "failed_solution.json"
    tool_history_path = artifact_dir / "tool_call_history.json"

    assert failed_solution_path.exists()
    assert tool_history_path.exists()
    with open(failed_solution_path, "r", encoding="utf-8") as handle:
        assert json.load(handle) == final_state["solve_history"]
    with open(tool_history_path, "r", encoding="utf-8") as handle:
        assert json.load(handle) == final_state["tool_call_history"]


def test_run_agent_missing_answer_label_check_is_failure(tmp_path):
    final_state = {
        "seed_info": _seed(),
        "breaked": False,
        "task_finished": "Terminated",
        "failure_reason": "",
        "fuzzy_task": "q",
        "checked_tools": [{"name": "Search", "parameters": {}}],
        "solve_history": [{"role": "assistant", "content": "<answer>   </answer>"}],
        "tool_call_history": [],
        "restrict": "",
        "task_background": "",
        "initial_workflow": "",
    }

    with patch("tracesynth.graph.graph_virtual_tools.graph.invoke", return_value=final_state):
        result = run_agent(_seed(), _config(tmp_path))

    rows = _read_failed_rows(tmp_path)
    assert result["breaked"] is True
    assert result["failure_reason"] == "predicted answer is missing"
    assert rows[0]["stage"] == "label_check"
    assert rows[0]["failure_type"] == "missing_answer"
    assert rows[0]["label_match_status"] == "missing_answer"
    assert rows[0]["match_score"] == 0.0


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_run_agent_graph_failure_writes_structured_failure_and_state(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_run_agent_missing_answer_label_check_is_failure(Path(directory))
    print("All graph failure output tests passed")
