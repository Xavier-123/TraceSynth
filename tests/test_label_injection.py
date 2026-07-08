"""Tests for injecting dataset label as final answer in plan execution."""

import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TEST_KEY", "test-api-key")

from tracesynth.graph.graph_virtual_tools import (
    _generate_final_answer_from_plan,
)
from tracesynth.graph.node_utils import use_label_as_answer
from tracesynth.io import extract_predicted_answer, check_label_match


def _base_state(label: str = "标准答案内容"):
    return {
        "breaked": False,
        "seed_info": {
            "id": "rag-001-bak",
            "question": "测试问题",
            "label": label,
            "context": "上下文",
        },
        "checked_tools": [
            {
                "name": "BM25Recall",
                "description": "test",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        "fuzzy_task": "测试问题",
        "restrict": "无额外约束",
        "solve_history": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "task prompt"},
        ],
        "tool_call_retry_count": 0,
        "plan_revision_count": 1,
    }


def _config(use_label: bool = True):
    return {
        "configurable": {
            "step_models": {
                "SolveAgent": {
                    "name": "test-model",
                    "api_base": "http://localhost",
                    "api_key_env": "TEST_KEY",
                }
            },
            "retry": {"tool_call_max_retries": 3},
            "evaluation": {"use_label_as_answer": use_label},
            "synthesis": {
                "task_complexity": {"num_tools": "4"},
                "iteration_complexity": {"max_iterations": "1"},
            },
        }
    }


def test_use_label_as_answer_default_true():
    assert use_label_as_answer({"configurable": {}}) is True
    assert use_label_as_answer(_config(True)) is True
    assert use_label_as_answer(_config(False)) is False


@patch("tracesynth.functions.execute_plan.solve_task_by_tools")
def test_generate_final_answer_injects_label_on_termination(mock_solve):
    mock_solve.return_value = (
        '{"action":"final_answer","answer":"数据集标准答案"}',
        None,
    )
    state = _base_state(label="数据集标准答案")
    result = _generate_final_answer_from_plan(state, _config(use_label=True), solver_turn_count=1)

    assert result["task_finished"] == "Terminated"
    assert result["solve_history"][-1] == {
        "role": "assistant",
        "content": json.dumps(
            {"action": "final_answer", "answer": "数据集标准答案"},
            ensure_ascii=False,
        ),
    }
    assert extract_predicted_answer(result["solve_history"]) == "数据集标准答案"
    match = check_label_match(
        extract_predicted_answer(result["solve_history"]),
        state["seed_info"]["label"],
    )
    assert match["label_match_status"] == "match"
    assert match["match_score"] == 1.0


@patch("tracesynth.functions.execute_plan.solve_task_by_tools")
def test_generate_final_answer_keeps_model_answer_when_flag_disabled(mock_solve):
    model_answer = '{"action":"final_answer","answer":"模型答案"}'
    mock_solve.return_value = (model_answer, None)
    state = _base_state(label="数据集标准答案")
    result = _generate_final_answer_from_plan(state, _config(use_label=False), solver_turn_count=1)

    assert result["solve_history"][-1]["content"] == model_answer
    assert extract_predicted_answer(result["solve_history"]) == "模型答案"


@patch("tracesynth.functions.execute_plan.solve_task_by_tools")
def test_generate_final_answer_falls_back_when_label_empty(mock_solve):
    model_answer = '{"action":"final_answer","answer":"模型答案"}'
    mock_solve.return_value = (model_answer, None)
    state = _base_state(label="")
    result = _generate_final_answer_from_plan(state, _config(use_label=True), solver_turn_count=1)

    assert result["solve_history"][-1]["content"] == model_answer


if __name__ == "__main__":
    test_use_label_as_answer_default_true()
    test_generate_final_answer_injects_label_on_termination()
    test_generate_final_answer_keeps_model_answer_when_flag_disabled()
    test_generate_final_answer_falls_back_when_label_empty()
    print("All label injection tests passed")
