import json
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracesynth.functions.evaluate_plan import (
    FAILURE_CATEGORY_MISSING_REQUIRED,
    FAILURE_CATEGORY_REVISION_EXHAUSTED,
    FAILURE_CATEGORY_SEMANTIC_REJECT,
    FAILURE_CATEGORY_UNKNOWN_TOOL,
    FAILURE_CATEGORY_VALID,
    PlanEvaluationStats,
    annotate_plan_evaluation,
    basic_plan_validation,
    build_preflight_evaluation,
    classify_validation_issues,
)
from tracesynth.graph.graph_virtual_tools import evaluate_plan_node


TOOLS = [
    {
        "name": "Query_Rewriter",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}},
        },
    },
    {
        "name": "Vector_Search",
        "parameters": {
            "type": "object",
            "required": ["queries"],
            "properties": {"queries": {"type": "array"}},
        },
    },
]


def _state_with_plan(plan):
    return {
        "breaked": False,
        "fuzzy_task": "What is gradient descent?",
        "restrict": "",
        "checked_tools": TOOLS,
        "plan": plan,
        "plan_revision_count": 1,
        "plan_evaluation": {},
    }


def test_classify_missing_required_issues():
    issues = ["plan[0] invalid tool call: tool_call.arguments missing required fields: ['query']"]
    assert classify_validation_issues(issues) == FAILURE_CATEGORY_MISSING_REQUIRED


def test_classify_unknown_tool_issues():
    issues = ["plan[1] references unknown tool: Missing_Tool"]
    assert classify_validation_issues(issues) == FAILURE_CATEGORY_UNKNOWN_TOOL


def test_preflight_evaluation_includes_actionable_suggestions():
    issues = ["plan[0] invalid tool call: tool_call.arguments missing required fields: ['query']"]
    evaluation = build_preflight_evaluation(issues)
    assert evaluation["is_valid"] is False
    assert evaluation["preflight_only"] is True
    assert evaluation["failure_category"] == FAILURE_CATEGORY_MISSING_REQUIRED
    assert evaluation["revision_suggestions"]


def test_basic_plan_validation_passes_with_required_arguments():
    plan = [
        {"tool_name": "Query_Rewriter", "arguments": {"query": "user query from task"}},
        {"tool_name": "Vector_Search", "arguments": {"queries": ["optimized query from step 1"]}},
    ]
    assert basic_plan_validation(plan, TOOLS) == []


def test_basic_plan_validation_fails_with_empty_arguments():
    plan = [{"tool_name": "Query_Rewriter", "arguments": {}}]
    issues = basic_plan_validation(plan, TOOLS)
    assert issues
    assert "missing required fields" in issues[0]


def test_annotate_semantic_reject_category():
    PlanEvaluationStats.reset()
    evaluation = annotate_plan_evaluation(
        {
            "is_valid": False,
            "reasons": ["missing retrieval stage"],
            "issues": [],
            "revision_suggestions": [],
        },
        [],
    )
    assert evaluation["failure_category"] == FAILURE_CATEGORY_SEMANTIC_REJECT


def test_annotate_revision_exhausted_category():
    evaluation = annotate_plan_evaluation(
        {
            "is_valid": False,
            "reasons": ["still incomplete"],
            "issues": [],
            "revision_suggestions": [],
        },
        [],
        revision_exhausted=True,
        record_stats=False,
    )
    assert evaluation["failure_category"] == FAILURE_CATEGORY_REVISION_EXHAUSTED


def test_evaluate_plan_node_preflight_skips_llm_call():
    PlanEvaluationStats.reset()
    state = _state_with_plan([{"tool_name": "Query_Rewriter", "arguments": {}}])
    config = {"configurable": {"processing": {"max_plan_revisions": 5}}}

    with patch("tracesynth.graph.graph_virtual_tools.call_and_parse") as mock_call:
        result = evaluate_plan_node(state, config)

    mock_call.assert_not_called()
    assert result["plan_is_valid"] is False
    assert result["plan_evaluation"]["preflight_only"] is True
    assert result["plan_evaluation"]["failure_category"] == FAILURE_CATEGORY_MISSING_REQUIRED


def test_evaluate_plan_node_llm_path_when_preflight_passes():
    PlanEvaluationStats.reset()
    plan = [{"tool_name": "Query_Rewriter", "arguments": {"query": "user query from task"}}]
    state = _state_with_plan(plan)
    config = {"configurable": {"processing": {"max_plan_revisions": 5}, "step_models": {}}}

    llm_evaluation = {
        "is_valid": True,
        "reasons": ["covers required stages"],
        "issues": [],
        "revision_suggestions": [],
    }

    with patch("tracesynth.graph.graph_virtual_tools.call_and_parse", return_value=(llm_evaluation, [])):
        with patch(
            "tracesynth.graph.graph_virtual_tools.create_step_config",
            return_value={"configurable": {}},
        ):
            with patch(
                "tracesynth.graph.graph_virtual_tools.ModelConfiguration.from_runnable_config",
                return_value=object(),
            ):
                result = evaluate_plan_node(state, config)

    assert result["plan_is_valid"] is True
    assert result["plan_evaluation"]["failure_category"] == FAILURE_CATEGORY_VALID


def test_simulated_pass_rate_improves_with_required_arguments():
    empty_args_plan = [{"tool_name": "Query_Rewriter", "arguments": {}}]
    filled_args_plan = [{"tool_name": "Query_Rewriter", "arguments": {"query": "user query from task"}}]

    empty_issues = basic_plan_validation(empty_args_plan, TOOLS)
    filled_issues = basic_plan_validation(filled_args_plan, TOOLS)

    assert empty_issues
    assert not filled_issues


def test_failure_record_includes_plan_eval_category(tmp_path):
    from tracesynth.io.samples import build_failure_record

    record = build_failure_record(
        seed_info={"id": "t1", "question": "q", "label": "a"},
        final_state={
            "plan_revision_count": 5,
            "max_plan_revisions": 5,
            "plan_evaluation": {
                "is_valid": False,
                "failure_category": FAILURE_CATEGORY_REVISION_EXHAUSTED,
                "issues": ["missing retrieval"],
            },
        },
        stage="graph",
        failure_type="generation_failed",
        failure_reason="EvaluatePlanAgent rejected plan after max_plan_revisions=5",
        attempt_index=1,
    )
    assert record["plan_eval_failure_category"] == FAILURE_CATEGORY_REVISION_EXHAUSTED
    assert record["plan_revision_count"] == 5


if __name__ == "__main__":
    import tempfile

    test_classify_missing_required_issues()
    test_classify_unknown_tool_issues()
    test_preflight_evaluation_includes_actionable_suggestions()
    test_basic_plan_validation_passes_with_required_arguments()
    test_basic_plan_validation_fails_with_empty_arguments()
    test_annotate_semantic_reject_category()
    test_annotate_revision_exhausted_category()
    test_evaluate_plan_node_preflight_skips_llm_call()
    test_evaluate_plan_node_llm_path_when_preflight_passes()
    test_simulated_pass_rate_improves_with_required_arguments()
    with tempfile.TemporaryDirectory() as directory:
        test_failure_record_includes_plan_eval_category(Path(directory))
    print("All evaluate plan validation tests passed")
