"""Smoke tests for 6-step Agentic RAG prompt placeholders."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracesynth.configuration import SynthesisComplexity
from tracesynth.functions import prompt


def _base_vars():
    complexity = SynthesisComplexity()
    vars_ = complexity.to_prompt_vars()
    vars_.update({
        "background_info": "test background",
        "initial_task_info": "test task info",
        "task_description": "test task",
        "tool_description": "[]",
        "task": "user task",
        "background": "user background",
        "restrict": "no restriction",
        "interaction": "[]",
        "tools": "[]",
        "world_state": "[]",
        "query": "{}",
        "label": "gold answer",
        "context": "reference context",
        "available_tools": "[]",
        "task_info": "test query",
        "task_background": "bg",
        "high_level_workflow": "workflow",
        "restrict_policy": "policy",
        "trajectories_summary": "traj",
    })
    return vars_


def test_to_prompt_vars_keys():
    vars_ = SynthesisComplexity().to_prompt_vars()
    expected = {
        "num_tools", "num_custom_tools", "distractor_tools",
        "max_iterations", "min_iterations", "max_iterations_val",
        "iteration_requirement", "complexity_summary",
    }
    assert expected.issubset(vars_.keys())
    assert "num_tool_categories" not in vars_
    assert "retrieval_rounds" not in vars_


def test_all_prompts_format():
    vars_ = _base_vars()
    prompts = [
        prompt.tool_set_prompt,
        prompt.fuzzy_task_prompt,
        prompt.tool_check_prompt,
        prompt.mock_user_prompt,
        prompt.tool_simulation_prompt_with_memory,
        prompt.solve_task_system_prompt,
        prompt.solve_task_user_prompt,
        prompt.rubric_user_prompt,
    ]
    for name, text in [
        ("tool_set_prompt", prompt.tool_set_prompt),
        ("fuzzy_task_prompt", prompt.fuzzy_task_prompt),
        ("tool_check_prompt", prompt.tool_check_prompt),
        ("mock_user_prompt", prompt.mock_user_prompt),
        ("tool_simulation_prompt_with_memory", prompt.tool_simulation_prompt_with_memory),
        ("solve_task_system_prompt", prompt.solve_task_system_prompt),
        ("solve_task_user_prompt", prompt.solve_task_user_prompt),
        ("rubric_user_prompt", prompt.rubric_user_prompt),
    ]:
        text.format(**vars_)


def test_from_run_config_legacy():
    cfg = SynthesisComplexity.from_run_config({
        "synthesis": {
            "task_complexity": {"num_tools": "5"},
            "iteration_complexity": {"retrieval_rounds": "2~3"},
        }
    })
    assert cfg.num_tools == "5"
    assert cfg.max_iterations == "2~3"


if __name__ == "__main__":
    test_to_prompt_vars_keys()
    test_all_prompts_format()
    test_from_run_config_legacy()
    import tracesynth.graph.graph_virtual_tools  # noqa: F401
    print("All 6-step smoke tests passed")
