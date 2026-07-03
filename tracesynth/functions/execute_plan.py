import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig

from tracesynth.configuration import ModelConfiguration
from tracesynth.functions import solve_task_by_tools
from tracesynth.functions.prompt import (
    execute_plan_evidence_section_prompt,
    execute_plan_final_answer_prompt,
    execute_plan_preapproved_prompt,
    planned_tool_message_template,
    solve_task_system_prompt,
    solve_task_user_prompt,
)
from tracesynth.graph.node_utils import (
    AgentState,
    build_failure,
    create_step_config,
    get_plan_max_revisions,
    get_synthesis_complexity,
    is_non_empty_text,
    normalize_tool_for_solver,
    use_label_as_answer,
)

logger = logging.getLogger(__name__)


def _initial_solve_history_from_plan(state: AgentState, config: RunnableConfig) -> List[Dict[str, Any]]:
    checked_tools = state["checked_tools"]
    task_info = state["fuzzy_task"]
    restrict = state.get("restrict", "")
    complexity = get_synthesis_complexity(config)

    tools_description = ""
    for tool in checked_tools:
        tools_description += json.dumps(
            {"type": "function", "function": normalize_tool_for_solver(tool)},
            ensure_ascii=False,
        ) + "\n"

    system_prompt = solve_task_system_prompt.format(available_tools=tools_description, restrict=restrict)
    prompt = solve_task_user_prompt.format(
        task_info=task_info,
        **complexity.to_prompt_vars(),
    )
    evidence_section = ""
    if state.get("tool_call_history"):
        evidence_section = execute_plan_evidence_section_prompt.format(
            evidence_json=json.dumps(state.get("tool_call_history", []), ensure_ascii=False, indent=2)
        )
    prompt += "\n\n" + execute_plan_preapproved_prompt.format(
        plan_json=json.dumps(state.get("plan", []), ensure_ascii=False, indent=2),
        evidence_section=evidence_section,
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def _format_planned_tool_message(step: Dict[str, Any], tool_call: str) -> str:
    return planned_tool_message_template.format(
        step_id=step.get("step_id"),
        purpose=step.get("purpose", ""),
        stage=step.get("stage", ""),
        tool_call=tool_call,
    )


def _generate_final_answer_from_plan(state: AgentState, config: RunnableConfig, solver_turn_count: int) -> Dict[str, Any]:
    step_config = create_step_config(config, "ExecutePlanAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)
    solve_history = state.get("solve_history") or _initial_solve_history_from_plan(state, config)

    solve_history.append({"role": "user", "content": execute_plan_final_answer_prompt})

    one_step_think_and_tool_call, tool_call_info = solve_task_by_tools(cfg, solve_history)
    if not is_non_empty_text(one_step_think_and_tool_call):
        return build_failure(
            "Final Response returned empty final answer content",
            **{
                "solve_history": solve_history,
                "tool_call_history": state.get("tool_call_history", []),
                "solver_turn_count": solver_turn_count,
            },
        )

    solve_history.append({"role": "assistant", "content": one_step_think_and_tool_call})
    if re.search(r"<answer>.*?</answer>", one_step_think_and_tool_call, re.DOTALL | re.IGNORECASE):
        if use_label_as_answer(config):
            label = (state["seed_info"].get("label") or "").strip()
            if label:
                solve_history[-1] = {
                    "role": "assistant",
                    "content": f"<answer>{label}</answer>",
                }
        return {
            "current_tool_call": None,
            "solve_history": solve_history,
            "task_finished": "Terminated",
            "solver_turn_count": solver_turn_count,
        }

    if tool_call_info is not None:
        max_revisions = get_plan_max_revisions(config)
        revision_count = int(state.get("plan_revision_count", 0) or 0)
        if revision_count >= max_revisions:
            return build_failure(
                "plan exhausted but evidence still insufficient after max revisions",
                **{
                    "solve_history": solve_history,
                    "tool_call_history": state.get("tool_call_history", []),
                    "solver_turn_count": solver_turn_count,
                    "plan_revision_count": revision_count,
                    "max_plan_revisions": max_revisions,
                    "plan_evaluation": {
                        "is_valid": False,
                        "reasons": [
                            "The completed plan did not provide enough evidence for the final answer."
                        ],
                        "issues": [
                            "Final Response requested an additional tool call after executing all planned steps."
                        ],
                        "revision_suggestions": [
                            "Revise the plan to include the missing evidence-gathering step before final answering."
                        ],
                    },
                    "plan_is_valid": False,
                },
            )
        return {
            "current_tool_call": None,
            "solve_history": solve_history,
            "task_finished": "Need replan",
            "solver_turn_count": solver_turn_count,
            "plan_evaluation": {
                "is_valid": False,
                "reasons": [
                    "The completed plan did not provide enough evidence for the final answer."
                ],
                "issues": [
                    "Final Response requested an additional tool call after executing all planned steps."
                ],
                "revision_suggestions": [
                    "Revise the plan to include the missing evidence-gathering step before final answering."
                ],
            },
            "plan_is_valid": False,
        }

    return build_failure(
        "Final Response completed planned execution but did not produce <answer>",
        **{
            "solve_history": solve_history,
            "tool_call_history": state.get("tool_call_history", []),
            "solver_turn_count": solver_turn_count,
        },
    )


initial_solve_history_from_plan = _initial_solve_history_from_plan
format_planned_tool_message = _format_planned_tool_message
generate_final_answer_from_plan = _generate_final_answer_from_plan
