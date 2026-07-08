import json
import logging
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig

from tracesynth.configuration import ModelConfiguration
from tracesynth.functions.call_llms import ParseError, parse_json_object
from tracesynth.functions import solve_task_by_tools
from tracesynth.functions.prompt import (
    execute_plan_evidence_section_prompt,
    execute_plan_final_answer_prompt,
    execute_plan_preapproved_prompt,
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
from tracesynth.io import check_label_match

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
        # 重规划后把既有工具证据注入新执行上下文，避免重复查询同一批信息。
        evidence_section = execute_plan_evidence_section_prompt.format(
            evidence_json=json.dumps(state.get("tool_call_history", []), ensure_ascii=False, indent=2)
        )
    # 计划已经过 Planner/Evaluator 批准，执行阶段只按该计划推进。
    prompt += "\n\n" + execute_plan_preapproved_prompt.format(
        plan_json=json.dumps(state.get("plan", []), ensure_ascii=False, indent=2),
        evidence_section=evidence_section,
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def _format_planned_tool_message(step: Dict[str, Any], tool_call: str) -> str:
    try:
        parsed_tool_call: Any = json.loads(tool_call)
    except (TypeError, json.JSONDecodeError):
        parsed_tool_call = tool_call
    return json.dumps(
        {
            "action": "tool_call",
            "reasoning": (
                f"Executing planned step {step.get('step_id')}: "
                f"{step.get('purpose', '')} Stage: {step.get('stage', '')}"
            ),
            "tool_call": parsed_tool_call,
        },
        ensure_ascii=False,
    )


def _insufficient_evidence_outcome(
    state: AgentState,
    config: RunnableConfig,
    solve_history: List[Dict[str, Any]],
    solver_turn_count: int,
    *,
    extra_reason: str = "",
    missing_evidence_message: str = "",
) -> Dict[str, Any]:
    """证据不足时：未超修订上限则重规划，否则判定失败。"""
    max_revisions = get_plan_max_revisions(config)
    revision_count = int(state.get("plan_revision_count", 0) or 0)
    reasons = ["The completed plan did not provide enough evidence for the final answer."]
    if extra_reason:
        reasons.append(extra_reason)
    if missing_evidence_message:
        reasons.append(f"Final Response reported missing evidence: {missing_evidence_message}")
    revision_suggestions = [
        "Revise the plan to include the missing evidence-gathering step before final answering."
    ]
    if missing_evidence_message:
        revision_suggestions.append(
            f"Add or revise tool steps to collect this missing evidence: {missing_evidence_message}"
        )
    plan_evaluation = {
        "is_valid": False,
        "reasons": reasons,
        "issues": [
            "Final Response requested an additional tool call after executing all planned steps."
        ],
        "revision_suggestions": revision_suggestions,
    }
    if missing_evidence_message:
        plan_evaluation["missing_evidence"] = missing_evidence_message
    if revision_count >= max_revisions:
        return build_failure(
            "plan exhausted but evidence still insufficient after max revisions",
            **{
                "solve_history": solve_history,
                "tool_call_history": state.get("tool_call_history", []),
                "solver_turn_count": solver_turn_count,
                "plan_revision_count": revision_count,
                "max_plan_revisions": max_revisions,
                "plan_evaluation": plan_evaluation,
                "plan_is_valid": False,
            },
        )
    return {
        "current_tool_call": None,
        "solve_history": solve_history,
        "task_finished": "Need replan",
        "solver_turn_count": solver_turn_count,
        "plan_evaluation": plan_evaluation,
        "plan_is_valid": False,
    }


def _generate_final_answer_from_plan(state: AgentState, config: RunnableConfig, solver_turn_count: int) -> Dict[str, Any]:
    step_config = create_step_config(config, "ExecutePlanAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)
    solve_history = list(state.get("solve_history") or _initial_solve_history_from_plan(state, config))
    solve_history = solve_history + [{"role": "user", "content": execute_plan_final_answer_prompt}]

    # 先让模型基于已收集证据自行判断能否给出终答，不再提前用金标短路。
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

    solve_history = solve_history + [{"role": "assistant", "content": one_step_think_and_tool_call}]
    try:
        action_payload = parse_json_object(one_step_think_and_tool_call)
    except ParseError as exc:
        return build_failure(
            f"Final Response returned invalid JSON after parser success: {exc}",
            **{
                "solve_history": solve_history,
                "tool_call_history": state.get("tool_call_history", []),
                "solver_turn_count": solver_turn_count,
            },
        )

    if action_payload.get("action") == "final_answer":
        model_answer = str(action_payload.get("answer", "")).strip()
        if use_label_as_answer(config):
            label = (state["seed_info"].get("label") or "").strip()
            if label:
                match_result = check_label_match(model_answer, label)
                if match_result["label_match_status"] != "match":
                    return _insufficient_evidence_outcome(
                        state,
                        config,
                        solve_history,
                        solver_turn_count,
                        extra_reason=(
                            f"Model's own final answer ('{model_answer}') does not match "
                            f"the supervised label; treating as insufficient/incorrect evidence chain."
                        ),
                    )
                solve_history[-1] = {
                    "role": "assistant",
                    "content": json.dumps(
                        {"action": "final_answer", "answer": label},
                        ensure_ascii=False,
                    ),
                }
        return {
            "current_tool_call": None,
            "solve_history": solve_history,
            "task_finished": "Terminated",
            "solver_turn_count": solver_turn_count,
        }

    if action_payload.get("action") in {"tool_call", "ask_user"} or tool_call_info is not None:
        missing_evidence_message = ""
        if action_payload.get("action") == "ask_user":
            missing_evidence_message = str(action_payload.get("message", "")).strip()
        return _insufficient_evidence_outcome(
            state,
            config,
            solve_history,
            solver_turn_count,
            extra_reason=f"Final Response returned action={action_payload.get('action')!r} instead of final_answer.",
            missing_evidence_message=missing_evidence_message,
        )

    return build_failure(
        "Final Response completed planned execution but did not produce final_answer action",
        **{
            "solve_history": solve_history,
            "tool_call_history": state.get("tool_call_history", []),
            "solver_turn_count": solver_turn_count,
        },
    )


initial_solve_history_from_plan = _initial_solve_history_from_plan
format_planned_tool_message = _format_planned_tool_message
generate_final_answer_from_plan = _generate_final_answer_from_plan
