import os
import json
import glob
import re
import threading
import logging
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from tracesynth.configuration import ModelConfiguration, SynthesisComplexity, parse_range
from tracesynth.io import (
    validate_seed_info,
    extract_predicted_answer,
    check_label_match,
    write_failure_record,
)
from tracesynth.functions import (
    generate_tool_set, generate_fuzzy_task, tool_check,
    mock_tool_response, solve_task_by_tools,
)
from tracesynth.functions.call_llms import ParseError, call_and_parse
from tracesynth.functions.prompt import solve_task_user_prompt, solve_task_system_prompt

# Add a lock for thread-safe file writing
log_file_lock = threading.Lock()
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    seed_info: Dict[str, Any]  # To store original task information
    breaked: bool  # It will be set to False when any processing step fails

    initial_toolset_create: str
    initial_tools: str
    initial_task: str
    initial_workflow: str
    restrict: str

    fuzzy_task: str
    checked_tools: List[Dict[str, Any]]
    task_background: str

    plan: List[Dict[str, Any]]
    plan_evaluation: Dict[str, Any]
    plan_is_valid: bool
    plan_revision_count: int
    max_plan_revisions: int
    current_plan_step: int
    executed_steps: List[Dict[str, Any]]
    step_results: List[Dict[str, Any]]

    solve_history: List[Dict[str, Any]]
    active_plan_revision: int
    tool_call_history: List[str]
    current_tool_call: str
    task_finished: str
    failure_reason: str
    tool_call_retry_count: int
    solver_turn_count: int


def build_failure(reason: str, **extra: Any) -> Dict[str, Any]:
    payload = dict(extra)
    payload["breaked"] = True
    payload["task_finished"] = "Terminated"
    payload["failure_reason"] = reason
    return payload


def is_non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def has_final_answer(solve_history: Any) -> bool:
    if not isinstance(solve_history, list):
        return False
    return any(
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and re.search(r"<answer>.*?</answer>", message.get("content") or "", re.DOTALL | re.IGNORECASE)
        for message in solve_history
    )


def normalize_tool_for_solver(tool: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI-style function signatures should not include virtual return schemas."""
    normalized = dict(tool)
    normalized.pop("outputs", None)
    normalized.pop("returns", None)
    return normalized


def validate_tool_call(tool_call: str, checked_tools: List[Dict[str, Any]]) -> tuple[bool, str | None]:
    try:
        parsed = json.loads(tool_call)
    except json.JSONDecodeError as exc:
        return False, f"tool_call is not valid JSON: {exc}"

    if not isinstance(parsed, dict):
        return False, "tool_call must be a JSON object"
    tool_name = parsed.get("name")
    if not tool_name:
        return False, "tool_call is missing name"
    if tool_name not in {tool.get("name") for tool in checked_tools}:
        return False, f"tool_call references unknown tool: {tool_name}"
    if "arguments" not in parsed or not isinstance(parsed["arguments"], dict):
        return False, "tool_call.arguments must be an object"
    tool_schema = next((tool for tool in checked_tools if tool.get("name") == tool_name), {})
    required_args = (tool_schema.get("parameters") or {}).get("required") or []
    missing_args = [arg for arg in required_args if arg not in parsed["arguments"]]
    if missing_args:
        return False, f"tool_call.arguments missing required fields: {missing_args}"
    return True, None


def _extract_xml_json(content: str, tag: str) -> Any:
    matches = re.findall(rf"<{tag}>(.+?)</{tag}>", content or "", re.DOTALL)
    if not matches:
        raise ParseError(f"missing <{tag}> tag")
    raw_json = matches[-1].strip()
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON in <{tag}>: {exc}") from exc


def _parse_plan_response(content: str) -> List[Dict[str, Any]]:
    plan = _extract_xml_json(content, "plan")
    if not isinstance(plan, list) or not plan:
        raise ParseError("plan must be a non-empty JSON array")

    for index, step in enumerate(plan):
        if not isinstance(step, dict):
            raise ParseError(f"plan[{index}] must be an object")
        if not isinstance(step.get("tool_name"), str) or not step["tool_name"].strip():
            raise ParseError(f"plan[{index}] missing tool_name")
        if "arguments" not in step or not isinstance(step["arguments"], dict):
            raise ParseError(f"plan[{index}].arguments must be an object")
        step.setdefault("step_id", index + 1)
        step.setdefault("stage", "")
        step.setdefault("purpose", "")
        step.setdefault("depends_on", [])
    return plan


def _parse_plan_evaluation_response(content: str) -> Dict[str, Any]:
    evaluation = _extract_xml_json(content, "plan_evaluation")
    if not isinstance(evaluation, dict):
        raise ParseError("plan_evaluation must be a JSON object")
    if "is_valid" not in evaluation or not isinstance(evaluation["is_valid"], bool):
        raise ParseError("plan_evaluation.is_valid must be a boolean")
    if not (
        isinstance(evaluation.get("reasons"), list)
        or isinstance(evaluation.get("evaluation"), str)
        or isinstance(evaluation.get("reason"), str)
    ):
        raise ParseError("plan_evaluation must include concrete reasons")
    evaluation.setdefault("reasons", [])
    evaluation.setdefault("issues", [])
    evaluation.setdefault("revision_suggestions", [])
    return evaluation


def _tools_for_prompt(checked_tools: List[Dict[str, Any]]) -> str:
    return json.dumps(checked_tools, ensure_ascii=False, indent=2)


def _build_plan_messages(state: AgentState, complexity: SynthesisComplexity) -> List[Dict[str, str]]:
    prior_evaluation = state.get("plan_evaluation") or {}
    return [
        {
            "role": "system",
            "content": (
                "You are a planning agent for an Agentic RAG LangGraph pipeline. "
                "Create a complete tool-use trajectory before execution. "
                "Return only one <plan> XML block containing a JSON array."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query:\n{state['fuzzy_task']}\n\n"
                f"Task background:\n{state.get('task_background', '')}\n\n"
                f"High-level workflow:\n{state.get('initial_workflow', '')}\n\n"
                f"Policy/restrictions:\n{state.get('restrict', '')}\n\n"
                f"Complexity:\n{complexity.to_prompt_vars()['complexity_summary']}\n\n"
                f"Available tools JSON:\n{_tools_for_prompt(state['checked_tools'])}\n\n"
                f"Previous evaluation, if any:\n{json.dumps(prior_evaluation, ensure_ascii=False)}\n\n"
                "Plan requirements:\n"
                "1. Select only useful tools from the available tool list and avoid distractor tools.\n"
                "2. Cover Agentic RAG step2 query optimization, step3 retrieval, step4 post-processing, "
                "and step5 sufficiency/relevance evaluation whenever matching tools exist.\n"
                "3. Include explicit step dependencies and parameter sources.\n"
                "4. If iterative retrieval may be needed, include evaluation-driven follow-up steps within "
                "the bounded iteration requirement.\n"
                "5. Do not generate the final answer and do not call tools.\n\n"
                "Return format:\n"
                "<plan>\n"
                "[{\"step_id\":1,\"stage\":\"query_optimization\",\"tool_name\":\"ToolName\","
                "\"arguments\":{},\"purpose\":\"why this step is needed\","
                "\"depends_on\":[],\"parameter_sources\":{\"arg\":\"input or prior step\"}}]\n"
                "</plan>"
            ),
        },
    ]


def _build_plan_evaluation_messages(state: AgentState) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a strict evaluator for a planned Agentic RAG tool trajectory. "
                "Evaluate the plan before any execution. Return only one <plan_evaluation> XML block "
                "containing a JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query:\n{state['fuzzy_task']}\n\n"
                f"Policy/restrictions:\n{state.get('restrict', '')}\n\n"
                f"Available tools JSON:\n{_tools_for_prompt(state['checked_tools'])}\n\n"
                f"Plan JSON:\n{json.dumps(state.get('plan', []), ensure_ascii=False, indent=2)}\n\n"
                "Evaluate these dimensions: tool legality, required parameters, process coverage, "
                "dependency correctness, distractor-tool avoidance, iteration design, and policy compliance. "
                "Whether valid or invalid, give concrete reasons.\n\n"
                "Return format:\n"
                "<plan_evaluation>\n"
                "{\"is_valid\": true, \"reasons\": [\"...\"], \"issues\": [], "
                "\"revision_suggestions\": []}\n"
                "</plan_evaluation>"
            ),
        },
    ]


def is_successful_final_state(final_state: Dict[str, Any], strict=True) -> bool:
    if strict:
        return (
            not final_state.get("breaked")
            and isinstance(final_state.get("checked_tools"), list)
            and bool(final_state["checked_tools"])
            and has_final_answer(final_state.get("solve_history"))
        )
    else:
        return not final_state.get("breaked") and has_final_answer(final_state.get("solve_history"))


def create_step_config(
        base_config: RunnableConfig, step_name: str,
) -> RunnableConfig:
    """Create a new configuration for a specific step with its designated model"""
    # cfg = AgentConfiguration.from_runnable_config(base_config)
    step_models = base_config["configurable"]["step_models"]
    fallback_step = "Fallback" if step_name in {"PlanTrajectoryAgent", "EvaluatePlanAgent", "ExecutePlanAgent"} else step_name
    step_model_config = step_models.get(step_name) or step_models[fallback_step]

    # Create a new config with the specific model for this step
    step_config = {}
    if "configurable" not in step_config:
        step_config["configurable"] = {}

    # Apply the step-specific model configuration
    step_config["configurable"]["model_name"] = step_model_config["name"]
    if "temperature" in step_model_config:
        step_config["configurable"]["temperature"] = step_model_config["temperature"]
    if "max_tokens" in step_model_config:
        step_config["configurable"]["max_tokens"] = step_model_config["max_tokens"]
    if "use_tools" in step_model_config:
        step_config["configurable"]["use_tools"] = step_model_config["use_tools"]
    if "use_thinking" in step_model_config:
        step_config["configurable"]["use_thinking"] = step_model_config["use_thinking"]
    if "api_base" in step_model_config:
        step_config["configurable"]["api_base"] = step_model_config["api_base"]
    if "api_key_env" in step_model_config:
        step_config["configurable"]["api_key_env"] = step_model_config["api_key_env"]

    retry_cfg = base_config["configurable"].get("retry", {})
    for key in ("api_max_retries", "api_retry_base", "parse_max_retries", "tool_call_max_retries"):
        if key in retry_cfg:
            step_config["configurable"][key] = retry_cfg[key]

    return step_config


def get_tool_call_max_retries(config: RunnableConfig) -> int:
    retry_cfg = config.get("configurable", {}).get("retry", {})
    return int(retry_cfg.get("tool_call_max_retries", 3))


def get_plan_max_revisions(config: RunnableConfig) -> int:
    configurable = config.get("configurable", {}) if config else {}
    for value in (
        configurable.get("max_plan_revisions"),
        (configurable.get("processing") or {}).get("max_plan_revisions"),
        (configurable.get("planner") or {}).get("max_revisions"),
        (configurable.get("retry") or {}).get("max_plan_revisions"),
    ):
        if value is not None:
            return _coerce_positive_int(value, 3)
    return 3


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def get_solver_max_turns(config: RunnableConfig) -> int:
    configurable = config.get("configurable", {}) if config else {}

    for value in (
        configurable.get("max_solver_turns"),
        (configurable.get("processing") or {}).get("max_solver_turns"),
        (configurable.get("solver") or {}).get("max_turns"),
        (configurable.get("retry") or {}).get("max_solver_turns"),
    ):
        if value is not None:
            return _coerce_positive_int(value, 18)

    _, max_iterations = parse_range(get_synthesis_complexity(config).max_iterations)
    return max(12, 6 * (max_iterations + 1) + 6)


def get_graph_recursion_limit(config: RunnableConfig, max_solver_turns: int) -> int:
    configurable = config.get("configurable", {}) if config else {}
    configured = (
        configurable.get("recursion_limit")
        or (configurable.get("processing") or {}).get("graph_recursion_limit")
        or (configurable.get("processing") or {}).get("recursion_limit")
    )
    estimated = max(60, 3 + (2 * max_solver_turns) + 10)
    return max(_coerce_positive_int(configured, estimated), estimated)


def is_graph_recursion_error(exc: Exception) -> bool:
    text = str(exc)
    return (
        exc.__class__.__name__ == "GraphRecursionError"
        or (
            "Recursion limit" in text
            and "recursion_limit" in text
        )
    )


def use_label_as_answer(config: RunnableConfig) -> bool:
    eval_cfg = config.get("configurable", {}).get("evaluation") or {}
    return bool(eval_cfg.get("use_label_as_answer", True))


def get_synthesis_complexity(config: RunnableConfig) -> SynthesisComplexity:
    return SynthesisComplexity.from_run_config(config.get("configurable", {}))


def is_supervised_seed(seed_info: Dict[str, Any]) -> bool:
    return bool(seed_info.get("label") and seed_info.get("question"))


def toolset_gen_node(state: AgentState, config: RunnableConfig):
    logger.debug("------------------ToolSetGenAgent------------------")

    # Create step-specific configuration
    step_config = create_step_config(config, "ToolSetGenAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)
    complexity = get_synthesis_complexity(config)

    seed_info = state["seed_info"]
    background_info = seed_info.get("background") or seed_info.get("question", "")
    all_content, task, tools, workflow, restrict = generate_tool_set(
        cfg=cfg, background_info=background_info, complexity=complexity,
    )
    if not all(is_non_empty_text(value) for value in (all_content, task, tools, workflow, restrict)):
        return build_failure(
            "ToolSetGenAgent did not return all required sections",
            **{
                "initial_toolset_create": all_content,
                "initial_task": task,
                "initial_tools": tools,
                "initial_workflow": workflow,
                "restrict": restrict,
            },
        )

    return {
        "initial_toolset_create": all_content,
        "initial_task": task,
        "initial_tools": tools,
        "initial_workflow": workflow,
        "restrict": restrict
    }


def fuzzy_task_node(state: AgentState, config: RunnableConfig):
    logger.debug("------------------FuzzyTaskAgent------------------")
    if state["breaked"]:
        return {}

    seed_info = state["seed_info"]
    if is_supervised_seed(seed_info):
        fuzzy_task = seed_info["question"]
        task_background_parts = []
        if seed_info.get("context"):
            task_background_parts.append(seed_info["context"])
        step_config = create_step_config(config, "FuzzyTaskAgent")
        cfg = ModelConfiguration.from_runnable_config(step_config)
        initial_toolset_create = state["initial_toolset_create"]
        complexity = get_synthesis_complexity(config)
        _, generated_background = generate_fuzzy_task(
            cfg=cfg, initial_task_info=initial_toolset_create, complexity=complexity,
        )
        if is_non_empty_text(generated_background):
            task_background_parts.append(generated_background)
        task_background = "\n\n".join(task_background_parts).strip()
        if not is_non_empty_text(task_background):
            return build_failure(
                "FuzzyTaskAgent did not return task/background in supervised mode",
                **{
                    "fuzzy_task": fuzzy_task,
                    "task_background": task_background,
                },
            )
        return {
            "fuzzy_task": fuzzy_task,
            "task_background": task_background,
        }

    # Create step-specific configuration
    step_config = create_step_config(config, "FuzzyTaskAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    initial_toolset_create = state["initial_toolset_create"]
    complexity = get_synthesis_complexity(config)
    fuzzy_task, task_background = generate_fuzzy_task(
        cfg=cfg, initial_task_info=initial_toolset_create, complexity=complexity,
    )
    if not all(is_non_empty_text(value) for value in (fuzzy_task, task_background)):
        return build_failure(
            "FuzzyTaskAgent did not return task/background",
            **{
                "fuzzy_task": fuzzy_task,
                "task_background": task_background,
            },
        )

    return {
        "fuzzy_task": fuzzy_task,
        "task_background": task_background
    }


def check_tools_node(state: AgentState, config: RunnableConfig):
    logger.debug("------------------ToolCheckAgent------------------")

    if state["breaked"]:
        return {}

    # Create step-specific configuration
    step_config = create_step_config(config, "ToolCheckAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    initial_tools = state["initial_tools"]
    fuzzy_task = state["fuzzy_task"]
    complexity = get_synthesis_complexity(config)
    checked_tools = tool_check(cfg, initial_tools, fuzzy_task, complexity=complexity)

    if checked_tools is None:
        logger.warning("ToolCheckAgent returned invalid tools for task %s", fuzzy_task)
        return build_failure("ToolCheckAgent returned invalid JSON", **{"checked_tools": None})

    return {
        "checked_tools": checked_tools
    }


def plan_trajectory_node(state: AgentState, config: RunnableConfig):
    logger.debug("------------------PlanTrajectoryAgent------------------")

    if state["breaked"]:
        return {}

    step_config = create_step_config(config, "PlanTrajectoryAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)
    complexity = get_synthesis_complexity(config)

    revision_count = int(state.get("plan_revision_count", 0) or 0) + 1
    plan, _ = call_and_parse(
        cfg,
        _build_plan_messages(state, complexity),
        _parse_plan_response,
        step_name="PlanTrajectoryAgent",
    )
    if plan is None:
        return build_failure(
            "PlanTrajectoryAgent returned invalid plan JSON",
            **{
                "plan_revision_count": revision_count,
                "plan": state.get("plan", []),
                "plan_evaluation": state.get("plan_evaluation", {}),
            },
        )

    return {
        "plan": plan,
        "plan_revision_count": revision_count,
        "plan_is_valid": False,
        "current_plan_step": 0,
        "executed_steps": [],
        "step_results": [],
        "solve_history": [],
        "active_plan_revision": revision_count,
    }


def _basic_plan_validation(plan: List[Dict[str, Any]], checked_tools: List[Dict[str, Any]]) -> List[str]:
    issues = []
    tool_names = {tool.get("name") for tool in checked_tools}
    for index, step in enumerate(plan):
        tool_name = step.get("tool_name")
        if tool_name not in tool_names:
            issues.append(f"plan[{index}] references unknown tool: {tool_name}")
        tool_call = json.dumps(
            {"name": tool_name, "arguments": step.get("arguments", {})},
            ensure_ascii=False,
        )
        is_valid, error = validate_tool_call(tool_call, checked_tools)
        if not is_valid:
            issues.append(f"plan[{index}] invalid tool call: {error}")
    return issues


def evaluate_plan_node(state: AgentState, config: RunnableConfig):
    logger.debug("------------------EvaluatePlanAgent------------------")

    if state["breaked"]:
        return {}

    step_config = create_step_config(config, "EvaluatePlanAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)
    evaluation, _ = call_and_parse(
        cfg,
        _build_plan_evaluation_messages(state),
        _parse_plan_evaluation_response,
        step_name="EvaluatePlanAgent",
    )
    if evaluation is None:
        return build_failure(
            "EvaluatePlanAgent returned invalid evaluation JSON",
            **{
                "plan": state.get("plan", []),
                "plan_evaluation": state.get("plan_evaluation", {}),
            },
        )

    basic_issues = _basic_plan_validation(state.get("plan", []), state["checked_tools"])
    if basic_issues:
        evaluation["is_valid"] = False
        evaluation.setdefault("issues", [])
        evaluation["issues"].extend(basic_issues)
        evaluation.setdefault("reasons", [])
        evaluation["reasons"].append("Basic deterministic validation found invalid tool calls.")

    max_revisions = get_plan_max_revisions(config)
    if not evaluation["is_valid"] and int(state.get("plan_revision_count", 0) or 0) >= max_revisions:
        return build_failure(
            f"EvaluatePlanAgent rejected plan after max_plan_revisions={max_revisions}",
            **{
                "plan": state.get("plan", []),
                "plan_evaluation": evaluation,
                "plan_is_valid": False,
                "max_plan_revisions": max_revisions,
            },
        )

    return {
        "plan_is_valid": bool(evaluation["is_valid"]),
        "plan_evaluation": evaluation,
        "max_plan_revisions": max_revisions,
    }


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
    prompt += (
        "\n\n## Pre-approved execution plan\n"
        "The planner and evaluator have already selected the following trajectory. "
        "During execution, follow this plan and do not invent extra tool calls unless the plan is exhausted "
        "and the accumulated evidence is still insufficient.\n"
        f"{json.dumps(state.get('plan', []), ensure_ascii=False, indent=2)}"
    )
    if state.get("tool_call_history"):
        prompt += (
            "\n\n## Evidence already gathered before this plan revision\n"
            f"{json.dumps(state.get('tool_call_history', []), ensure_ascii=False, indent=2)}"
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def _format_planned_tool_message(step: Dict[str, Any], tool_call: str) -> str:
    return (
        f"Executing planned step {step.get('step_id')}: {step.get('purpose', '')}\n"
        f"Stage: {step.get('stage', '')}\n"
        f"<tool_call>{tool_call}</tool_call>"
    )


def _generate_final_answer_from_plan(state: AgentState, config: RunnableConfig, solver_turn_count: int) -> Dict[str, Any]:
    step_config = create_step_config(config, "ExecutePlanAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)
    solve_history = state.get("solve_history") or _initial_solve_history_from_plan(state, config)

    final_prompt = (
        "The planned tool trajectory has completed. Use only the accumulated tool responses and "
        "the task context to produce the final answer. Return the answer wrapped in <answer></answer>. "
        "If evidence is insufficient, briefly state the missing evidence instead of inventing facts."
    )
    solve_history.append({"role": "user", "content": final_prompt})

    one_step_think_and_tool_call, tool_call_info = solve_task_by_tools(cfg, solve_history)
    if not is_non_empty_text(one_step_think_and_tool_call):
        return build_failure(
            "SolveAgent returned empty final answer content",
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
                            "SolveAgent requested an additional tool call after executing all planned steps."
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
                    "SolveAgent requested an additional tool call after executing all planned steps."
                ],
                "revision_suggestions": [
                    "Revise the plan to include the missing evidence-gathering step before final answering."
                ],
            },
            "plan_is_valid": False,
        }

    return build_failure(
        "SolveAgent completed planned execution but did not produce <answer>",
        **{
            "solve_history": solve_history,
            "tool_call_history": state.get("tool_call_history", []),
            "solver_turn_count": solver_turn_count,
        },
    )


def execute_plan_node(state: AgentState, config: RunnableConfig):
    logger.debug("------------------ExecutePlanAgent------------------")

    if state["breaked"]:
        return {
            "current_tool_call": None,
            "task_finished": "Terminated"
        }

    solver_turn_count = int(state.get("solver_turn_count", 0) or 0) + 1
    max_solver_turns = get_solver_max_turns(config)
    if solver_turn_count > max_solver_turns:
        return build_failure(
            f"ExecutePlanAgent exceeded max_solver_turns={max_solver_turns} without producing <answer>",
            **{
                "solve_history": state.get("solve_history", []),
                "tool_call_history": state.get("tool_call_history", []),
                "solver_turn_count": solver_turn_count,
                "max_solver_turns": max_solver_turns,
            },
        )

    plan = state.get("plan") or []
    if not plan:
        return build_failure("ExecutePlanAgent cannot run without a non-empty plan")

    solve_history = state.get("solve_history") or _initial_solve_history_from_plan(state, config)
    current_plan_step = int(state.get("current_plan_step", 0) or 0)
    if current_plan_step >= len(plan):
        return _generate_final_answer_from_plan(state, config, solver_turn_count)

    step = plan[current_plan_step]
    tool_call_obj = {
        "name": step["tool_name"],
        "arguments": step.get("arguments", {}),
    }
    tool_call = json.dumps(tool_call_obj, ensure_ascii=False)
    is_valid, error = validate_tool_call(tool_call, state["checked_tools"])
    if not is_valid:
        return build_failure(
            error or "Invalid planned tool_call",
            **{
                "plan": plan,
                "current_plan_step": current_plan_step,
                "solve_history": solve_history,
                "tool_call_history": state.get("tool_call_history", []),
            },
        )

    solve_history.append({
        "role": "assistant",
        "content": _format_planned_tool_message(step, tool_call),
    })

    return {
        "current_tool_call": tool_call,
        "solve_history": solve_history,
        "task_finished": "Tool call",
        "solver_turn_count": solver_turn_count,
    }


def mock_tools_node(state: AgentState, config: RunnableConfig):
    logger.debug("------------------MockToolsAgent------------------")
    if state["breaked"]:
        return {}

    step_config = create_step_config(config, "MockToolAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    tool_call = state["current_tool_call"]
    tools_description = state["checked_tools"]
    tool_call_history = state["tool_call_history"]
    solve_history = state["solve_history"]

    tool_response, new_bg_introduced = mock_tool_response(
        cfg,
        tool_call,
        tools_description,
        tool_call_history,
        complexity=get_synthesis_complexity(config),
        label=state["seed_info"].get("label", ""),
        context=state["seed_info"].get("context", "") or "",
    )
    if tool_response is None:
        return build_failure(
            "MockToolAgent returned no tool response",
            **{
                "solve_history": solve_history,
                "tool_call_history": tool_call_history,
                "current_tool_call": tool_call,
            },
        )

    tool_response_message = {"role": "tool", "content": f"<tool_response>{tool_response}</tool_response>"}

    solve_history.append(tool_response_message)
    if new_bg_introduced:
        tool_call_history.append(f"Query:\n{tool_call}, Response:\n{tool_response}")

    update = {
        "tool_call_history": tool_call_history,
        "solve_history": solve_history
    }
    if state.get("plan"):
        current_plan_step = int(state.get("current_plan_step", 0) or 0)
        plan = state.get("plan", [])
        executed_steps = list(state.get("executed_steps") or [])
        step_results = list(state.get("step_results") or [])
        if 0 <= current_plan_step < len(plan):
            planned_step = plan[current_plan_step]
            executed_steps.append(planned_step)
            step_results.append({
                "step_id": planned_step.get("step_id", current_plan_step + 1),
                "tool_call": tool_call,
                "tool_response": tool_response,
                "new_bg_introduced": bool(new_bg_introduced),
            })
        update.update({
            "executed_steps": executed_steps,
            "step_results": step_results,
            "current_plan_step": current_plan_step + 1,
        })

    return update


def should_execute_or_replan(state: AgentState):
    if state.get("breaked"):
        return "end"
    if state.get("plan_is_valid"):
        return "execute"

    max_revisions = int(state.get("max_plan_revisions", 3) or 3)
    if int(state.get("plan_revision_count", 0) or 0) < max_revisions:
        return "replan"
    return "end"


def should_continue_execution(state: AgentState):
    if state.get("breaked") or state.get("task_finished") == "Terminated":
        return "end"
    if state.get("task_finished") == "Tool call":
        return "tool_call"
    if state.get("task_finished") == "Need replan":
        return "replan"
    return "end"


# Build the graph
builder = StateGraph(AgentState, config_schema=RunnableConfig)
builder.add_node("toolset_gen", toolset_gen_node)
builder.add_node("fuzzy_task", fuzzy_task_node)
builder.add_node("check_tools", check_tools_node)
builder.add_node("plan_trajectory", plan_trajectory_node)
builder.add_node("evaluate_plan", evaluate_plan_node)
builder.add_node("execute_plan", execute_plan_node)
builder.add_node("mock_tools", mock_tools_node)

builder.set_entry_point("toolset_gen")
builder.add_edge("toolset_gen", "fuzzy_task")
builder.add_edge("fuzzy_task", "check_tools")
builder.add_edge("check_tools", "plan_trajectory")
builder.add_edge("plan_trajectory", "evaluate_plan")
builder.add_conditional_edges(
    "evaluate_plan",
    should_execute_or_replan,
    # {"execute": "execute_plan", "replan": "plan_trajectory"}
    {"execute": "execute_plan", "replan": "plan_trajectory", "end": END}
)
builder.add_conditional_edges(
    "execute_plan",
    should_continue_execution,
    {"tool_call": "mock_tools", "replan": "plan_trajectory", "end": END},
)
builder.add_edge("mock_tools", "execute_plan")
graph = builder.compile()


def save_architecture_diagram(output_path: str) -> None:
    """Save the graph diagram only when explicitly requested."""
    img_bytes = graph.get_graph().draw_mermaid_png()
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(img_bytes)

# --- 运行入口 ---
def save_failure_artifacts(solve_path: str, final_state: Dict[str, Any]) -> None:
    """Persist failed state and any partial reasoning trajectory."""
    os.makedirs(solve_path, exist_ok=True)
    with open(os.path.join(solve_path, "failed_state.json"), 'w', encoding='utf-8') as f:
        f.write(json.dumps(final_state, ensure_ascii=False, indent=4) + '\n')

    solve_history = final_state.get("solve_history")
    if isinstance(solve_history, list):
        with open(os.path.join(solve_path, "failed_solution.json"), 'w', encoding='utf-8') as f:
            f.write(json.dumps(solve_history, ensure_ascii=False, indent=4) + '\n')

    tool_call_history = final_state.get("tool_call_history")
    if isinstance(tool_call_history, list):
        with open(os.path.join(solve_path, "tool_call_history.json"), 'w', encoding='utf-8') as f:
            f.write(json.dumps(tool_call_history, ensure_ascii=False, indent=4) + '\n')


def run_agent(seed_info: dict, run_config: dict = None):
    run_config = run_config or {}
    seed_info = validate_seed_info(seed_info)
    virtual_tool_use_task_path = run_config["logging"]["task_file_path"]
    failed_task_path = run_config["logging"].get(
        "failed_task_file_path",
        f"{virtual_tool_use_task_path}.failed",
    )
    solve_path = run_config["logging"]["solve_path"]
    eval_cfg = run_config.get("evaluation") or {}
    skip_label_match = bool(eval_cfg.get("skip_label_match", False))

    log_dir = os.path.dirname(virtual_tool_use_task_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    failed_log_dir = os.path.dirname(failed_task_path)
    if failed_log_dir and not os.path.exists(failed_log_dir):
        os.makedirs(failed_log_dir, exist_ok=True)

    solve_path = os.path.join(solve_path, f"{seed_info['id']}")
    if not os.path.exists(solve_path):
        os.makedirs(solve_path, exist_ok=True)

    run_config = {"configurable": run_config or {}}

    initial_state = {
        "seed_info": seed_info,
        "breaked": False,
        "task_finished": False,
        "failure_reason": "",
        "plan": [],
        "plan_evaluation": {},
        "plan_is_valid": False,
        "plan_revision_count": 0,
        "max_plan_revisions": get_plan_max_revisions(run_config),
        "current_plan_step": 0,
        "executed_steps": [],
        "step_results": [],
        "solve_history": [],
        "tool_call_history": [],
        "tool_call_retry_count": 0,
        "solver_turn_count": 0,
    }
    max_solver_turns = get_solver_max_turns(run_config)
    run_config["recursion_limit"] = get_graph_recursion_limit(run_config, max_solver_turns)
    try:
        final_state = graph.invoke(initial_state, config=run_config)
    except Exception as exc:
        if not is_graph_recursion_error(exc):
            raise
        final_state = build_failure(
            f"LangGraph recursion limit reached before stop condition: {exc}",
            **{
                **initial_state,
                "solver_turn_count": initial_state.get("solver_turn_count", 0),
                "max_solver_turns": max_solver_turns,
                "recursion_limit": run_config["recursion_limit"],
            },
        )

    if not is_successful_final_state(final_state):
        failure_reason = final_state.get("failure_reason") or "generation did not produce a valid final answer"
        failure_type = (
            "graph_recursion_limit"
            if "recursion limit" in failure_reason.lower()
            else "generation_failed"
        )
        with log_file_lock:
            write_failure_record(
                failed_task_path,
                seed_info=seed_info,
                final_state=final_state,
                stage="graph",
                failure_type=failure_type,
                failure_reason=failure_reason,
            )
            save_failure_artifacts(solve_path, final_state)
        return final_state

    predicted_answer = extract_predicted_answer(final_state.get("solve_history"))
    label_check = check_label_match(
        predicted_answer,
        seed_info.get("label", ""),
        skip=skip_label_match,
    )
    if label_check["label_match_status"] in {"mismatch", "missing_answer"}:
        failure_reason = (
            "predicted answer is missing"
            if label_check["label_match_status"] == "missing_answer"
            else "predicted answer does not match label"
        )
        with log_file_lock:
            write_failure_record(
                failed_task_path,
                seed_info=seed_info,
                final_state=final_state,
                stage="label_check",
                failure_type=label_check["label_match_status"],
                failure_reason=failure_reason,
                label_check=label_check,
            )
            failed_state = {**final_state, **label_check, "breaked": True, "failure_reason": failure_reason}
            save_failure_artifacts(solve_path, failed_state)
        return failed_state

    save_data = {
        "id": seed_info["id"],
        "question": seed_info.get("question"),
        "label": seed_info.get("label"),
        "context_present": bool(seed_info.get("context")),
        "fuzzy_task": final_state["fuzzy_task"],
        "checked_tools": final_state["checked_tools"],
        "plan": final_state.get("plan"),
        "plan_evaluation": final_state.get("plan_evaluation"),
        "artifact_dir": solve_path,
        "predicted_answer": predicted_answer,
        "label_match_status": label_check["label_match_status"],
        "match_score": label_check.get("match_score"),
    }

    with log_file_lock:
        solution_files = glob.glob(f"{solve_path}/solution*.json")
        existing_numbers = []
        for file in solution_files:
            basename = os.path.basename(file)
            match = re.match(r'solution(\d+)\.json$', basename)
            if match:
                existing_numbers.append(int(match.group(1)))

        next_number = max(existing_numbers) + 1 if existing_numbers else 1
        solution_filename = f"{solve_path}/solution{next_number}.json"
        save_data["solution_file"] = os.path.basename(solution_filename)

        with open(solution_filename, 'w', encoding='utf-8') as f:
            f.write(json.dumps(final_state["solve_history"], ensure_ascii=False, indent=4) + '\n')

        with open(f"{solve_path}/tool_call_history.json", 'w', encoding='utf-8') as f:
            f.write(json.dumps(final_state["tool_call_history"], ensure_ascii=False, indent=4) + '\n')

        more_info = {
            "question": seed_info.get("question"),
            "label": seed_info.get("label"),
            "context": seed_info.get("context"),
            "context_present": bool(seed_info.get("context")),
            "restrict": final_state["restrict"],
            "task_background": final_state["task_background"],
            "initial_workflow": final_state["initial_workflow"],
            "plan": final_state.get("plan", []),
            "plan_evaluation": final_state.get("plan_evaluation", {}),
            "executed_steps": final_state.get("executed_steps", []),
            "step_results": final_state.get("step_results", []),
            "predicted_answer": predicted_answer,
            "label_match_status": label_check["label_match_status"],
            "match_score": label_check.get("match_score"),
            "synthesis_complexity": SynthesisComplexity.from_run_config(
                run_config.get("configurable", {})
            ).model_dump(),
        }
        with open(f"{solve_path}/more_info.json", 'w', encoding='utf-8') as f:
            f.write(json.dumps(more_info, ensure_ascii=False, indent=4) + '\n')

        with open(virtual_tool_use_task_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(save_data, ensure_ascii=False) + '\n')

    return final_state


if __name__ == "__main__":
    import yaml

    with open("configs/tool_use_data_gen.yaml", 'r', encoding='utf-8') as f:
        agent_config = yaml.safe_load(f)

    with open("configs/seed_qa_sample.jsonl", 'r', encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    for task in tasks:
        run_agent(task, run_config=agent_config)
