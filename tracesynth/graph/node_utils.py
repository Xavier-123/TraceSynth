import json
import re
from typing import Any, Dict, List, TypedDict

from langchain_core.runnables import RunnableConfig

from tracesynth.configuration import SynthesisComplexity, parse_range


class AgentState(TypedDict):
    # 原始种子与中断标记，所有图节点都会依赖这两个字段判断是否继续推进。
    seed_info: Dict[str, Any]
    breaked: bool

    # ToolSetGenAgent 产出的初始任务、工具、工作流和约束，供后续模糊化与规划使用。
    initial_toolset_create: str
    initial_tools: str
    initial_task: str
    initial_workflow: str
    restrict: str

    # FuzzyTaskAgent 与 ToolCheckAgent 产物：最终给 Solver 的模糊任务、可用工具和背景信息。
    fuzzy_task: str
    checked_tools: List[Dict[str, Any]]
    task_background: str

    # Plan-Execute 流水线状态：计划、评估结果、当前执行进度和每步工具返回。
    plan: List[Dict[str, Any]]
    plan_evaluation: Dict[str, Any]
    plan_is_valid: bool
    plan_revision_count: int
    max_plan_revisions: int
    current_plan_step: int
    executed_steps: List[Dict[str, Any]]
    step_results: List[Dict[str, Any]]

    # Reason-Act / ExecutePlan 共用的求解轨迹与工具调用记忆。
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
    # 先校验 JSON 结构，再校验工具名和 required 参数，避免 MockToolAgent 执行不存在的虚拟工具。
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
    # 只检查 required 字段是否存在；参数类型和语义仍交给提示词与 MockToolAgent 约束。
    missing_args = [arg for arg in required_args if arg not in parsed["arguments"]]
    if missing_args:
        return False, f"tool_call.arguments missing required fields: {missing_args}"
    return True, None


def is_successful_final_state(final_state: Dict[str, Any], strict=True) -> bool:
    if strict:
        # 严格模式用于落盘前验收：必须未中断、工具集有效，并且轨迹里真的出现最终答案标签。
        return (
            not final_state.get("breaked")
            and isinstance(final_state.get("checked_tools"), list)
            and bool(final_state["checked_tools"])
            and has_final_answer(final_state.get("solve_history"))
        )
    return not final_state.get("breaked") and has_final_answer(final_state.get("solve_history"))


def create_step_config(
        base_config: RunnableConfig, step_name: str,
) -> RunnableConfig:
    """Create a new configuration for a specific step with its designated model."""
    step_models = base_config["configurable"]["step_models"]
    # 规划/评估/执行 Agent 优先共用求解模型作为兜底，其他节点优先找自身配置再走通用回退。
    fallback_names = (
        ("FallbackModel", "Fallback", "SolveAgent")
        if step_name in {"PlanTrajectoryAgent", "EvaluatePlanAgent", "ExecutePlanAgent"}
        else (step_name, "FallbackModel", "Fallback")
    )
    step_model_config = step_models.get(step_name)
    for fallback_name in fallback_names:
        step_model_config = step_model_config or step_models.get(fallback_name)
    if step_model_config is None:
        raise KeyError(f"No model configuration found for step '{step_name}'")

    step_config = {"configurable": {}}
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
    # 重试配置是跨 Agent 的运行策略，需要透传到每个步骤的 ModelConfiguration。
    for key in ("api_max_retries", "api_retry_base", "parse_max_retries", "tool_call_max_retries"):
        if key in retry_cfg:
            step_config["configurable"][key] = retry_cfg[key]

    return step_config


def get_tool_call_max_retries(config: RunnableConfig) -> int:
    retry_cfg = config.get("configurable", {}).get("retry", {})
    return int(retry_cfg.get("tool_call_max_retries", 3))


def get_plan_max_revisions(config: RunnableConfig) -> int:
    configurable = config.get("configurable", {}) if config else {}
    # 兼容不同 YAML 层级的历史配置，取到第一个有效值后统一转成正整数。
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
    # 默认轮次随任务迭代复杂度增长，给每轮“执行工具 + 回到 Solver”预留足够图节点步数。
    return max(12, 6 * (max_iterations + 1) + 6)


def get_graph_recursion_limit(config: RunnableConfig, max_solver_turns: int) -> int:
    configurable = config.get("configurable", {}) if config else {}
    configured = (
        configurable.get("recursion_limit")
        or (configurable.get("processing") or {}).get("graph_recursion_limit")
        or (configurable.get("processing") or {}).get("recursion_limit")
    )
    # LangGraph 的递归上限按最大 Solver 回合估算，避免正常工具循环被框架提前截断。
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
    # 合成数据默认用金标替换最终答案，保证轨迹可评测；可在配置中关闭以保留模型原答案。
    return bool(eval_cfg.get("use_label_as_answer", True))


def get_synthesis_complexity(config: RunnableConfig) -> SynthesisComplexity:
    return SynthesisComplexity.from_run_config(config.get("configurable", {}))
