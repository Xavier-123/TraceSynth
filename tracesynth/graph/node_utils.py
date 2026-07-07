import glob
import json
import os
import re
from typing import Any, Dict, List, TypedDict

from langchain_core.runnables import RunnableConfig

from tracesynth.configuration import ModelConfiguration, SynthesisComplexity, parse_range
from tracesynth.functions import mock_tool_response


class AgentState(TypedDict):
    # 当前样本的原始输入信息；通常包含 id、question、label、context/background，
    # 是工具集生成、监督答案校验、MockTool 构造虚拟知识库的根数据。
    seed_info: Dict[str, Any]

    # 图级中断标记；任一节点调用 build_failure 后置为 True，
    # 后续节点/路由据此停止继续生成并进入失败落盘流程。
    breaked: bool

    # ToolSetGenAgent 的完整原始输出，去除了 reasoning 后保留 task/tools/workflow/restriction 等区块。
    initial_toolset_create: str

    # ToolSetGenAgent 设计的初始虚拟工具描述文本；会交给 ToolCheckAgent 校验并转成可执行 schema。
    initial_tools: str

    # ToolSetGenAgent 从种子背景中抽象出的初始任务描述；主要用于审计和调试，后续求解使用 fuzzy_task。
    initial_task: str

    # ToolSetGenAgent 设计的高层 RAG 工作流说明；会注入 Planner，帮助计划对齐 step2~step5。
    initial_workflow: str

    # 对 Solver/工具调用的策略约束，例如必须澄清缺失参数、不得臆造信息等。
    restrict: str

    # FuzzyTaskAgent 产出的用户侧任务描述；这是 Solver 实际看到并尝试完成的问题。
    fuzzy_task: str

    # ToolCheckAgent 审核后的工具 schema 列表；Planner 和 Solver 只能调用这里列出的工具。
    checked_tools: List[Dict[str, Any]]

    # FuzzyTaskAgent 产出的背景资料；不一定直接暴露给 Solver，但供 MockUser/MockTool 模拟交互与知识库状态。
    task_background: str

    # PlanTrajectoryAgent 生成的计划步骤列表；每步通常包含 step_id、stage、purpose、tool_name、arguments。
    plan: List[Dict[str, Any]]

    # EvaluatePlanAgent 对当前 plan 的结构和语义评估结果；重规划时会作为反馈注入下一轮 Planner。
    plan_evaluation: Dict[str, Any]

    # 当前 plan 是否已通过评估；路由函数据此决定进入 execute_plan 还是回到 plan_trajectory。
    plan_is_valid: bool

    # 已生成/修订 plan 的次数；每次进入 PlanTrajectoryAgent 都递增，用于限制无限重规划。
    plan_revision_count: int

    # 允许的最大 plan 修订次数；来自配置，超过后若仍无有效计划则终止本样本。
    max_plan_revisions: int

    # Plan-Execute 模式下当前准备执行的 plan 下标；MockTool 返回后递增。
    current_plan_step: int

    # 已经执行过的计划步骤快照；最终 more_info 和失败诊断会用它还原执行进度。
    executed_steps: List[Dict[str, Any]]

    # 每个已执行步骤对应的工具调用、工具返回和是否引入新背景信息。
    step_results: List[Dict[str, Any]]

    # Solver 对话轨迹；包含 system/user/assistant/tool 消息，也是最终 solution*.json 的主要内容。
    solve_history: List[Dict[str, Any]]

    # 当前执行所绑定的 plan 修订编号；用于标记执行轨迹对应哪一版计划，目前主要作为审计字段。
    active_plan_revision: int

    # 虚拟世界的工具调用记忆；仅当工具返回引入新背景时追加，后续 MockTool/重规划会复用它。
    tool_call_history: List[Any]

    # 当前待执行的工具调用 JSON 字符串；路由到 MockTool 后由它读取，终止/重规划时可为空。
    current_tool_call: str

    # 图路由信号；常见值包括 "Tool call"、"Transfer to user"、"Need replan"、"Terminated"。
    task_finished: str

    # 失败终止原因；build_failure 统一写入，成功状态下通常为空字符串。
    failure_reason: str

    # 工具调用纠错重试计数；保留给非法 tool_call 自纠错/兼容旧流程，目前初始化后未在主流程中递增。
    tool_call_retry_count: int

    # Solver/ExecutePlan 已推进的轮次数；用于业务层最大轮次保护，避免图循环长期不产出终答。
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
    for message in solve_history:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            payload = None
        if (
            isinstance(payload, dict)
            and payload.get("action") == "final_answer"
            and isinstance(payload.get("answer"), str)
        ):
            return True
        if re.search(r"<answer>.*?</answer>", content, re.DOTALL | re.IGNORECASE):
            return True
    return False


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
        # 严格模式用于落盘前验收：必须未中断、工具集有效，并且轨迹里真的出现最终答案。
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
    # 规划/评估/执行 Agent 优先共用求解模型作为兜底，其他节点走通用回退。
    fallback_names = (
        ("SolveAgent", "FallbackModel", "Fallback")
        if step_name in {"PlanTrajectoryAgent", "EvaluatePlanAgent", "ExecutePlanAgent"}
        else ("FallbackModel", "Fallback")
    )
    step_model_config = step_models.get(step_name)
    for fallback_name in fallback_names:
        step_model_config = step_model_config or step_models.get(fallback_name)
    if step_model_config is None:
        raise KeyError(f"No model configuration found for step '{step_name}'")

    step_config = {"configurable": dict(step_model_config)}
    step_config["configurable"]["model_name"] = step_config["configurable"].pop("name")

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


def should_paraphrase_question(config: RunnableConfig) -> bool:
    configurable = config.get("configurable", {}) if config else {}
    synthesis_cfg = configurable.get("synthesis") or {}
    return bool(synthesis_cfg.get("paraphrase_supervised_question", True))


def allocate_next_solution_path(solve_path: str) -> str:
    """Scan solve_path for existing solutionN.json files and return the next path."""
    existing_numbers = []
    for file in glob.glob(f"{solve_path}/solution*.json"):
        match = re.match(r"solution(\d+)\.json$", os.path.basename(file))
        if match:
            existing_numbers.append(int(match.group(1)))
    next_number = max(existing_numbers) + 1 if existing_numbers else 1
    return f"{solve_path}/solution{next_number}.json"


def apply_mock_tool_response(
    state: AgentState,
    config: RunnableConfig,
    *,
    complexity=None,
    label: str = "",
    context: str = "",
) -> Dict[str, Any]:
    """Call MockToolAgent and return solve_history/tool_call_history updates.

    On failure returns a build_failure(...) payload; callers should check ``breaked``.
    """
    step_config = create_step_config(config, "MockToolAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    tool_call = state["current_tool_call"]
    tools_description = state["checked_tools"]
    tool_call_history = state["tool_call_history"]

    mock_kwargs: Dict[str, Any] = {}
    if complexity is not None:
        mock_kwargs["complexity"] = complexity
    if label:
        mock_kwargs["label"] = label
    if context:
        mock_kwargs["context"] = context

    tool_response, new_bg_introduced = mock_tool_response(
        cfg,
        tool_call,
        tools_description,
        tool_call_history,
        **mock_kwargs,
    )
    if tool_response is None:
        return build_failure(
            "MockToolAgent returned no tool response",
            solve_history=state["solve_history"],
            tool_call_history=tool_call_history,
            current_tool_call=tool_call,
        )

    tool_response_message = {
        "role": "tool",
        "content": json.dumps({"tool_response": tool_response}, ensure_ascii=False),
    }
    new_solve_history = state["solve_history"] + [tool_response_message]
    new_tool_call_history = tool_call_history
    if new_bg_introduced:
        try:
            parsed_tool_call: Any = json.loads(tool_call)
        except (TypeError, json.JSONDecodeError):
            parsed_tool_call = tool_call
        new_tool_call_history = tool_call_history + [
            {"query": parsed_tool_call, "response": tool_response}
        ]

    update: Dict[str, Any] = {
        "tool_call_history": new_tool_call_history,
        "solve_history": new_solve_history,
    }
    if state.get("plan"):
        current_plan_step = int(state.get("current_plan_step", 0) or 0)
        plan = state.get("plan", [])
        if 0 <= current_plan_step < len(plan):
            planned_step = plan[current_plan_step]
            update.update({
                "executed_steps": (state.get("executed_steps") or []) + [planned_step],
                "step_results": (state.get("step_results") or []) + [{
                    "step_id": planned_step.get("step_id", current_plan_step + 1),
                    "tool_call": tool_call,
                    "tool_response": tool_response,
                    "new_bg_introduced": bool(new_bg_introduced),
                }],
                "current_plan_step": current_plan_step + 1,
            })

    return update
