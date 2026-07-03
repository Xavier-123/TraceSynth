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
    mock_tool_response,
)
from tracesynth.functions.call_llms import call_and_parse
from tracesynth.functions.fuzzy_task import is_supervised_seed
from tracesynth.functions.plan_trajectory import _build_plan_messages, _parse_plan_response
from tracesynth.functions.execute_plan import _initial_solve_history_from_plan, _generate_final_answer_from_plan, _format_planned_tool_message
from tracesynth.functions.evaluate_plan import validate_tool_call, _build_plan_evaluation_messages, _parse_plan_evaluation_response
from tracesynth.graph.node_utils import (
    AgentState,
    build_failure,
    create_step_config,
    get_synthesis_complexity,
    is_non_empty_text,
    get_plan_max_revisions,
    get_solver_max_turns,
    get_graph_recursion_limit,
    is_graph_recursion_error,
    is_successful_final_state,
)

# 多线程批量生成时会并发写 JSONL，统一用锁保护追加写入和失败快照。
log_file_lock = threading.Lock()
logger = logging.getLogger(__name__)


def toolset_gen_node(state: AgentState, config: RunnableConfig):
    logger.debug("------------------ToolSetGenAgent------------------")

    # Create step-specific configuration
    step_config = create_step_config(config, "ToolSetGenAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)
    complexity = get_synthesis_complexity(config)

    seed_info = state["seed_info"]
    background_info = seed_info.get("background") or seed_info.get("question", "")
    # 工具设计阶段只接触种子背景，负责生成初始任务、工具清单、工作流和约束。
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
        # 监督数据已有真实问题，不能再让 LLM 改写问题；这里只补虚拟交互所需背景。
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
        # 背景由原始 context 和 LLM 生成背景拼接，既保留证据又补足场景设定。
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
    # 每次重规划都会把上一轮评估反馈注入 prompt，并递增 revision_count 供上限判断。
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
        # LLM 评估可能漏掉结构性错误，因此再用确定性校验强制拦截未知工具和缺参计划。
        evaluation["is_valid"] = False
        evaluation.setdefault("issues", [])
        evaluation["issues"].extend(basic_issues)
        evaluation.setdefault("reasons", [])
        evaluation["reasons"].append("Basic deterministic validation found invalid tool calls.")

    max_revisions = get_plan_max_revisions(config)
    if not evaluation["is_valid"] and int(state.get("plan_revision_count", 0) or 0) >= max_revisions:
        # 达到最大修订次数后不再继续重规划，避免图在坏计划上无限循环。
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
        # Solver 回合上限是业务层停止条件，优先于 LangGraph 递归错误暴露更明确的失败原因。
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
        # 所有计划步骤完成后进入最终回答；若证据仍不足，会返回 Need replan 触发重规划。
        return _generate_final_answer_from_plan(state, config, solver_turn_count)

    step = plan[current_plan_step]
    # 计划步骤转换为标准 tool_call 字符串后，继续复用统一工具校验逻辑。
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
        # Plan-Execute 模式下记录执行过的计划步和工具返回，供终答、重规划和落盘审计使用。
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
    # 评估节点后的核心路由：有效计划进入执行，无效但未超限则回到规划，否则终止。
    if state.get("breaked"):
        return "end"
    if state.get("plan_is_valid"):
        return "execute"

    max_revisions = int(state.get("max_plan_revisions", 3) or 3)
    if int(state.get("plan_revision_count", 0) or 0) < max_revisions:
        return "replan"
    return "end"


def should_continue_execution(state: AgentState):
    # ExecutePlanAgent 通过 task_finished 字符串声明下一步：调用工具、重规划或结束。
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
    # 失败样本保留完整状态和部分轨迹，便于人工复盘或后续重试。
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
    # 运行入口先统一种子 schema，再读取日志、失败日志和求解产物目录配置。
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
        # 一次图调用覆盖完整合成生命周期：工具生成、模糊任务、计划、执行和最终回答。
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
        # 图级失败会写入失败 JSONL 和快照文件，避免只在日志里丢失失败样本。
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
        # P0 标签校验放在成功状态之后，确保落盘轨迹不仅有答案，而且答案与金标一致。
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

        # 同一任务可能多次采样，按已有 solutionN.json 自动分配下一个编号。
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
