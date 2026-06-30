import os
import json
import yaml
import glob
import re
import threading
import logging
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from tracesynth.configuration import ModelConfiguration, SynthesisComplexity
from tracesynth.io import (
    validate_seed_info,
    extract_predicted_answer,
    check_label_match,
)
from tracesynth.functions import (
    generate_tool_set, generate_fuzzy_task, tool_check,
    mock_tool_response, solve_task_by_tools, mock_user_response
)
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

    solve_history: List[Dict[str, Any]]
    tool_call_history: List[str]
    current_tool_call: str
    task_finished: str
    failure_reason: str
    tool_call_retry_count: int


def build_failure(reason: str, **extra: Any) -> Dict[str, Any]:
    payload = {
        "breaked": True,
        "task_finished": "Terminated",
        "failure_reason": reason,
    }
    payload.update(extra)
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
    return True, None


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
    step_model_config = base_config["configurable"]["step_models"][step_name]

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


def use_label_as_answer(config: RunnableConfig) -> bool:
    eval_cfg = config.get("configurable", {}).get("evaluation") or {}
    return bool(eval_cfg.get("use_label_as_answer", True))


def get_synthesis_complexity(config: RunnableConfig) -> SynthesisComplexity:
    return SynthesisComplexity.from_run_config(config.get("configurable", {}))


def is_supervised_seed(seed_info: Dict[str, Any]) -> bool:
    return bool(seed_info.get("label") and seed_info.get("question"))


def toolset_gen_node(state: AgentState, config: RunnableConfig):
    logger.info("------------------ToolSetGenAgent------------------")

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
    logger.info("------------------FuzzyTaskAgent------------------")
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
    logger.info("------------------ToolCheckAgent------------------")

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


def solve_task_node(state: AgentState, config: RunnableConfig):
    logger.info("------------------SolveTaskAgent------------------")

    if state["breaked"]:
        return {
            "current_tool_call": None,
            "task_finished": "Terminated"
        }

    # Create step-specific configuration
    step_config = create_step_config(config, "SolveAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    if not len(state.get("solve_history", [])):
        checked_tools = state["checked_tools"]
        task_info = state["fuzzy_task"]
        restrict = state["restrict"]
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

        solve_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        solve_history = state["solve_history"]

    one_step_think_and_tool_call, tool_call_info = solve_task_by_tools(cfg, solve_history)
    if not is_non_empty_text(one_step_think_and_tool_call):
        return build_failure("SolveAgent returned empty content", **{"solve_history": solve_history})

    one_step_think_and_tool_call_message = {
        "role": "assistant", "content": one_step_think_and_tool_call
    }
    solve_history.append(one_step_think_and_tool_call_message)

    if not re.search(r"<answer>.*?</answer>", one_step_think_and_tool_call, re.DOTALL | re.IGNORECASE):
        if tool_call_info is None:
            task_finished = "Transfer to user"
        else:
            is_valid, error = validate_tool_call(tool_call_info, state["checked_tools"])
            if not is_valid:
                retry_count = state.get("tool_call_retry_count", 0)
                max_retries = get_tool_call_max_retries(config)
                if retry_count >= max_retries:
                    return build_failure(
                        error or "Invalid tool_call",
                        **{"solve_history": solve_history},
                    )
                solve_history.append({
                    "role": "tool",
                    "content": (
                        f"<tool_response>Your tool_call is invalid: {error}. "
                        "Please fix it and output a valid tool_call.</tool_response>"
                    ),
                })
                return {
                    "current_tool_call": None,
                    "solve_history": solve_history,
                    "task_finished": "Retry solve",
                    "tool_call_retry_count": retry_count + 1,
                }
            task_finished = "Tool call"
    else:
        if use_label_as_answer(config):
            label = (state["seed_info"].get("label") or "").strip()
            if label:
                solve_history[-1] = {
                    "role": "assistant",
                    "content": f"<answer>{label}</answer>",
                }
        task_finished = "Terminated"

    return {
        "current_tool_call": tool_call_info,
        "solve_history": solve_history,
        "task_finished": task_finished
    }


def mock_tools_node(state: AgentState, config: RunnableConfig):
    logger.info("------------------MockToolsAgent------------------")
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
        return build_failure("MockToolAgent returned no tool response", **{"solve_history": solve_history})

    tool_response_message = {"role": "tool", "content": f"<tool_response>{tool_response}</tool_response>"}

    solve_history.append(tool_response_message)
    if new_bg_introduced:
        tool_call_history.append(f"Query:\n{tool_call}, Response:\n{tool_response}")

    return {
        "tool_call_history": tool_call_history,
        "solve_history": solve_history
    }

def mock_user_node(state: AgentState, config: RunnableConfig):
    logger.info("------------------MockUserAgent------------------")

    if state["breaked"]:
        return {}

    step_config = create_step_config(config, "MockUserAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    fuzzy_task = state["fuzzy_task"]
    task_background = state["task_background"]
    restrict = state["restrict"]
    solve_history = state["solve_history"]

    user_response = mock_user_response(cfg, fuzzy_task, task_background, restrict, solve_history)
    if user_response is None:
        return build_failure("MockUserAgent returned no user response", **{"solve_history": solve_history})

    solve_history.append({"role": "user", "content": user_response})

    return {
        "solve_history": solve_history
    }


def should_call_tool(state: AgentState):
    if state.get("breaked") or state["task_finished"] == "Terminated":
        return "end"
    elif state["task_finished"] == "Tool call":
        return "tool_call"
    elif state["task_finished"] == "Retry solve":
        return "retry_solve"
    else:
        return "user"


# Build the graph
builder = StateGraph(AgentState, config_schema=RunnableConfig)
builder.add_node("toolset_gen", toolset_gen_node)
builder.add_node("fuzzy_task", fuzzy_task_node)
builder.add_node("check_tools", check_tools_node)
builder.add_node("reason_and_act", solve_task_node)
builder.add_node("mock_tools", mock_tools_node)
builder.add_node("mock_user", mock_user_node)

builder.set_entry_point("toolset_gen")
builder.add_edge("toolset_gen", "fuzzy_task")
builder.add_edge("fuzzy_task", "check_tools")
builder.add_edge("check_tools", "reason_and_act")
builder.add_conditional_edges(
    "reason_and_act",
    should_call_tool,
    {"tool_call": "mock_tools", "user": "mock_user", "retry_solve": "reason_and_act", "end": END}
)
builder.add_edge("mock_tools", "reason_and_act")
builder.add_edge("mock_user", "reason_and_act")
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
        "solve_history": [],
        "tool_call_history": [],
        "tool_call_retry_count": 0,
    }
    run_config["recursion_limit"] = 100
    final_state = graph.invoke(initial_state, config=run_config)

    if not is_successful_final_state(final_state):
        failure_data = {
            "id": seed_info["id"],
            "question": seed_info.get("question"),
            "label": seed_info.get("label"),
            "context_present": bool(seed_info.get("context")),
            "failure_reason": final_state.get("failure_reason", "generation did not produce a valid final answer"),
            "fuzzy_task": final_state.get("fuzzy_task"),
            "checked_tools": final_state.get("checked_tools"),
        }
        with log_file_lock:
            with open(failed_task_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(failure_data, ensure_ascii=False) + '\n')
            with open(os.path.join(solve_path, "failed_state.json"), 'w', encoding='utf-8') as f:
                f.write(json.dumps(final_state, ensure_ascii=False, indent=4) + '\n')
        return final_state

    predicted_answer = extract_predicted_answer(final_state.get("solve_history"))
    label_check = check_label_match(
        predicted_answer,
        seed_info.get("label", ""),
        skip=skip_label_match,
    )
    if label_check["label_match_status"] == "mismatch":
        failure_data = {
            "id": seed_info["id"],
            "question": seed_info.get("question"),
            "label": seed_info.get("label"),
            "context_present": bool(seed_info.get("context")),
            "failure_reason": "predicted answer does not match label",
            "fuzzy_task": final_state.get("fuzzy_task"),
            "checked_tools": final_state.get("checked_tools"),
            **label_check,
        }
        with log_file_lock:
            with open(failed_task_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(failure_data, ensure_ascii=False) + '\n')
            with open(os.path.join(solve_path, "failed_state.json"), 'w', encoding='utf-8') as f:
                f.write(json.dumps({**final_state, **label_check}, ensure_ascii=False, indent=4) + '\n')
        return {**final_state, **label_check, "breaked": True, "failure_reason": failure_data["failure_reason"]}

    save_data = {
        "id": seed_info["id"],
        "question": seed_info.get("question"),
        "label": seed_info.get("label"),
        "context_present": bool(seed_info.get("context")),
        "fuzzy_task": final_state["fuzzy_task"],
        "checked_tools": final_state["checked_tools"],
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
    with open("configs/tool_use_data_gen.yaml", 'r', encoding='utf-8') as f:
        agent_config = yaml.safe_load(f)

    with open("configs/seed_qa_sample.jsonl", 'r', encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    for task in tasks:
        run_agent(task, run_config=agent_config)
