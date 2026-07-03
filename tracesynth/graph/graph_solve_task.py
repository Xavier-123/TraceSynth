import os
import json
import yaml
import glob
import re
import threading

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from tracesynth.configuration import ModelConfiguration
from tracesynth.functions import (
    mock_tool_response, solve_task_by_tools, mock_user_response
)
from tracesynth.graph.node_utils import (
    AgentState,
    build_failure,
    create_step_config,
    get_graph_recursion_limit,
    get_solver_max_turns,
    is_graph_recursion_error,
    validate_tool_call,
)

log_file_lock = threading.Lock()
def solve_task_node(state: AgentState, config: RunnableConfig):
    if state["breaked"]:
        return {
            "current_tool_call": None,
            "task_finished": "Terminated"
        }

    solver_turn_count = int(state.get("solver_turn_count", 0) or 0) + 1
    max_solver_turns = get_solver_max_turns(config)
    if solver_turn_count > max_solver_turns:
        return build_failure(
            f"SolveAgent exceeded max_solver_turns={max_solver_turns} without producing <answer>",
            solve_history=state.get("solve_history", []),
            tool_call_history=state.get("tool_call_history", []),
            solver_turn_count=solver_turn_count,
            max_solver_turns=max_solver_turns,
        )

    step_config = create_step_config(config, "SolveAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    if not len(state.get("solve_history", [])):
        checked_tools = state["checked_tools"]
        task_info = state["fuzzy_task"]
        restrict = state["restrict"]

        tools_description = ""
        for tool in checked_tools:
            tools_description += json.dumps({"type": "function", "function": tool}) + "\n"

        system_prompt = """<policy>{restrict}</policy>
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{available_tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""
        system_prompt = system_prompt.format(available_tools=tools_description, restrict=restrict)
        prompt = f"""Task Description: {task_info}.

### Requirements:
1. Please call only one tool at a time, and you must provide your brief reasoning process before using any tool. You can not just give a tool call without providing your reasoning process.

2. Once the task is complete, output the final answer, wrapping the answer in `<answer></answer>` as a termination signal. 

3. IMPORTANT: The user most likely provided insufficient information, you are encouraged to interact with the user to gather more information if needed. Before calling any tool, if **any required parameter is uncertain, missing, ambiguous, or not explicitly provided by the user**, you **MUST ask the user for clarification first**. Do NOT guess or fabricate parameters!!!
"""
        solve_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        solve_history = state["solve_history"]

    one_step_think_and_tool_call, tool_call_info = solve_task_by_tools(cfg, solve_history)
    one_step_think_and_tool_call_message = {
        "role": "assistant", "content": one_step_think_and_tool_call
    }
    solve_history.append(one_step_think_and_tool_call_message)
    
    if "<answer>" not in one_step_think_and_tool_call:
        if tool_call_info is None:
            task_finished = "Transfer to user"
        else:
            is_valid, error = validate_tool_call(tool_call_info, state["checked_tools"])
            if not is_valid:
                return build_failure(
                    error or "Invalid tool_call",
                    current_tool_call=tool_call_info,
                    solve_history=solve_history,
                    tool_call_history=state.get("tool_call_history", []),
                    solver_turn_count=solver_turn_count,
                )
            task_finished = "Tool call"
    else:
        task_finished = "Terminated"
    

    return {
        "current_tool_call": tool_call_info,
        "solve_history": solve_history,
        "task_finished": task_finished,
        "solver_turn_count": solver_turn_count,
    }

def mock_tools_node(state: AgentState, config: RunnableConfig):
    if state["breaked"]:
        return {}

    step_config = create_step_config(config, "MockToolAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    tool_call = state["current_tool_call"]
    tools_description = state["checked_tools"]
    tool_call_history = state["tool_call_history"]
    solve_history = state["solve_history"]

    tool_response, new_bg_introduced = mock_tool_response(cfg, tool_call, tools_description, tool_call_history)
    if tool_response is None:
        return build_failure(
            "MockToolAgent returned no tool response",
            solve_history=solve_history,
            tool_call_history=tool_call_history,
            current_tool_call=tool_call,
        )

    tool_response_message = {
        "role": "tool", "content": f"<tool_response>{tool_response}</tool_response>"
    }
    solve_history.append(tool_response_message)
    if new_bg_introduced:
        tool_call_history.append(f"Query:\n{tool_call}, Response:\n{tool_response}")
    
    return {
        "tool_call_history": tool_call_history,
        "solve_history": solve_history
    }

def mock_user_node(state: AgentState, config: RunnableConfig):
    if state["breaked"]:
        return {}

    try:
        step_config = create_step_config(config, "MockUserAgent")
    except KeyError:
        step_config = create_step_config(config, "MockToolAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    fuzzy_task = state["fuzzy_task"]
    task_background = state["task_background"]
    restrict = state["restrict"]
    solve_history = state["solve_history"]

    user_response = mock_user_response(cfg, fuzzy_task, task_background, restrict, solve_history)
    solve_history.append({"role": "user", "content": user_response})

    return {
        "solve_history": solve_history
    }

def should_call_tool(state: AgentState):
    if state.get("breaked") or state["task_finished"] == "Terminated":
        return "end"
    elif state["task_finished"] == "Tool call":
        return "tool_call"
    else:
        return "user"

# Build the graph
builder = StateGraph(AgentState, config_schema=RunnableConfig)
builder.add_node("reason_and_act", solve_task_node)
builder.add_node("mock_tools", mock_tools_node)
builder.add_node("mock_user", mock_user_node)

builder.set_entry_point("reason_and_act")
builder.add_conditional_edges(
    "reason_and_act",
    should_call_tool,
    {"tool_call": "mock_tools", "user": "mock_user", "end": END}
)
builder.add_edge("mock_tools", "reason_and_act")
builder.add_edge("mock_user", "reason_and_act")
graph = builder.compile()

# --- 运行入口 ---
def run_agent(seed_info: dict, run_config: dict = None):
    already_processed_path = run_config["logging"]["already_processed_path"]
    solve_path = run_config["logging"]["solve_path"]
    solve_path = os.path.join(solve_path, f"{seed_info['id']}")
    if not os.path.exists(solve_path):
        os.makedirs(solve_path)
    repeat_times = int(run_config["logging"]["repeat_times"])

    tool_call_history_path = f"{solve_path}/tool_call_history.json"
    more_info_path = f"{solve_path}/more_info.json"
    graph_config = {"configurable": run_config or {}}
    max_solver_turns = get_solver_max_turns(graph_config)
    graph_config["recursion_limit"] = get_graph_recursion_limit(graph_config, max_solver_turns)

    if len(glob.glob(f"{solve_path}/rubrics_output.json")) > 0:
        return

    for _ in range(repeat_times):
        solution_files = glob.glob(f"{solve_path}/solution*.json")
        # Extract numbers from existing solution files
        existing_numbers = []
        for file in solution_files:
            basename = os.path.basename(file)
            # Match files with pattern solution<number>.json
            match = re.match(r'solution(\d+)\.json$', basename)
            if match:
                existing_numbers.append(int(match.group(1)))

        next_number = max(existing_numbers) + 1 if existing_numbers else 1

        if os.path.exists(tool_call_history_path):
            with open(tool_call_history_path, 'r', encoding='utf-8') as f:
                tool_call_history = json.load(f)
        else:
            tool_call_history = []

        if os.path.exists(more_info_path):
            with open(more_info_path, 'r', encoding='utf-8') as f:
                more_info = json.load(f)
        else:
            more_info = {}

        initial_state = {
            "fuzzy_task": seed_info["fuzzy_task"],
            "checked_tools": seed_info["checked_tools"],
            "task_background": more_info.get("task_background", ""),
            "restrict": more_info.get("restrict", ""),
            "breaked": False,
            "task_finished": False,
            "failure_reason": "",
            "solve_history": [],
            "tool_call_history": tool_call_history,
            "tool_call_retry_count": 0,
            "solver_turn_count": 0,
        }
        try:
            final_state = graph.invoke(initial_state, config=graph_config)
        except Exception as exc:
            if not is_graph_recursion_error(exc):
                raise
            final_state = build_failure(
                f"LangGraph recursion limit reached before stop condition: {exc}",
                **{
                    **initial_state,
                    "max_solver_turns": max_solver_turns,
                    "recursion_limit": graph_config["recursion_limit"],
                },
            )

        solution_filename = f"{solve_path}/solution{next_number}.json"

        with open(solution_filename, 'w', encoding='utf-8') as f:
            f.write(json.dumps(final_state["solve_history"], ensure_ascii=False, indent=4) + '\n')

        with open(f"{solve_path}/tool_call_history.json", 'w', encoding='utf-8') as f:
            f.write(json.dumps(final_state["tool_call_history"], ensure_ascii=False, indent=4) + '\n')

    with log_file_lock:
        # Save basic task data
        with open(already_processed_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"id": seed_info['id']}, ensure_ascii=False) + '\n')
    
    return final_state


if __name__ == "__main__":
    with open("configs/solve_task.yaml", 'r', encoding='utf-8') as f:
        agent_config = yaml.safe_load(f)

    # Example usage
    with open("output/virtual_tool_use.jsonl", 'r', encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f]

    for task in tasks:
        with open(f"output/solve_tool_use/{task['id']}/tool_call_history.json", 'r', encoding='utf-8') as f:
            tool_call_history = json.load(f)
        
        new_task = {
            "id": task["id"],
            "fuzzy_task": task["fuzzy_task"],
            "checked_tools": task["checked_tools"],
            "tool_call_history": tool_call_history
        }
        run_agent(new_task, run_config=agent_config)

