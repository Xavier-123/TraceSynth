import json
import copy
from typing import List, Dict, Optional, Tuple

from .call_llms import ParseError, call_and_parse, parse_json_object


def _parse_solver_response(content: str) -> Tuple[str, Optional[str]]:
    if not content or not content.strip():
        raise ParseError("empty solver content")
    payload = parse_json_object(content)
    action = payload.get("action")
    if action not in {"tool_call", "ask_user", "final_answer"}:
        raise ParseError("solver JSON field action must be tool_call, ask_user, or final_answer")

    tool_call = None
    if action == "tool_call":
        raw_tool_call = payload.get("tool_call")
        if not isinstance(raw_tool_call, dict):
            raise ParseError("solver tool_call action requires object field tool_call")
        name = raw_tool_call.get("name")
        arguments = raw_tool_call.get("arguments")
        if not isinstance(name, str) or not name.strip():
            raise ParseError("solver tool_call.name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise ParseError("solver tool_call.arguments must be an object")
        tool_call = json.dumps(
            {"name": name.strip(), "arguments": arguments},
            ensure_ascii=False,
        )
    elif action == "ask_user":
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ParseError("solver ask_user action requires non-empty field message")
    else:
        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ParseError("solver final_answer action requires non-empty field answer")
    return content, tool_call


def solve_task_by_tools(cfg, solve_history):
    # 深拷贝防止 call_and_parse 在解析失败回灌反馈时污染调用方持有的原始轨迹。
    solve_history = copy.deepcopy(solve_history)

    parsed, _ = call_and_parse(
        cfg,
        solve_history,
        _parse_solver_response,
        step_name="FinalLLMResponse",
        # json_mode=True,
    )
    if parsed is None:
        return None, None
    return parsed
