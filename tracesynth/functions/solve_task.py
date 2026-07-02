import re
import copy
from typing import List, Dict, Optional, Tuple

from .call_llms import ParseError, call_and_parse, call_llm_messages


def _parse_solver_response(content: str) -> Tuple[str, Optional[str]]:
    if not content or not content.strip():
        raise ParseError("empty solver content")
    tool_call_matches = re.findall(r"<tool_call>(.+?)</tool_call>", content, re.DOTALL)
    if tool_call_matches:
        tool_call = tool_call_matches[-1].strip()
    else:
        tool_call = None
    return content, tool_call


def solve_task_by_tools(cfg, solve_history):
    solve_history = copy.deepcopy(solve_history)

    parsed, _ = call_and_parse(
        cfg,
        solve_history,
        _parse_solver_response,
        step_name="FinalLLMResponse",
    )
    if parsed is None:
        return None, None
    return parsed
