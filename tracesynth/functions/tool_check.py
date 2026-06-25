import json
import re
from typing import Any, Dict, List, Optional

from .call_llms import ParseError, call_and_parse
from .prompt import tool_check_prompt


def _parse_checked_tools(content: str) -> List[Dict[str, Any]]:
    tool_matches = re.findall(r"<tools>(.+?)</tools>", content, re.DOTALL)
    if not tool_matches:
        raise ParseError("missing <tools> tag")

    tools_str = tool_matches[-1].strip()
    try:
        checked_tools = json.loads(tools_str)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ParseError(f"invalid JSON in <tools>: {exc}") from exc

    if not isinstance(checked_tools, list) or not checked_tools:
        raise ParseError("checked_tools is empty or not a list")

    for tool in checked_tools:
        if not isinstance(tool, dict) or not tool.get("name") or not isinstance(tool.get("parameters"), dict):
            raise ParseError("checked_tools contains invalid tool schema")

    return checked_tools


def tool_check(cfg, tool_description, task_description, complexity=None) -> Optional[List[Dict[str, Any]]]:
    from tracesynth.configuration import SynthesisComplexity
    if complexity is None:
        complexity = SynthesisComplexity()
    prompt = tool_check_prompt.format(
        task_description=task_description,
        tool_description=tool_description,
        **complexity.to_prompt_vars(),
    )
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]
    parsed, _ = call_and_parse(
        cfg,
        messages,
        _parse_checked_tools,
        step_name="ToolCheckAgent",
    )
    return parsed
