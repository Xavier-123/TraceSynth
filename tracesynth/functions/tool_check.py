from typing import Any, Dict, List, Optional

from .call_llms import ParseError, call_and_parse, parse_json_object
from .prompt import tool_check_prompt


def _parse_checked_tools(content: str) -> List[Dict[str, Any]]:
    payload = parse_json_object(content)
    checked_tools = payload.get("tools")

    if not isinstance(checked_tools, list) or not checked_tools:
        raise ParseError("JSON field tools must be a non-empty array")

    for tool in checked_tools:
        # 最小 schema 校验：名称和 parameters 必须存在，required 参数细节由 validate_tool_call 再检查。
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
        # json_mode=True,
    )
    return parsed
