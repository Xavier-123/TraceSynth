import json
from typing import Any, Dict, Tuple, Union

from .call_llms import ParseError, call_and_parse, parse_json_object
from .prompt import mock_tool_system_prompt, mock_tool_user_prompt


def _parse_mock_tool_response(response_text: str) -> Tuple[Union[Dict[str, Any], str], bool]:
    parsed_data = parse_json_object(response_text)
    if "tool_response" not in parsed_data:
        raise ParseError("tool response JSON is missing required field 'tool_response'")

    tool_response = parsed_data["tool_response"]
    new_bg_introduced = parsed_data.get("new_bg_introduced", "NO")

    # 只有显式 YES 才把这次工具返回写入世界状态，防止无新增信息污染历史。
    return tool_response, new_bg_introduced == "YES"


def mock_tool_response(
    cfg,
    query,
    tool_description,
    history_interactions,
    complexity=None,
    label: str = "",
    context: str = "",
):
    from tracesynth.configuration import SynthesisComplexity

    if complexity is None:
        complexity = SynthesisComplexity()
    # prompt = tool_simulation_prompt_with_memory.format(
    # MockToolAgent 同时接收金标和原始上下文，但这些信息只用于模拟虚拟知识库，不直接暴露给 Solver。
    prompt = mock_tool_user_prompt.format(
        query=query,
        tools=tool_description,
        world_state=json.dumps(history_interactions),
        label=label or "(not provided)",
        context=context or "(not provided)",
        **complexity.to_prompt_vars(),
    )
    messages = [
        {"role": "system", "content": mock_tool_system_prompt},
        {"role": "user", "content": prompt},
    ]
    parsed, _ = call_and_parse(
        cfg,
        messages,
        _parse_mock_tool_response,
        step_name="MockToolAgent",
        # json_mode=True,
    )
    if parsed is None:
        return None, False
    return parsed
