import json
import re
from typing import Tuple, Any, Dict, Union
from .call_llms import ParseError, call_and_parse
from .prompt import tool_simulation_prompt_with_memory


def _parse_mock_tool_response(response_text: str) -> Tuple[Union[Dict[str, Any], str], bool]:
    cleaned_text = response_text.strip()

    # 正则提取 Markdown 代码块中的 JSON 内容
    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(json_block_pattern, cleaned_text)
    if match:
        json_str = match.group(1).strip()
    else:
        json_str = cleaned_text

    try:
        parsed_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"解析 JSON 失败。原始文本:\n{response_text}") from e

    # 提取字段
    if "tool_response" not in parsed_data:
        raise ValueError("JSON 数据中未包含必需的 'tool_response' 字段")

    tool_response = parsed_data["tool_response"]
    new_bg_introduced = parsed_data.get("new_bg_introduced", "NO")  # 若缺失默认返回 "NO"

    if new_bg_introduced == "YES":
        return tool_response, True
    else:
        return tool_response, False


def mock_tool_response(
    cfg,
    query,
    tool_description,
    history_interactions,
    complexity=None,
    label: str = "",
    context: str = "",
):
    try:
        from tracesynth.configuration import SynthesisComplexity
        if complexity is None:
            complexity = SynthesisComplexity()
        prompt = tool_simulation_prompt_with_memory.format(
            query=query,
            tools=tool_description,
            world_state=json.dumps(history_interactions),
            label=label or "（未提供）",
            context=context or "（未提供）",
            **complexity.to_prompt_vars(),
        )
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]
        parsed, _ = call_and_parse(
            cfg,
            messages,
            _parse_mock_tool_response,
            step_name="MockToolAgent",
        )
        if parsed is None:
            return None, False
        return parsed
    except Exception as e:
        print(e)


