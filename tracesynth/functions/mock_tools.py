import json
import re
from typing import Any, Dict, Tuple, Union

from .call_llms import ParseError, call_and_parse
from .prompt import tool_simulation_prompt_with_memory


def _parse_mock_tool_response(response_text: str) -> Tuple[Union[Dict[str, Any], str], bool]:
    cleaned_text = response_text.strip()

    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(json_block_pattern, cleaned_text)
    json_str = match.group(1).strip() if match else cleaned_text

    try:
        parsed_data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON in tool response: {exc}") from exc

    if not isinstance(parsed_data, dict):
        raise ParseError("tool response JSON must be an object")
    if "tool_response" not in parsed_data:
        raise ParseError("tool response JSON is missing required field 'tool_response'")

    tool_response = parsed_data["tool_response"]
    new_bg_introduced = parsed_data.get("new_bg_introduced", "NO")

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
    prompt = tool_simulation_prompt_with_memory.format(
        query=query,
        tools=tool_description,
        world_state=json.dumps(history_interactions),
        label=label or "(not provided)",
        context=context or "(not provided)",
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
