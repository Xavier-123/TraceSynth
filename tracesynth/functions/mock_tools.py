import json
import re
from .call_llms import ParseError, call_and_parse
from .prompt import tool_simulation_prompt_with_memory


def _parse_mock_tool_response(content: str):
    tool_response_matches = re.findall(
        r"<tool_response>(.+?)</tool_response>", content, re.DOTALL,
    )
    if not tool_response_matches:
        raise ParseError("missing <tool_response> tag")

    tool_response = tool_response_matches[-1].strip()
    new_bg_introduced_matches = re.findall(
        r"<new_bg_introduced>(.+?)</new_bg_introduced>", content, re.DOTALL,
    )
    if new_bg_introduced_matches:
        new_bg_introduced = new_bg_introduced_matches[-1].strip()
    else:
        new_bg_introduced = "YES"

    return tool_response, "YES" in new_bg_introduced


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
