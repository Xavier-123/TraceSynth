import json
import re
from .call_llms import ParseError, call_and_parse
from .prompt import tool_simulation_prompt_with_memory

_TOOL_RESPONSE_TAG_RE = re.compile(
    r"<tool_response(?:\s[^>]*)?>(.+?)</tool_response>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_RESPONSE_OPEN_RE = re.compile(
    r"<tool_response(?:\s[^>]*)?>\s*(.+)$",
    re.DOTALL | re.IGNORECASE,
)
_NEW_BG_TAG_RE = re.compile(
    r"<new_bg_introduced(?:\s[^>]*)?>(.+?)</new_bg_introduced>",
    re.DOTALL | re.IGNORECASE,
)
_STRUCTURED_TAG_RE = re.compile(r"<[a-zA-Z_][\w-]*(?:\s[^>]*)?>")


def _extract_tool_response(content: str) -> str | None:
    matches = _TOOL_RESPONSE_TAG_RE.findall(content)
    if matches:
        return matches[-1].strip()

    open_match = _TOOL_RESPONSE_OPEN_RE.search(content)
    if open_match:
        return open_match.group(1).strip()

    return None


def _parse_new_bg_introduced(content: str) -> bool:
    matches = _NEW_BG_TAG_RE.findall(content)
    if matches:
        return "YES" in matches[-1].strip().upper()
    return True


def _parse_mock_tool_response(content: str):
    tool_response = _extract_tool_response(content)
    if tool_response:
        return tool_response, _parse_new_bg_introduced(content)

    cleaned = content.strip()
    if cleaned and not _STRUCTURED_TAG_RE.search(cleaned):
        return cleaned, True

    raise ParseError("missing <tool_response> tag")


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
