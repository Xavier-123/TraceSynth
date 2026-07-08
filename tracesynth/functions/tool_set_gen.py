import json

from .call_llms import ParseError, call_and_parse, parse_json_object
from .prompt import tool_set_prompt


def _is_non_empty_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _parse_tool_set_response(content: str):
    payload = parse_json_object(content)
    task = payload.get("task")
    tools = payload.get("tools")
    workflow = payload.get("workflow")
    restrict = payload.get("restriction")

    if not _is_non_empty_text(task):
        raise ParseError("missing required JSON field: task")
    if not isinstance(tools, list) or not tools:
        raise ParseError("JSON field tools must be a non-empty array")
    if not _is_non_empty_text(workflow):
        raise ParseError("missing required JSON field: workflow")
    if not _is_non_empty_text(restrict):
        raise ParseError("missing required JSON field: restriction")

    all_content = json.dumps(payload, ensure_ascii=False, indent=2)
    tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
    return all_content, task.strip(), tools_json, workflow.strip(), restrict.strip()


def generate_tool_set(cfg, background_info, complexity=None):
    from tracesynth.configuration import SynthesisComplexity
    if complexity is None:
        complexity = SynthesisComplexity()
    # complexity 控制工具数量、干扰工具数量和迭代复杂度，直接注入工具设计提示词。
    prompt = tool_set_prompt.format(
        background_info=background_info,
        **complexity.to_prompt_vars(),
    )
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]
    parsed, _ = call_and_parse(
        cfg,
        messages,
        _parse_tool_set_response,
        step_name="ToolSetGenAgent",
        # json_mode=True,
    )
    if parsed is None:
        return None, None, None, None, None
    return parsed
