import re
import json
from .call_llms import ParseError, call_and_parse
from .prompt import fixed_tool_set_prompt


def _is_non_empty_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _parse_tool_set_response(content: str):
    # reasoning 只用于模型思考，不进入后续业务字段。
    all_content = re.sub(r"<reasoning>(.+?)</reasoning>", "", content, flags=re.DOTALL)
    # The catalog is injected by code; discard any model-authored tool block completely.
    all_content = re.sub(r"<tools>(.+?)</tools>", "", all_content, flags=re.DOTALL)

    # 各段取最后一次匹配，兼容模型先输出草稿再修正的情况。
    workflow_matches = re.findall(r"<workflow>(.+?)</workflow>", all_content, re.DOTALL)
    workflow = workflow_matches[-1].strip() if workflow_matches else None

    task_matches = re.findall(r"<task>(.+?)</task>", all_content, re.DOTALL)
    task = task_matches[-1].strip() if task_matches else None

    restrict_matches = re.findall(r"<restriction>(.+?)</restriction>", all_content, re.DOTALL)
    restrict = restrict_matches[-1].strip() if restrict_matches else None

    if not all(_is_non_empty_text(value) for value in (task, workflow, restrict)):
        raise ParseError("missing required sections: task/workflow/restriction")

    return task, workflow, restrict


def generate_tool_set(cfg, background_info, tool_catalog, complexity=None):
    from tracesynth.configuration import SynthesisComplexity
    if complexity is None:
        complexity = SynthesisComplexity()
    # complexity 控制工具数量、干扰工具数量和迭代复杂度，直接注入工具设计提示词。
    tools = json.dumps(tool_catalog, ensure_ascii=False, indent=2)
    prompt = fixed_tool_set_prompt.format(
        background_info=background_info,
        available_tools=tools,
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
    )
    if parsed is None:
        return None, None, None, None, None
    task, workflow, restrict = parsed
    all_content = (
        f"<task>{task}</task>\n"
        f"<tools>{tools}</tools>\n"
        f"<restriction>{restrict}</restriction>\n"
        f"<workflow>{workflow}</workflow>"
    )
    return all_content, task, tools, workflow, restrict
