import json
import logging
from typing import Any, Dict, List


from tracesynth.functions.call_llms import ParseError
from tracesynth.functions.prompt import (
    plan_evaluation_system_prompt,
    plan_evaluation_user_prompt,
)
from tracesynth.graph.node_utils import (
    AgentState,
    validate_tool_call,
)
# from tracesynth.graph.plan_trajectory_node import _extract_xml_json, _tools_for_prompt
from tracesynth.functions.plan_trajectory import extract_xml_json, tools_for_prompt

logger = logging.getLogger(__name__)


def _parse_plan_evaluation_response(content: str) -> Dict[str, Any]:
    evaluation = extract_xml_json(content, "plan_evaluation")
    if not isinstance(evaluation, dict):
        raise ParseError("plan_evaluation must be a JSON object")
    if "is_valid" not in evaluation or not isinstance(evaluation["is_valid"], bool):
        raise ParseError("plan_evaluation.is_valid must be a boolean")
    if not (
        isinstance(evaluation.get("reasons"), list)
        or isinstance(evaluation.get("evaluation"), str)
        or isinstance(evaluation.get("reason"), str)
    ):
        raise ParseError("plan_evaluation must include concrete reasons")
    # 统一补齐可选字段，图节点合并确定性校验结果时可以直接 append/extend。
    evaluation.setdefault("reasons", [])
    evaluation.setdefault("issues", [])
    evaluation.setdefault("revision_suggestions", [])
    return evaluation


def _build_plan_evaluation_messages(state: AgentState) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": plan_evaluation_system_prompt,
        },
        {
            "role": "user",
            "content": plan_evaluation_user_prompt.format(
                fuzzy_task=state["fuzzy_task"],
                restrict=state.get("restrict", ""),
                available_tools=tools_for_prompt(state["checked_tools"]),
                plan_json=json.dumps(state.get("plan", []), ensure_ascii=False, indent=2),
            ),
        },
    ]


def _basic_plan_validation(plan: List[Dict[str, Any]], checked_tools: List[Dict[str, Any]]) -> List[str]:
    issues = []
    tool_names = {tool.get("name") for tool in checked_tools}
    for index, step in enumerate(plan):
        tool_name = step.get("tool_name")
        if tool_name not in tool_names:
            issues.append(f"plan[{index}] references unknown tool: {tool_name}")
        # 把计划步还原为标准 tool_call，复用 Solver 的工具合法性校验规则。
        tool_call = json.dumps(
            {"name": tool_name, "arguments": step.get("arguments", {})},
            ensure_ascii=False,
        )
        is_valid, error = validate_tool_call(tool_call, checked_tools)
        if not is_valid:
            issues.append(f"plan[{index}] invalid tool call: {error}")
    return issues


parse_plan_evaluation_response = _parse_plan_evaluation_response
build_plan_evaluation_messages = _build_plan_evaluation_messages
basic_plan_validation = _basic_plan_validation


