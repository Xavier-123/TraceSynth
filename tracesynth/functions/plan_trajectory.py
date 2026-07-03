import json
import logging
import re
from typing import Any, Dict, List

from tracesynth.configuration import SynthesisComplexity
from tracesynth.functions.call_llms import ParseError
from tracesynth.functions.prompt import (
    plan_trajectory_system_prompt,
    plan_trajectory_user_prompt,
)
from tracesynth.graph.node_utils import AgentState

logger = logging.getLogger(__name__)


def _extract_xml_json(content: str, tag: str) -> Any:
    matches = re.findall(rf"<{tag}>(.+?)</{tag}>", content or "", re.DOTALL)
    if not matches:
        raise ParseError(f"missing <{tag}> tag")
    raw_json = matches[-1].strip()
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON in <{tag}>: {exc}") from exc


def _parse_plan_response(content: str) -> List[Dict[str, Any]]:
    plan = _extract_xml_json(content, "plan")
    if not isinstance(plan, list) or not plan:
        raise ParseError("plan must be a non-empty JSON array")

    for index, step in enumerate(plan):
        if not isinstance(step, dict):
            raise ParseError(f"plan[{index}] must be an object")
        if not isinstance(step.get("tool_name"), str) or not step["tool_name"].strip():
            raise ParseError(f"plan[{index}] missing tool_name")
        if "arguments" not in step or not isinstance(step["arguments"], dict):
            raise ParseError(f"plan[{index}].arguments must be an object")
        step.setdefault("step_id", index + 1)
        step.setdefault("stage", "")
        step.setdefault("purpose", "")
        step.setdefault("depends_on", [])
    return plan


def _tools_for_prompt(checked_tools: List[Dict[str, Any]]) -> str:
    return json.dumps(checked_tools, ensure_ascii=False, indent=2)


def _build_plan_messages(state: AgentState, complexity: SynthesisComplexity) -> List[Dict[str, str]]:
    prior_evaluation = state.get("plan_evaluation") or {}
    return [
        {
            "role": "system",
            "content": plan_trajectory_system_prompt,
        },
        {
            "role": "user",
            "content": plan_trajectory_user_prompt.format(
                fuzzy_task=state["fuzzy_task"],
                task_background=state.get("task_background", ""),
                initial_workflow=state.get("initial_workflow", ""),
                restrict=state.get("restrict", ""),
                complexity_summary=complexity.to_prompt_vars()["complexity_summary"],
                available_tools=_tools_for_prompt(state["checked_tools"]),
                prior_evaluation=json.dumps(prior_evaluation, ensure_ascii=False),
            ),
        },
    ]


extract_xml_json = _extract_xml_json
parse_plan_response = _parse_plan_response
tools_for_prompt = _tools_for_prompt
build_plan_messages = _build_plan_messages


