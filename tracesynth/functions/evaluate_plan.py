import json
import logging
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

from tracesynth.functions.call_llms import ParseError
from tracesynth.functions.plan_trajectory import extract_xml_json, tools_for_prompt
from tracesynth.functions.prompt import (
    plan_evaluation_system_prompt,
    plan_evaluation_user_prompt,
)
from tracesynth.graph.node_utils import AgentState, validate_tool_call

logger = logging.getLogger(__name__)

FAILURE_CATEGORY_VALID = "valid"
FAILURE_CATEGORY_MISSING_REQUIRED = "deterministic_missing_required"
FAILURE_CATEGORY_UNKNOWN_TOOL = "deterministic_unknown_tool"
FAILURE_CATEGORY_INVALID_TOOL_CALL = "deterministic_invalid_tool_call"
FAILURE_CATEGORY_DETERMINISTIC_MULTIPLE = "deterministic_multiple"
FAILURE_CATEGORY_SEMANTIC_REJECT = "semantic_reject"
FAILURE_CATEGORY_PARSE_FAILURE = "parse_failure"
FAILURE_CATEGORY_REVISION_EXHAUSTED = "revision_exhausted"


class PlanEvaluationStats:
    """Thread-safe counters for plan-evaluation outcomes across a batch run."""

    _lock = threading.Lock()
    _counts: Dict[str, int] = defaultdict(int)

    @classmethod
    def record(cls, category: str) -> None:
        with cls._lock:
            cls._counts[category] += 1

    @classmethod
    def summary(cls) -> Dict[str, int]:
        with cls._lock:
            return dict(cls._counts)

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._counts.clear()

    @classmethod
    def log_summary(cls) -> None:
        summary = cls.summary()
        if not summary:
            return
        parts = ", ".join(f"{key}={value}" for key, value in sorted(summary.items()))
        logger.info("Plan evaluation outcome summary: %s", parts)


def _classify_issue(issue: str) -> str:
    lowered = issue.lower()
    if "references unknown tool" in lowered:
        return FAILURE_CATEGORY_UNKNOWN_TOOL
    if "missing required fields" in lowered:
        return FAILURE_CATEGORY_MISSING_REQUIRED
    return FAILURE_CATEGORY_INVALID_TOOL_CALL


def classify_validation_issues(issues: List[str]) -> str:
    if not issues:
        return FAILURE_CATEGORY_VALID
    categories = {_classify_issue(issue) for issue in issues}
    if len(categories) == 1:
        return categories.pop()
    return FAILURE_CATEGORY_DETERMINISTIC_MULTIPLE


def _build_revision_suggestions_from_issues(issues: List[str]) -> List[str]:
    suggestions: List[str] = []
    for issue in issues:
        if "missing required fields" in issue:
            suggestions.append(
                f"For {issue}, add every required argument key with a concrete value or a "
                "reference to a prior step output (e.g. \"optimized queries from step 1\")."
            )
        elif "references unknown tool" in issue:
            suggestions.append(
                f"For {issue}, replace the tool with one from the available tools list."
            )
        else:
            suggestions.append(f"Fix structural tool-call issue: {issue}")
    return suggestions


def _validate_dependencies(plan: List[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    step_ids = {step.get("step_id", idx + 1) for idx, step in enumerate(plan)}
    for index, step in enumerate(plan):
        current_id = step.get("step_id", index + 1)
        for dep in step.get("depends_on") or []:
            if dep not in step_ids:
                issues.append(f"plan[{index}] depends_on references missing step_id: {dep}")
            elif dep >= current_id:
                issues.append(f"plan[{index}] depends_on references a non-earlier step_id: {dep}")
    return issues


def basic_plan_validation(plan: List[Dict[str, Any]], checked_tools: List[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    tool_names = {tool.get("name") for tool in checked_tools}
    for index, step in enumerate(plan):
        tool_name = step.get("tool_name")
        if tool_name not in tool_names:
            issues.append(f"plan[{index}] references unknown tool: {tool_name}")
            continue
        tool_call = json.dumps(
            {"name": tool_name, "arguments": step.get("arguments", {})},
            ensure_ascii=False,
        )
        is_valid, error = validate_tool_call(tool_call, checked_tools)
        if not is_valid:
            issues.append(f"plan[{index}] invalid tool call: {error}")
    issues.extend(_validate_dependencies(plan))
    return issues


def build_preflight_evaluation(issues: List[str]) -> Dict[str, Any]:
    category = classify_validation_issues(issues)
    return {
        "is_valid": False,
        "failure_category": category,
        "reasons": [
            "Deterministic preflight validation failed before semantic evaluation.",
            "Fix the structural tool-call issues below before re-evaluating plan quality.",
        ],
        "issues": issues,
        "revision_suggestions": _build_revision_suggestions_from_issues(issues),
        "preflight_only": True,
    }


def resolve_failure_category(
    evaluation: Dict[str, Any],
    basic_issues: List[str],
    *,
    parse_failed: bool = False,
    revision_exhausted: bool = False,
) -> str:
    if parse_failed:
        return FAILURE_CATEGORY_PARSE_FAILURE
    if basic_issues:
        return classify_validation_issues(basic_issues)
    if evaluation.get("is_valid"):
        return FAILURE_CATEGORY_VALID
    if revision_exhausted:
        return FAILURE_CATEGORY_REVISION_EXHAUSTED
    if evaluation.get("preflight_only"):
        return str(evaluation.get("failure_category") or FAILURE_CATEGORY_DETERMINISTIC_MULTIPLE)
    return FAILURE_CATEGORY_SEMANTIC_REJECT


def annotate_plan_evaluation(
    evaluation: Dict[str, Any],
    basic_issues: List[str],
    *,
    parse_failed: bool = False,
    revision_exhausted: bool = False,
    record_stats: bool = True,
) -> Dict[str, Any]:
    if basic_issues:
        evaluation["is_valid"] = False
        evaluation.setdefault("issues", [])
        evaluation["issues"].extend(basic_issues)
        evaluation.setdefault("reasons", [])
        evaluation["reasons"].append("Basic deterministic validation found invalid tool calls.")
        evaluation.setdefault("revision_suggestions", [])
        for suggestion in _build_revision_suggestions_from_issues(basic_issues):
            if suggestion not in evaluation["revision_suggestions"]:
                evaluation["revision_suggestions"].append(suggestion)

    category = resolve_failure_category(
        evaluation,
        basic_issues,
        parse_failed=parse_failed,
        revision_exhausted=revision_exhausted and not evaluation.get("is_valid"),
    )
    evaluation["failure_category"] = category
    if record_stats:
        PlanEvaluationStats.record(category)
    return evaluation


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


parse_plan_evaluation_response = _parse_plan_evaluation_response
build_plan_evaluation_messages = _build_plan_evaluation_messages
