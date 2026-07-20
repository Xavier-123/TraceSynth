"""Canonical fixed tool catalog for RAG trajectory synthesis."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping


REQUIRED_TOOL_SCHEMA_NAMES = (
    "RetrievalRequest",
    "EvidenceItem",
    "AnswerAttempt",
)

QUERY_TOOL_NAMES = frozenset({"rewrite_query", "decompose_query"})
RETRIEVAL_TOOL_NAMES = frozenset(
    {"retrieve_dense", "retrieve_sparse", "retrieve_graph"}
)
POST_RETRIEVAL_TOOL_NAMES = frozenset(
    {"deduplicate_evidence", "rerank", "filter_low_quality"}
)
ANSWER_TOOL_NAME = "generate_answer"
CRITIQUE_TOOL_NAME = "critique_answer"


def _configurable(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    nested = config.get("configurable")
    return nested if isinstance(nested, Mapping) else config


def _load_type_schemas(config: Mapping[str, Any] | None) -> Dict[str, Dict[str, Any]]:
    raw = _configurable(config).get("tool_schemas")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "Missing required run configuration 'tool_schemas' with keys: "
            + ", ".join(REQUIRED_TOOL_SCHEMA_NAMES)
        )

    schemas: Dict[str, Dict[str, Any]] = {}
    for name in REQUIRED_TOOL_SCHEMA_NAMES:
        schema = raw.get(name)
        if not isinstance(schema, dict) or not schema:
            raise ValueError(f"tool_schemas.{name} must be a non-empty JSON Schema object")
        if schema.get("type") != "object":
            raise ValueError(f"tool_schemas.{name}.type must be 'object'")
        schemas[name] = deepcopy(schema)
    return schemas


def _tool(
    name: str,
    description: str,
    properties: Dict[str, Dict[str, Any]],
    required: List[str],
    outputs: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "outputs": outputs,
    }


def build_fixed_tool_catalog(config: Mapping[str, Any] | None) -> List[Dict[str, Any]]:
    """Build the immutable ten-tool catalog from configured domain schemas."""
    schemas = _load_type_schemas(config)
    retrieval_request = schemas["RetrievalRequest"]
    evidence_item = schemas["EvidenceItem"]
    answer_attempt = schemas["AnswerAttempt"]
    evidence_list = {"type": "array", "items": deepcopy(evidence_item)}

    return [
        _tool(
            "rewrite_query",
            "Query optimization: rewrite the input query.",
            {"query": {"type": "string", "description": "Original query."}},
            ["query"],
            {"type": "string"},
        ),
        _tool(
            "decompose_query",
            "Query optimization: decompose a complex query into subqueries.",
            {"query": {"type": "string", "description": "Original query."}},
            ["query"],
            {"type": "array", "items": {"type": "string"}},
        ),
        _tool(
            "retrieve_dense",
            "Retrieval: perform dense semantic retrieval.",
            {"request": deepcopy(retrieval_request)},
            ["request"],
            deepcopy(evidence_list),
        ),
        _tool(
            "retrieve_sparse",
            "Retrieval: perform sparse lexical retrieval.",
            {"request": deepcopy(retrieval_request)},
            ["request"],
            deepcopy(evidence_list),
        ),
        _tool(
            "retrieve_graph",
            "Retrieval: perform graph-based retrieval.",
            {"request": deepcopy(retrieval_request)},
            ["request"],
            deepcopy(evidence_list),
        ),
        _tool(
            "deduplicate_evidence",
            "Post-retrieval: remove duplicate evidence items.",
            {"evidence_list": deepcopy(evidence_list)},
            ["evidence_list"],
            deepcopy(evidence_list),
        ),
        _tool(
            "rerank",
            "Post-retrieval: rerank evidence by relevance.",
            {"evidence_list": deepcopy(evidence_list)},
            ["evidence_list"],
            deepcopy(evidence_list),
        ),
        _tool(
            "filter_low_quality",
            "Post-retrieval: filter low-quality evidence.",
            {"evidence_list": deepcopy(evidence_list)},
            ["evidence_list"],
            deepcopy(evidence_list),
        ),
        _tool(
            "generate_answer",
            "Answer generation: create an answer attempt from query and evidence.",
            {
                "query": {"type": "string", "description": "User query."},
                "evidence": deepcopy(evidence_list),
            },
            ["query", "evidence"],
            deepcopy(answer_attempt),
        ),
        _tool(
            "critique_answer",
            "Answer evaluation: judge an answer against the query and evidence.",
            {
                "query": {"type": "string", "description": "User query."},
                "answer": {"type": "string", "description": "Candidate answer."},
                "evidence": deepcopy(evidence_list),
            },
            ["query", "answer", "evidence"],
            {
                "type": "array",
                "prefixItems": [{"type": "boolean"}, {"type": "string"}],
                "minItems": 2,
                "maxItems": 2,
            },
        ),
    ]


def validate_fixed_tool_catalog(
    tools: Any,
    config: Mapping[str, Any] | None,
) -> List[Dict[str, Any]]:
    """Reject catalogs containing missing, reordered, renamed, or mutated tools."""
    expected = build_fixed_tool_catalog(config)
    if tools != expected:
        raise ValueError(
            "checked_tools must exactly match the canonical fixed RAG tool catalog"
        )
    return deepcopy(expected)


def validate_fixed_plan_sequence(plan: Any) -> List[str]:
    """Validate mandatory stages and the generate/critique closure."""
    if not isinstance(plan, list):
        return ["plan must be a list"]

    names = [step.get("tool_name") if isinstance(step, dict) else None for step in plan]
    issues: List[str] = []
    if not any(name in QUERY_TOOL_NAMES for name in names):
        issues.append("plan must include at least one query optimization tool")
    if not any(name in RETRIEVAL_TOOL_NAMES for name in names):
        issues.append("plan must include at least one retrieval tool")
    if not any(name in POST_RETRIEVAL_TOOL_NAMES for name in names):
        issues.append("plan must include at least one post-retrieval tool")

    generate_indices = [i for i, name in enumerate(names) if name == ANSWER_TOOL_NAME]
    critique_indices = [i for i, name in enumerate(names) if name == CRITIQUE_TOOL_NAME]
    if len(generate_indices) != 1:
        issues.append("plan must include exactly one generate_answer step")
    if len(critique_indices) != 1:
        issues.append("plan must include exactly one critique_answer step")

    if len(generate_indices) == 1:
        generate_index = generate_indices[0]
        evidence_indices = [
            i
            for i, name in enumerate(names)
            if name in RETRIEVAL_TOOL_NAMES or name in POST_RETRIEVAL_TOOL_NAMES
        ]
        if evidence_indices and generate_index <= max(evidence_indices):
            issues.append("generate_answer must run after all retrieval and post-retrieval steps")

    if len(generate_indices) == 1 and len(critique_indices) == 1:
        if critique_indices[0] != generate_indices[0] + 1:
            issues.append("critique_answer must immediately follow generate_answer")
        if critique_indices[0] != len(names) - 1:
            issues.append("critique_answer must be the final planned tool step")
    return issues
