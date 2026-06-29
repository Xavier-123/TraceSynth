"""Unified input schema and loaders for supervised QA seed data."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Union

from pydantic import BaseModel, Field


class SeedRecordError(ValueError):
    """Raised when a raw row cannot be normalized into a SeedRecord."""


class InputConfig(BaseModel):
    """Field mapping for heterogeneous input sources."""

    question_fields: List[str] = Field(
        default_factory=lambda: ["question", "query"],
    )
    label_fields: List[str] = Field(
        default_factory=lambda: ["label", "answer", "gold"],
    )
    context_field: str = "context"
    id_field: str = "id"
    dataset_split: str = "test"
    legacy_persona_mode: bool = False

    @classmethod
    def from_run_config(cls, config: Optional[Dict[str, Any]] = None) -> "InputConfig":
        if not config:
            return cls()
        raw = config.get("input") or {}
        return cls(**{k: v for k, v in raw.items() if k in cls.model_fields})


class SeedRecord(BaseModel):
    """Canonical supervised seed sample."""

    id: str
    question: str
    label: str
    context: Optional[str] = None

    @property
    def context_present(self) -> bool:
        return bool(self.context and self.context.strip())

    def build_background_for_prompt(self) -> str:
        """Composite background for ToolSetGen and legacy prompt compatibility."""
        parts = [
            f"用户问题：{self.question}",
            f"标准答案（仅供工具设计与虚拟知识库构建，勿泄露给求解智能体）：{self.label}",
        ]
        if self.context_present:
            parts.append(f"参考上下文：{self.context}")
        return "\n".join(parts)

    def to_seed_info(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "label": self.label,
            "context": self.context,
            "background": self.build_background_for_prompt(),
        }


def _first_non_empty(row: Dict[str, Any], fields: List[str]) -> Optional[str]:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_context(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return "\n\n".join(parts) if parts else None
    text = str(value).strip()
    return text or None


def _stable_id(question: str, label: str) -> str:
    digest = hashlib.sha256(f"{question}\n{label}".encode("utf-8")).hexdigest()
    return f"seed-{digest[:16]}"


def normalize_seed_record(
    row: Dict[str, Any],
    input_config: Optional[InputConfig] = None,
) -> SeedRecord:
    """Normalize a raw JSON/dataset row into a SeedRecord."""
    cfg = input_config or InputConfig()
    if not isinstance(row, dict):
        raise SeedRecordError("seed row must be a JSON object")

    question = _first_non_empty(row, cfg.question_fields)
    label = _first_non_empty(row, cfg.label_fields)
    context = _normalize_context(row.get(cfg.context_field))

    if cfg.legacy_persona_mode and not question:
        question = _first_non_empty(row, ["persona", "background"])
    if cfg.legacy_persona_mode and not label and question:
        label = question

    if not question:
        raise SeedRecordError(
            f"missing question/query field; tried {cfg.question_fields}"
        )
    if not label:
        raise SeedRecordError(
            f"missing label field; tried {cfg.label_fields}"
        )

    record_id = row.get(cfg.id_field)
    if isinstance(record_id, str) and record_id.strip():
        record_id = record_id.strip()
    elif record_id is not None and str(record_id).strip():
        record_id = str(record_id).strip()
    else:
        record_id = _stable_id(question, label)

    return SeedRecord(
        id=record_id,
        question=question,
        label=label,
        context=context,
    )


def read_processed_ids(log_file_path: Union[str, Path]) -> Set[str]:
    """Read already-processed task ids from a JSONL progress log."""
    processed_ids: Set[str] = set()
    path = Path(log_file_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        return processed_ids

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict) and entry.get("id"):
                processed_ids.add(str(entry["id"]))
    return processed_ids


def load_seed_records(
    source_path: Union[str, Path],
    input_config: Optional[InputConfig] = None,
    processed_ids: Optional[Set[str]] = None,
    max_tasks: Optional[int] = None,
) -> List[SeedRecord]:
    """Load seed records from JSONL file or HuggingFace datasets directory."""
    cfg = input_config or InputConfig()
    skip_ids = processed_ids or set()
    records: List[SeedRecord] = []
    path = Path(source_path)

    def _append_row(row: Dict[str, Any]) -> None:
        record = normalize_seed_record(row, cfg)
        if record.id in skip_ids:
            return
        records.append(record)
        if max_tasks and len(records) >= max_tasks:
            return

    if path.is_dir():
        from datasets import load_dataset

        dataset = load_dataset(str(path), split=cfg.dataset_split)
        for row in dataset:
            if max_tasks and len(records) >= max_tasks:
                break
            if not isinstance(row, dict):
                continue
            try:
                _append_row(row)
            except SeedRecordError:
                continue
            if max_tasks and len(records) >= max_tasks:
                break
    elif path.is_file():
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if max_tasks and len(records) >= max_tasks:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                try:
                    _append_row(row)
                except SeedRecordError:
                    continue
    else:
        raise FileNotFoundError(f"data source does not exist: {path}")

    return records


def iter_seed_records(
    source_path: Union[str, Path],
    input_config: Optional[InputConfig] = None,
) -> Iterator[SeedRecord]:
    """Stream seed records without filtering by processed ids."""
    for record in load_seed_records(source_path, input_config=input_config):
        yield record


class TaskRecord(BaseModel):
    """Normalized synthesized task manifest row."""

    id: str
    fuzzy_task: str
    checked_tools: List[Dict[str, Any]]
    question: Optional[str] = None
    label: Optional[str] = None
    context_present: bool = False


def normalize_task_record(row: Dict[str, Any]) -> TaskRecord:
    if not isinstance(row, dict):
        raise SeedRecordError("task row must be a JSON object")
    task_id = row.get("id")
    fuzzy_task = row.get("fuzzy_task")
    checked_tools = row.get("checked_tools")
    if not task_id:
        raise SeedRecordError("task row missing id")
    if not fuzzy_task or not str(fuzzy_task).strip():
        raise SeedRecordError(f"task {task_id} missing fuzzy_task")
    if not isinstance(checked_tools, list) or not checked_tools:
        raise SeedRecordError(f"task {task_id} missing checked_tools")
    return TaskRecord(
        id=str(task_id),
        fuzzy_task=str(fuzzy_task).strip(),
        checked_tools=checked_tools,
        question=row.get("question"),
        label=row.get("label"),
        context_present=bool(row.get("context_present")),
    )


def validate_seed_info(seed_info: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize seed_info passed into the synthesis graph."""
    record = normalize_seed_record(seed_info)
    return record.to_seed_info()


def _normalize_answer_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text)
    return text


def extract_predicted_answer(solve_history: Any) -> Optional[str]:
    """Extract final answer text from solver trajectory."""
    if not isinstance(solve_history, list):
        return None
    for message in reversed(solve_history):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            return answer or None
    return None


def check_label_match(
    predicted_answer: Optional[str],
    label: str,
    *,
    skip: bool = False,
) -> Dict[str, Any]:
    """P0 label consistency check using normalized substring overlap."""
    if skip:
        return {
            "label_match_status": "skipped",
            "predicted_answer": predicted_answer,
            "label": label,
            "match_score": None,
        }

    if not predicted_answer:
        return {
            "label_match_status": "missing_answer",
            "predicted_answer": None,
            "label": label,
            "match_score": 0.0,
        }

    pred_norm = _normalize_answer_text(predicted_answer)
    label_norm = _normalize_answer_text(label)
    if not pred_norm or not label_norm:
        return {
            "label_match_status": "mismatch",
            "predicted_answer": predicted_answer,
            "label": label,
            "match_score": 0.0,
        }

    if label_norm in pred_norm or pred_norm in label_norm:
        status = "match"
        score = 1.0
    else:
        pred_tokens = set(pred_norm.split())
        label_tokens = set(label_norm.split())
        overlap = pred_tokens & label_tokens
        score = len(overlap) / max(len(label_tokens), 1)
        status = "match" if score >= 0.6 else "mismatch"

    return {
        "label_match_status": status,
        "predicted_answer": predicted_answer,
        "label": label,
        "match_score": score,
    }
