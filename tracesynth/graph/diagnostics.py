"""Node-level diagnostics and failure artifact helpers for synthesis graphs."""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from functools import wraps
from time import perf_counter
from typing import Any, Callable, Dict, Optional

from langchain_core.runnables import RunnableConfig

from tracesynth.graph.node_utils import AgentState, build_failure
from tracesynth.io import write_failure_record


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trace_entry(node_name: str, status: str, started_at: str, started: float) -> Dict[str, Any]:
    return {
        "node": node_name,
        "status": status,
        "started_at": started_at,
        "duration_ms": round((perf_counter() - started) * 1000, 3),
    }


def _append_trace(state: Dict[str, Any], entry: Dict[str, Any]) -> list[Dict[str, Any]]:
    trace = list(state.get("node_trace") or [])
    trace.append(entry)
    return trace


def instrument_node(
    graph_name: str,
    node_name: str,
    node_fn: Callable[[AgentState, RunnableConfig], Dict[str, Any]],
) -> Callable[[AgentState, RunnableConfig], Dict[str, Any]]:
    """Wrap a graph node so both returned failures and exceptions identify their origin."""

    @wraps(node_fn)
    def wrapped(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        started_at = _utc_now()
        started = perf_counter()

        if state.get("breaked"):
            entry = _trace_entry(node_name, "skipped", started_at, started)
            entry["reason"] = "a previous node already failed"
            return {"node_trace": _append_trace(state, entry)}

        try:
            update = node_fn(state, config)
            if not isinstance(update, dict):
                raise TypeError(f"node returned {type(update).__name__}; expected dict")
        except Exception as exc:
            entry = _trace_entry(node_name, "failed", started_at, started)
            entry["failure_type"] = "node_exception"
            entry["exception_type"] = exc.__class__.__name__
            return build_failure(
                f"{exc.__class__.__name__}: {exc}",
                failure_node=node_name,
                failure_type="node_exception",
                exception_type=exc.__class__.__name__,
                exception_message=str(exc),
                failure_traceback=traceback.format_exc(),
                node_trace=_append_trace(state, entry),
                failure_graph=graph_name,
            )

        status = "failed" if update.get("breaked") else "success"
        entry = _trace_entry(node_name, status, started_at, started)
        if status == "failed":
            entry["failure_type"] = update.setdefault("failure_type", "node_failure")
            update.setdefault("failure_node", node_name)
            update.setdefault("failure_graph", graph_name)
        update["node_trace"] = _append_trace(state, entry)
        return update

    return wrapped


def failure_from_exception(
    state: Dict[str, Any],
    exc: Exception,
    *,
    graph_name: str,
    failure_node: str,
    failure_type: str,
) -> Dict[str, Any]:
    """Convert a non-node exception into a serializable failed state."""
    return {
        **state,
        **build_failure(
            f"{exc.__class__.__name__}: {exc}",
            failure_node=failure_node,
            failure_type=failure_type,
            exception_type=exc.__class__.__name__,
            exception_message=str(exc),
            failure_traceback=traceback.format_exc(),
            failure_graph=graph_name,
        ),
    }


def save_failure_artifacts(
    solve_path: str,
    final_state: Dict[str, Any],
    *,
    attempt_index: Optional[int] = None,
) -> str:
    """Persist the latest failure plus an immutable per-attempt snapshot."""
    os.makedirs(solve_path, exist_ok=True)
    serialized = json.dumps(final_state, ensure_ascii=False, indent=4, default=str) + "\n"
    latest_path = os.path.join(solve_path, "failed_state.json")
    with open(latest_path, "w", encoding="utf-8") as handle:
        handle.write(serialized)

    if attempt_index is not None:
        attempt_path = os.path.join(solve_path, f"failure_attempt_{attempt_index}.json")
        with open(attempt_path, "w", encoding="utf-8") as handle:
            handle.write(serialized)
    else:
        attempt_path = latest_path

    solve_history = final_state.get("solve_history")
    if isinstance(solve_history, list):
        with open(os.path.join(solve_path, "failed_solution.json"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(solve_history, ensure_ascii=False, indent=4, default=str) + "\n")

    tool_call_history = final_state.get("tool_call_history")
    if isinstance(tool_call_history, list):
        with open(os.path.join(solve_path, "tool_call_history.json"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(tool_call_history, ensure_ascii=False, indent=4, default=str) + "\n")

    return attempt_path


def emit_failure_summary(
    logger: logging.Logger,
    final_state: Dict[str, Any],
    *,
    task_id: Any,
    graph_name: str,
    diagnostic_path: Optional[str] = None,
) -> None:
    """Write detailed diagnostics to logs and a concise, flushed terminal line."""
    node = final_state.get("failure_node") or "unknown"
    failure_type = final_state.get("failure_type") or "generation_failed"
    reason = " ".join(str(final_state.get("failure_reason") or "unknown failure").split())
    path_text = os.path.abspath(diagnostic_path) if diagnostic_path else "unavailable"
    summary = (
        f"[TraceSynth][FAILED] id={task_id} graph={graph_name} node={node} "
        f"type={failure_type} reason={reason} diagnostic={path_text}"
    )
    has_handlers = logger.hasHandlers()
    has_terminal_handler = False
    current_logger: Optional[logging.Logger] = logger
    while current_logger is not None:
        for handler in current_logger.handlers:
            stream = getattr(handler, "stream", None)
            if stream in {sys.stderr, sys.stdout}:
                has_terminal_handler = True
        if not current_logger.propagate:
            break
        current_logger = current_logger.parent

    failure_traceback = final_state.get("failure_traceback")
    if failure_traceback:
        logger.error("%s\n%s", summary, failure_traceback)
    else:
        logger.error("%s", summary)
    # With no configured handlers, logging's lastResort already writes ERROR to stderr.
    if not logger.isEnabledFor(logging.ERROR) or (has_handlers and not has_terminal_handler):
        print(summary, file=sys.stderr, flush=True)


def persist_and_report_failure(
    final_state: Dict[str, Any],
    *,
    failed_task_path: Optional[str],
    solve_path: Optional[str],
    seed_info: Dict[str, Any],
    stage: str,
    logger: logging.Logger,
    graph_name: str,
    label_check: Optional[Dict[str, Any]] = None,
    lock: Any = None,
) -> Dict[str, Any]:
    """Persist a failure, report it, and never let diagnostics abort a batch."""
    diagnostic_path: Optional[str] = None
    if failed_task_path and solve_path:
        try:
            lock_context = lock if lock is not None else nullcontext()
            with lock_context:
                failure_record = write_failure_record(
                    failed_task_path,
                    seed_info=seed_info,
                    final_state=final_state,
                    stage=stage,
                    failure_type=final_state.get("failure_type") or "generation_failed",
                    failure_reason=final_state.get("failure_reason") or "unknown failure",
                    label_check=label_check,
                    diagnostic_dir=solve_path,
                )
                diagnostic_path = save_failure_artifacts(
                    solve_path,
                    final_state,
                    attempt_index=failure_record["attempt_index"],
                )
        except Exception as exc:
            original_node = final_state.get("failure_node")
            original_reason = final_state.get("failure_reason")
            final_state = failure_from_exception(
                {
                    **final_state,
                    "caused_by_failure_node": original_node,
                    "caused_by_failure_reason": original_reason,
                },
                exc,
                graph_name=graph_name,
                failure_node="__persistence__",
                failure_type="persistence_exception",
            )

    emit_failure_summary(
        logger,
        final_state,
        task_id=seed_info.get("id", "unknown"),
        graph_name=graph_name,
        diagnostic_path=diagnostic_path,
    )
    return final_state
