"""Configuration loading helpers shared by command-line entry points."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


PATH_KEYS: Mapping[str, tuple[str, ...]] = {
    "logging": (
        "task_file_path",
        "solve_path",
        "failed_task_file_path",
        "already_processed_path",
    ),
    "paths": (
        "data_file",
        "solution_path",
    ),
}


def resolve_path(path_value: str, base_dir: Path) -> str:
    """Resolve a possibly relative config path against the config file directory."""
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def normalize_config_paths(
    config: Dict[str, Any],
    config_path: str | Path,
    *,
    path_keys: Mapping[str, Iterable[str]] = PATH_KEYS,
) -> Dict[str, Any]:
    """Return a copy of config with known path values resolved consistently."""
    normalized = deepcopy(config)
    base_dir = Path(config_path).resolve().parent

    for section, keys in path_keys.items():
        section_config = normalized.get(section)
        if not isinstance(section_config, dict):
            continue
        for key in keys:
            value = section_config.get(key)
            if isinstance(value, str) and value:
                section_config[key] = resolve_path(value, base_dir)

    return normalized


def load_run_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML run config and normalize path fields."""
    import yaml

    path = Path(config_path).resolve()
    with open(path, "r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return normalize_config_paths(raw_config, path)

