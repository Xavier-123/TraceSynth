import logging
import concurrent.futures
from tqdm import tqdm
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracesynth.io import (
    InputConfig,
    SeedRecordError,
    load_seed_records,
    read_failed_ids,
    read_processed_ids,
    write_failure_record,
)


def resolve_path(path_value: str, base_dir: Path) -> str:
    """Resolve config paths relative to the config file location."""
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def normalize_config_paths(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    base_dir = config_path.parent
    for section, keys in {
        "logging": ("task_file_path", "solve_path", "failed_task_file_path"),
        "paths": ("data_file",),
    }.items():
        for key in keys:
            if key in config.get(section, {}):
                config[section][key] = resolve_path(config[section][key], base_dir)
    return config


def apply_complexity_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI complexity flags into config['synthesis']."""
    overrides: Dict[str, Dict[str, str]] = {"task_complexity": {}, "iteration_complexity": {}}
    mapping = {
        "num_tools": ("task_complexity", "num_tools"),
        "num_custom_tools": ("task_complexity", "num_custom_tools"),
        "distractor_tools": ("task_complexity", "distractor_tools"),
        "max_iterations": ("iteration_complexity", "max_iterations"),
    }
    for arg_name, (section, key) in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[section][key] = value

    if not overrides["task_complexity"] and not overrides["iteration_complexity"]:
        return config

    synthesis = dict(config.get("synthesis") or {})
    for section, values in overrides.items():
        if values:
            merged = dict(synthesis.get(section) or {})
            merged.update(values)
            synthesis[section] = merged
    config["synthesis"] = synthesis
    return config


def get_failed_task_path(run_config: Dict[str, Any]) -> str:
    logging_cfg = run_config.get("logging") or {}
    task_file_path = logging_cfg.get("task_file_path", "virtual_tool_use.jsonl")
    return logging_cfg.get("failed_task_file_path", f"{task_file_path}.failed")


def write_exception_failure(seed_info: Dict[str, Any], run_config: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
    return write_failure_record(
        get_failed_task_path(run_config),
        seed_info=seed_info,
        final_state={},
        stage="entrypoint",
        failure_type="exception",
        failure_reason=str(exc),
        extra={"exception_class": exc.__class__.__name__},
    )


def get_ids_to_skip(run_config: Dict[str, Any]) -> set[str]:
    processed_ids = read_processed_ids(run_config["logging"]["task_file_path"])
    retry_failed_tasks = run_config.get("processing", {}).get("retry_failed_tasks", True)
    if retry_failed_tasks:
        return processed_ids
    return processed_ids | read_failed_ids(get_failed_task_path(run_config))


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic tool-use tasks from supervised QA seeds')
    parser.add_argument('--config', type=str, default=str(PROJECT_ROOT / 'configs' / 'tool_use_data_gen.yaml'),
                       help='Path to the configuration file')
    complexity = parser.add_argument_group('synthesis complexity overrides')
    complexity.add_argument('--num-tools', type=str, help='e.g. "4~6"')
    complexity.add_argument('--num-custom-tools', type=str, help='e.g. "1"')
    complexity.add_argument('--distractor-tools', type=str, help='e.g. "1~2"')
    complexity.add_argument('--max-iterations', type=str, help='e.g. "1~2", use "0" for no iteration')
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    import yaml

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config = normalize_config_paths(config, config_path)
    config = apply_complexity_cli_overrides(config, args)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    input_config = InputConfig.from_run_config(config)

    try:
        from tracesynth.graph.graph_virtual_tools import save_architecture_diagram

        save_architecture_diagram(str(PROJECT_ROOT / "architecture.png"))
    except Exception as exc:
        logger.warning("Failed to save architecture diagram: %s", exc)

    def process_single_task(seed_info: Dict[str, Any], run_config: Dict[str, Any]) -> bool:
        try:
            from tracesynth.graph.graph_virtual_tools import is_successful_final_state, run_agent

            task_id = seed_info["id"]
            logger.info(f"Processing task: {task_id}")
            final_state = run_agent(seed_info=seed_info, run_config=run_config)
            return is_successful_final_state(final_state)
        except Exception as e:
            logger.error(f"Error processing task {seed_info.get('id', 'unknown')}: {e}")
            write_exception_failure(seed_info, run_config, e)
            return False

    def run_tasks_from_file(run_config: Dict[str, Any]):
        processed_ids = get_ids_to_skip(run_config)
        source_path = run_config["paths"]["data_file"]
        max_tasks = run_config["processing"].get("max_tasks")

        try:
            seed_records = load_seed_records(
                source_path,
                input_config=input_config,
                processed_ids=processed_ids,
                max_tasks=max_tasks,
            )
        except FileNotFoundError:
            logger.error(f"Data source does not exist: {source_path}")
            return
        except Exception as e:
            logger.error(f"Error loading seed records from {source_path}: {e}")
            return

        tasks_to_process = [record.to_seed_info() for record in seed_records]
        if not tasks_to_process:
            logger.info("No new tasks to process")
            return

        logger.info(
            f"Starting concurrent processing of {len(tasks_to_process)} tasks "
            f"with {run_config['processing']['max_workers']} worker threads"
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=run_config["processing"]["max_workers"]) as executor:
            future_to_task = {
                executor.submit(process_single_task, seed_info, run_config): seed_info
                for seed_info in tasks_to_process
            }

            completed_tasks = 0
            failed_tasks = 0
            total_tasks = len(tasks_to_process)

            with tqdm(
                total=total_tasks,
                desc="Synthesizing",
                unit="task",
                dynamic_ncols=True,
                colour="green",
            ) as pbar:
                for future in concurrent.futures.as_completed(future_to_task):
                    seed_info = future_to_task[future]
                    try:
                        success = future.result()
                        if success:
                            completed_tasks += 1
                        else:
                            failed_tasks += 1
                    except SeedRecordError as e:
                        logger.error(f"Task {seed_info.get('id', 'unknown')} invalid seed: {e}")
                        failed_tasks += 1
                    except Exception as e:
                        logger.error(f"Task {seed_info.get('id', 'unknown')} generated an exception: {e}")
                        failed_tasks += 1
                    finally:
                        pbar.update(1)
                        pbar.set_postfix(success=completed_tasks, failed=failed_tasks)

            logger.info(f"Processing completed: {completed_tasks} successful, {failed_tasks} failed")

    run_tasks_from_file(config)


if __name__ == "__main__":
    main()
