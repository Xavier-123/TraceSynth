import json
import logging
import yaml
import concurrent.futures
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracesynth.graph.graph_virtual_tools import is_successful_final_state, run_agent


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
        "num_tool_categories": ("task_complexity", "num_tool_categories"),
        "num_pipeline_stages": ("task_complexity", "num_pipeline_stages"),
        "num_custom_tools": ("task_complexity", "num_custom_tools"),
        "distractor_tools": ("task_complexity", "distractor_tools"),
        "retrieval_rounds": ("iteration_complexity", "retrieval_rounds"),
        "info_gaps": ("iteration_complexity", "info_gaps"),
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


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate synthetic tool-use tasks from seed personas')
    parser.add_argument('--config', type=str, default=str(PROJECT_ROOT / 'configs' / 'tool_use_data_gen.yaml'),
                       help='Path to the configuration file')
    complexity = parser.add_argument_group('synthesis complexity overrides')
    complexity.add_argument('--num-tools', type=str, help='e.g. "2~4"')
    complexity.add_argument('--num-tool-categories', type=str, help='e.g. "2~3"')
    complexity.add_argument('--num-pipeline-stages', type=str, help='e.g. "4~6"')
    complexity.add_argument('--num-custom-tools', type=str, help='e.g. "1"')
    complexity.add_argument('--distractor-tools', type=str, help='e.g. "1~2"')
    complexity.add_argument('--retrieval-rounds', type=str, help='e.g. "1~2", use "0" for no iteration')
    complexity.add_argument('--info-gaps', type=str, help='e.g. "1~3"')
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config_path = Path(args.config).resolve()
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

    def get_processed_ids(log_file_path: str) -> Set[str]:
        """Get a set of already processed task IDs from the log file"""
        processed_ids = set()
        
        # If log file doesn't exist, return empty set
        if not Path(log_file_path).exists():
            # Create directory path if it doesn't exist
            Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
            return processed_ids
        
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        if "id" in log_entry:
                            processed_ids.add(log_entry["id"])
                    except json.JSONDecodeError:
                        # Skip invalid lines
                        continue
        except Exception as e:
            logger.warning(f"Warning: Error reading log file: {e}")
        
        return processed_ids

    def process_single_task(task_data: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Process a single task and return True if successful"""
        try:
            task_id = task_data["id"]
            persona = task_data.get("persona") or task_data.get("question") or task_data.get("background")
            if not persona:
                logger.error(f"Task {task_id} has no persona/question/background field")
                return False

            logger.info(f"Processing task: {task_id}")
            final_state = run_agent(
                seed_info={
                    "id": task_id,
                    "background": persona
                },
                run_config=config  # Pass the full config
            )
            return is_successful_final_state(final_state)
        except Exception as e:
            logger.error(f"Error processing task {task_data['id']}: {e}")
            return False

    def run_tasks_from_file(config: Dict[str, Any]):
        """Run math tasks from a JSONL file with concurrent processing"""
        # Get already processed IDs
        processed_ids = get_processed_ids(config["logging"]["task_file_path"])
        tasks_to_process: List[Dict[str, Any]] = []

        source_path = config["paths"]["data_file"]
        max_tasks = config["processing"].get("max_tasks")

        # ================= 1. 多源数据加载阶段 =================
        # 情况 A：如果传入的是本地目录
        if Path(source_path).is_dir():
            from datasets import load_dataset
            datasets = load_dataset(source_path, split="test")
            for dataset in datasets:
                task_id = dataset.get("id")
                if not task_id:
                    logger.warning("Skipping dataset row without id")
                    continue
                if task_id in processed_ids:
                    logger.info(f"Skipping processed task: {task_id}")
                    continue

                persona = dataset.get("persona") or dataset.get("question") or dataset.get("background")
                if not persona:
                    logger.warning(f"Skipping task {task_id} without persona/question/background")
                    continue
                task_data = {"persona": persona, "id": task_id}
                tasks_to_process.append(task_data)

                # Break if we've reached the max tasks limit
                if max_tasks and len(tasks_to_process) >= max_tasks:
                    break

        # 情况 B：如果传入的是本地单一文件
        elif Path(source_path).is_file():
            logger.info(f"Detected single file path: {source_path}")
            try:
                with open(source_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        task_data = json.loads(line.strip())

                        # Skip if already processed
                        if task_data['id'] in processed_ids:
                            logger.info(f"Skipping processed task: {task_data['id']}")
                            continue

                        tasks_to_process.append(task_data)

                        # Break if we've reached the max tasks limit
                        if max_tasks and len(tasks_to_process) >= max_tasks:
                            break

            except Exception as e:
                logger.error(f"Error reading single file {source_path}: {e}")
        else:
            logger.error(f"Data source does not exist: {source_path}")
        
        if not tasks_to_process:
            logger.info("No new tasks to process")
            return
        
        logger.info(f"Starting concurrent processing of {len(tasks_to_process)} tasks with {config['processing']['max_workers']} worker threads")
        
        # Process tasks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=config["processing"]["max_workers"]) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(process_single_task, task_data, config): task_data 
                for task_data in tasks_to_process
            }
            
            # Collect results
            completed_tasks = 0
            failed_tasks = 0
            
            for future in concurrent.futures.as_completed(future_to_task):
                task_data = future_to_task[future]
                try:
                    success = future.result()
                    if success:
                        completed_tasks += 1
                    else:
                        failed_tasks += 1
                except Exception as e:
                    logger.error(f"Task {task_data['id']} generated an exception: {e}")
                    failed_tasks += 1
            
            logger.info(f"🏁 Processing completed: {completed_tasks} successful, {failed_tasks} failed")

    # Run with configuration
    run_tasks_from_file(config)

if __name__ == "__main__":
    main()
