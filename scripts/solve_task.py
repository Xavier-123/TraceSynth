import json
import logging
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracesynth.concurrency import run_concurrent_tasks

from tracesynth.graph.graph_solve_task import run_agent
from tracesynth.config_loader import load_run_config
from tracesynth.io import SeedRecordError, normalize_task_record, read_processed_ids


def main():
    parser = argparse.ArgumentParser(description='Run solve-only tasks from synthesized manifest JSONL')
    parser.add_argument('--config', type=str, default='configs/solve_task.yaml',
                       help='Path to the configuration file (default: configs/solve_task.yaml)')
    args = parser.parse_args()

    config = load_run_config(args.config)

    already_processed_file = Path(config["logging"]["already_processed_path"])
    already_processed_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    def process_single_task(task_data: Dict[str, Any], run_config: Dict[str, Any]) -> bool:
        try:
            task = normalize_task_record(task_data)
            logger.info(f"Processing task: {task.id}")
            new_task: dict[str, Any] = {
                "id": task.id,
                "fuzzy_task": task.fuzzy_task,
                "checked_tools": task.checked_tools,
            }
            run_agent(seed_info=new_task, run_config=run_config)
            return True
        except SeedRecordError as e:
            logger.error(f"Invalid task record {task_data.get('id', 'unknown')}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error processing task {task_data.get('id', 'unknown')}: {e}")
            return False

    def run_tasks_from_file(run_config: Dict[str, Any]):
        processed_ids = read_processed_ids(run_config["logging"]["already_processed_path"])
        tasks_to_process: List[Dict[str, Any]] = []

        with open(run_config["paths"]["data_file"], 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    task_data = json.loads(line.strip())
                    if task_data['id'] in processed_ids:
                        logger.info(f"Skipping processed task: {task_data['id']}")
                        continue
                    tasks_to_process.append(task_data)
                    if run_config["processing"]["max_tasks"] and len(tasks_to_process) >= run_config["processing"]["max_tasks"]:
                        break
                except json.JSONDecodeError:
                    logger.warning("Warning: Skipping invalid JSON line")
                except Exception as e:
                    logger.error(f"Error loading task: {e}")

        if not tasks_to_process:
            logger.info("No new tasks to process")
            return

        logger.info(
            f"Starting concurrent processing of {len(tasks_to_process)} tasks "
            f"with {run_config['processing']['max_workers']} worker threads"
        )

        completed_tasks, failed_tasks = run_concurrent_tasks(
            tasks_to_process,
            lambda task_data: process_single_task(task_data, run_config),
            max_workers=run_config["processing"]["max_workers"],
            get_id=lambda task_data: str(task_data.get("id", "unknown")),
            logger=logger,
        )
        logger.info(f"Processing completed: {completed_tasks} successful, {failed_tasks} failed")

    run_tasks_from_file(config)


if __name__ == "__main__":
    main()
