import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional, Tuple

from tqdm import tqdm


def run_concurrent_tasks(
    items: list,
    process_fn: Callable[[Any], bool],
    *,
    max_workers: int,
    get_id: Callable[[Any], str] = lambda item: str(item.get("id", "unknown")),
    desc: str = "Processing",
    show_progress: bool = False,
    logger: Optional[logging.Logger] = None,
    on_exception: Optional[Callable[[Any, Exception], None]] = None,
) -> Tuple[int, int]:
    """Submit all items to a thread pool and return (completed, failed) counts."""
    if not items:
        return 0, 0

    completed_tasks = 0
    failed_tasks = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(process_fn, item): item
            for item in items
        }

        if show_progress:
            iterator = tqdm(
                as_completed(future_to_item),
                total=len(items),
                desc=desc,
                unit="task",
                dynamic_ncols=True,
                colour="green",
            )
        else:
            iterator = as_completed(future_to_item)

        for future in iterator:
            item = future_to_item[future]
            try:
                success = future.result()
                if success:
                    completed_tasks += 1
                else:
                    failed_tasks += 1
            except Exception as exc:
                if on_exception is not None:
                    on_exception(item, exc)
                elif logger is not None:
                    logger.error("Task %s generated an exception: %s", get_id(item), exc)
                failed_tasks += 1
            finally:
                if show_progress and isinstance(iterator, tqdm):
                    iterator.set_postfix(success=completed_tasks, failed=failed_tasks)

    return completed_tasks, failed_tasks
