from typing import Dict, Any
from .call_llms import ParseError, call_and_parse, parse_json_object
from .prompt import fuzzy_task_prompt


def _parse_fuzzy_task_response(content: str):
    payload = parse_json_object(content)
    task = payload.get("task")
    bg = payload.get("background")

    if not isinstance(task, str) or not task.strip():
        raise ParseError("missing required JSON field: task")
    if not isinstance(bg, str) or not bg.strip():
        raise ParseError("missing required JSON field: background")

    return task.strip(), bg.strip()


def generate_fuzzy_task(cfg, initial_task_info, complexity=None):
    from tracesynth.configuration import SynthesisComplexity
    if complexity is None:
        complexity = SynthesisComplexity()
    # 模糊任务生成会隐藏过于直接的种子信息，同时保留足够背景供 MockUser/MockTool 交互。
    prompt = fuzzy_task_prompt.format(
        initial_task_info=initial_task_info,
        **complexity.to_prompt_vars(),
    )
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]
    parsed, _ = call_and_parse(
        cfg,
        messages,
        _parse_fuzzy_task_response,
        step_name="FuzzyTaskAgent",
        # json_mode=True,
    )
    if parsed is None:
        return None, None
    return parsed


def is_supervised_seed(seed_info: Dict[str, Any]) -> bool:
    # 同时具备 question 和 label 的样本视为监督 QA，主图会避免再次改写原问题。
    return bool(seed_info.get("label") and seed_info.get("question"))
