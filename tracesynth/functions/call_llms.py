import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ParseError(Exception):
    """Raised when LLM output cannot be parsed or validated."""


def messages_for_chat_completion(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert TraceSynth pseudo-tool messages for chat-completion compatibility."""
    converted_messages: List[Dict[str, str]] = []
    for message in messages:
        if message.get("role") == "tool":
            # 兼容不支持 OpenAI tool role 的普通聊天接口，把工具响应伪装成用户消息。
            converted = dict(message)
            converted["role"] = "user"
            converted_messages.append(converted)
        else:
            converted_messages.append(message)
    return converted_messages


def is_retryable_api_error(exc: Exception) -> bool:
    # 只重试瞬时错误：网络、超时、限流和 5xx；格式错误或鉴权失败应立即暴露。
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)):
        return True
    if isinstance(exc, APIError):
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None) if response is not None else None
        if status_code is not None and (status_code >= 500 or status_code == 429):
            return True
    return False


def _sleep_with_backoff(attempt: int, base: float) -> None:
    delay = base * (2 ** attempt) + random.uniform(0, 0.5)
    time.sleep(delay)


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _thinking_extra_body(api_base: Optional[str], use_thinking: bool) -> Dict[str, Any]:
    if api_base in ["https://apihub.agnes-ai.com/v1", "https://api-inference.modelscope.cn/v1"]:
        return {"enable_thinking": use_thinking}
    if api_base in ["https://api.siliconflow.cn/v1", "https://dashscope.aliyuncs.com/compatible-mode/v1"]:
        return {"chat_template_kwargs": {"enable_thinking": use_thinking}}
    return {}


def _build_chat_completion_request(
    *,
    api_base: Optional[str],
    model_name: str,
    messages: List[Dict[str, str]],
    max_tokens: Optional[int],
    temperature: Optional[float],
    use_thinking: bool,
    llm_params: Optional[Dict[str, Any]],
    extra_body: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    params = dict(llm_params or {})
    reserved_keys = {"model", "messages"} & params.keys()
    if reserved_keys:
        raise ValueError(f"llm_params cannot override reserved request keys: {sorted(reserved_keys)}")

    if temperature is not None and "temperature" not in params:
        params["temperature"] = temperature
    if max_tokens is not None and "max_completion_tokens" not in params and "max_tokens" not in params:
        params["max_completion_tokens"] = max_tokens

    nested_extra_body = params.pop("extra_body", None)
    if nested_extra_body is not None and not isinstance(nested_extra_body, dict):
        raise ValueError("llm_params.extra_body must be a mapping")
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError("extra_body must be a mapping")

    merged_extra_body = _thinking_extra_body(api_base, use_thinking)
    if nested_extra_body:
        merged_extra_body = _merge_dicts(merged_extra_body, nested_extra_body)
    if extra_body:
        merged_extra_body = _merge_dicts(merged_extra_body, extra_body)
    if merged_extra_body:
        params["extra_body"] = merged_extra_body

    return {
        "model": model_name,
        "messages": messages,
        **params,
    }


def create_chat_completion_with_retry(
    *,
    api_base: Optional[str],
    api_key: Optional[str],
    model_name: str,
    messages: List[Dict[str, str]],
    max_tokens: Optional[int],
    temperature: Optional[float],
    use_thinking: bool = False,
    api_max_retries: int = 3,
    api_retry_base: float = 1.0,
    llm_params: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    client = OpenAI(api_key=api_key, base_url=api_base, max_retries=0)
    last_exc: Optional[Exception] = None
    request = _build_chat_completion_request(
        api_base=api_base,
        model_name=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        use_thinking=use_thinking,
        llm_params=llm_params,
        extra_body=extra_body,
    )

    for attempt in range(api_max_retries):
        try:
            response = client.chat.completions.create(**request)

            return response.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            if not is_retryable_api_error(exc) or attempt >= api_max_retries - 1:
                raise
            # 指数退避加少量抖动，降低并发批处理时的重试碰撞。
            logger.warning(
                "API call failed (attempt %d/%d) for model %s: %s",
                attempt + 1,
                api_max_retries,
                model_name,
                exc,
            )
            _sleep_with_backoff(attempt, api_retry_base)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("API call failed without exception")


def call_llm_messages(
    messages: List[Dict[str, str]],
    api_base: Optional[str],
    api_key: Optional[str],
    model_name: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
    use_thinking: bool = False,
    api_max_retries: int = 3,
    api_retry_base: float = 1.0,
    llm_params: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    content = create_chat_completion_with_retry(
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        messages=messages_for_chat_completion(messages),
        max_tokens=max_tokens,
        temperature=temperature,
        use_thinking=use_thinking,
        api_max_retries=api_max_retries,
        api_retry_base=api_retry_base,
        llm_params=llm_params,
        extra_body=extra_body,
    )
    updated_messages = list(messages)
    updated_messages.append({"role": "assistant", "content": content})
    return updated_messages


def call_llm_api(
    user_prompt: str,
    system_prompt: str,
    api_base: Optional[str],
    api_key: Optional[str],
    model_name: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
    use_thinking: bool = False,
    api_max_retries: int = 3,
    api_retry_base: float = 1.0,
    llm_params: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return call_llm_messages(
        messages=messages,
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        use_thinking=use_thinking,
        api_max_retries=api_max_retries,
        api_retry_base=api_retry_base,
        llm_params=llm_params,
        extra_body=extra_body,
    )


def call_and_parse(
    cfg: Any,
    messages: List[Dict[str, str]],
    parse_fn: Callable[[str], T],
    *,
    step_name: str = "LLM",
    feedback_on_error: bool = True,
) -> tuple[Optional[T], List[Dict[str, str]]]:
    """Call LLM with API retry; on parse failure, resample up to parse_max_retries times."""
    parse_max_retries = getattr(cfg, "parse_max_retries", 2)
    total_attempts = parse_max_retries + 1
    last_error: Optional[str] = None
    working_messages = list(messages)
    last_content: Optional[str] = None
    last_result_messages: Optional[List[Dict[str, str]]] = None

    for attempt in range(total_attempts):
        try:
            # 第一层容错在 API 调用内部处理；这里拿到内容后再做结构解析。
            result_messages = call_llm_messages(
                messages=working_messages,
                api_base=cfg.api_base,
                api_key=cfg.api_key,
                model_name=cfg.model_name,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                use_thinking=cfg.use_thinking,
                api_max_retries=getattr(cfg, "api_max_retries", 3),
                api_retry_base=getattr(cfg, "api_retry_base", 1.0),
                llm_params=getattr(cfg, "llm_params", None),
                extra_body=getattr(cfg, "extra_body", None),
            )
            last_result_messages = result_messages
        except Exception as exc:
            logger.exception("%s API call failed; aborting parse loop: %s", step_name, str(exc))
            raise

        try:
            content = result_messages[-1]["content"]
            last_content = content
            parsed = parse_fn(content)
            if parsed is None:
                raise ParseError("parse_fn returned None")
            return parsed, result_messages
        except ParseError as exc:
            last_error = str(exc)
            if attempt < total_attempts - 1:
                logger.warning(
                    "%s parse failed (attempt %d/%d): %s; resampling",
                    step_name,
                    attempt + 1,
                    total_attempts,
                    last_error,
                )
                if feedback_on_error and last_content is not None:
                    # 解析失败时把上一版错误输出和错误原因回灌给模型，引导它按指定格式重采样。
                    working_messages = list(working_messages)
                    working_messages.append(
                        {"role": "assistant", "content": last_content},
                    )
                    working_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Previous output could not be parsed: {last_error}. "
                                "Please strictly follow the required output format and try again."
                            ),
                        },
                    )
            else:
                logger.warning(
                    "%s parse failed after %d attempts: %s",
                    step_name,
                    total_attempts,
                    last_error,
                )

    return None, last_result_messages or messages
