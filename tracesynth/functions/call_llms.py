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


def create_chat_completion_with_retry(
    *,
    api_base: Optional[str],
    api_key: Optional[str],
    model_name: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    use_thinking: bool = False,
    api_max_retries: int = 3,
    api_retry_base: float = 1.0,
) -> str:
    client = OpenAI(api_key=api_key, base_url=api_base, max_retries=0)
    last_exc: Optional[Exception] = None

    for attempt in range(api_max_retries):
        try:
            if api_base in ["https://apihub.agnes-ai.com/v1", "https://api-inference.modelscope.cn/v1"]:
                # Agnes/ModelScope 使用 enable_thinking 与 max_completion_tokens 控制思考和长度。
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    extra_body={
                        "enable_thinking": use_thinking,
                        "max_completion_tokens": max_tokens,
                    },
                )
            elif api_base in ["https://api.siliconflow.cn/v1", "https://dashscope.aliyuncs.com/compatible-mode/v1"]:
                # SiliconFlow/DashScope 把思考开关放在 chat_template_kwargs 中。
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": use_thinking},
                    },
                )
            else:
                logger.debug("Using default API base!!!")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    extra_body={
                    },
                )

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
    max_tokens: int,
    temperature: float,
    use_thinking: bool = False,
    api_max_retries: int = 3,
    api_retry_base: float = 1.0,
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
    max_tokens: int,
    temperature: float,
    use_thinking: bool = False,
    api_max_retries: int = 3,
    api_retry_base: float = 1.0,
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
