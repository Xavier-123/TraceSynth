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


def is_retryable_api_error(exc: Exception) -> bool:
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
    client = OpenAI(api_key=api_key, base_url=api_base)
    last_exc: Optional[Exception] = None

    for attempt in range(api_max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                extra_body={
                    "enable_thinking": use_thinking,
                    "max_completion_tokens": max_tokens,
                },
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            if not is_retryable_api_error(exc) or attempt >= api_max_retries - 1:
                raise
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
        messages=messages,
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
) -> tuple[Optional[T], List[Dict[str, str]]]:
    """Call LLM with API retry; on parse failure, resample up to parse_max_retries times."""
    parse_max_retries = getattr(cfg, "parse_max_retries", 2)
    total_attempts = parse_max_retries + 1
    last_error: Optional[str] = None

    for attempt in range(total_attempts):
        try:
            result_messages = call_llm_messages(
                messages=messages,
                api_base=cfg.api_base,
                api_key=cfg.api_key,
                model_name=cfg.model_name,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                use_thinking=cfg.use_thinking,
                api_max_retries=getattr(cfg, "api_max_retries", 3),
                api_retry_base=getattr(cfg, "api_retry_base", 1.0),
            )
            content = result_messages[-1]["content"]
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
            else:
                logger.warning(
                    "%s parse failed after %d attempts: %s",
                    step_name,
                    total_attempts,
                    last_error,
                )

    return None, messages
