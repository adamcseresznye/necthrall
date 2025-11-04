from typing import Callable, Any, Dict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)
from loguru import logger
import asyncio
import httpx


class LLMRetryConfig:
    def __init__(
        self, max_attempts: int = 3, min_backoff: float = 1.0, max_backoff: float = 4.0
    ):
        # Note: default 3 attempts (initial + 2 retries). If caller wants three retries (4 attempts)
        # they can pass max_attempts=4. This keeps default aligned to requirements while allowing
        # configurability in tests.
        self.max_attempts = max_attempts
        self.min_backoff = min_backoff
        self.max_backoff = max_backoff


def tenacity_retry_decorator(retry_config: LLMRetryConfig):
    """Return a tenacity retry decorator configured for LLM calls.

    Retries on Exception by default but callers can wrap functions that raise specific
    LLM-related exceptions.
    """

    def _decorator(fn: Callable[..., Any]):
        # Use predicate-based retry to avoid retrying on non-retryable exceptions
        return retry(
            stop=stop_after_attempt(retry_config.max_attempts),
            wait=wait_exponential(
                multiplier=1, min=retry_config.min_backoff, max=retry_config.max_backoff
            ),
            retry=retry_if_exception(is_retryable_exception),
            before_sleep=before_sleep_log(logger, "WARNING"),
        )(fn)

    return _decorator


def is_retryable_exception(exc: Exception) -> bool:
    """Lightweight classifier for LLM-related exceptions to decide retryability."""
    # Rate limiting / quota / resource issues
    from google.api_core.exceptions import ResourceExhausted, InvalidArgument

    if isinstance(exc, ResourceExhausted):
        return True
    if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException)):
        return True
    # Authentication issues should not be retried
    if isinstance(exc, InvalidArgument):
        return False
    # Network errors
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError)):
        return True

    # Default to retry to be conservative for transient unknown errors
    return True
