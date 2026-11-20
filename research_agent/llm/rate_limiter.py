import asyncio
import logging
import random
import re
import time
from collections.abc import Callable, Coroutine
from typing import Any

from research_agent.config.settings import get_settings

logger = logging.getLogger(__name__)


class AsyncTokenBucketRateLimiter:
    """Async token bucket rate limiter tracking requests per minute (RPM) and requests per day (RPD).

    Shared across all LLM calls to respect Gemini Free Tier limits.
    """

    def __init__(self, rpm: int | None = None, rpd: int | None = None):
        settings = get_settings()
        self.rpm = rpm or settings.llm_rpm
        self.rpd = rpd or settings.llm_rpd

        self.capacity = float(self.rpm)
        self.tokens = float(self.rpm)
        self.fill_rate = float(self.rpm) / 60.0  # tokens added per second
        self.last_update = time.monotonic()
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

        # Daily usage tracking
        self.day_start = time.time()
        self.daily_requests_count = 0

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop != loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    def _reset_daily_if_needed(self) -> None:
        now = time.time()
        # Reset if 24 hours have passed
        if now - self.day_start >= 86400:
            self.day_start = now
            self.daily_requests_count = 0

    def reset(self) -> None:
        """Resets daily request count and tokens to initial capacity."""
        self.day_start = time.time()
        self.daily_requests_count = 0
        self.tokens = float(self.rpm)
        self.last_update = time.monotonic()

    @property
    def remaining_daily_quota(self) -> int:
        self._reset_daily_if_needed()
        return max(0, self.rpd - self.daily_requests_count)

    async def acquire(self) -> None:
        """Wait until a rate limit token is available and ensure daily quota is not exceeded."""
        async with self._get_lock():
            self._reset_daily_if_needed()

            if self.daily_requests_count >= self.rpd:
                logger.warning(
                    f"Daily request quota reached: {self.daily_requests_count}/{self.rpd}. "
                    "Rate limiter will still attempt with caution."
                )

            while True:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.last_update = now

                # Add new tokens based on elapsed time
                self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    self.daily_requests_count += 1
                    return

                # Calculate sleep duration until next token is ready
                wait_time = (1.0 - self.tokens) / self.fill_rate
                await asyncio.sleep(max(wait_time, 0.05))


# Global rate limiter instance
_global_limiter: AsyncTokenBucketRateLimiter | None = None


def get_rate_limiter() -> AsyncTokenBucketRateLimiter:
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = AsyncTokenBucketRateLimiter()
    return _global_limiter


def reset_rate_limiter() -> None:
    """Resets the singleton global rate limiter."""
    if _global_limiter is not None:
        _global_limiter.reset()


def parse_retry_after(error_str: str) -> float | None:
    """Parse retry-after hints from error messages (e.g. 'retry after 20s' or 'RESOURCE_EXHAUSTED')."""
    match = re.search(r"retry\s*after\s*(\d+(?:\.\d+)?)\s*s?", error_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


async def call_with_rate_limit(
    coroutine_func: Callable[..., Coroutine[Any, Any, Any]],
    *args,
    max_retries: int = 5,
    base_backoff: float = 2.0,
    **kwargs,
) -> Any:
    """Executes an async LLM call with global rate limiting and jittered exponential backoff on 429.

    Honors retry-after hints and prevents crashing on resource exhaustion.
    """
    limiter = get_rate_limiter()
    attempt = 0

    while True:
        await limiter.acquire()
        try:
            return await coroutine_func(*args, **kwargs)
        except Exception as exc:
            err_text = str(exc).lower()
            is_transient = (
                "429" in err_text
                or "resource_exhausted" in err_text
                or "quota" in err_text
                or "rate limit" in err_text
                or "disconnected" in err_text
                or "remoteprotocolerror" in err_text
                or "connection error" in err_text
                or "connecterror" in err_text
                or "readtimeout" in err_text
                or "503" in err_text
                or "502" in err_text
                or "504" in err_text
            )

            if not is_transient or attempt >= max_retries:
                logger.error(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {exc}"
                )
                raise

            attempt += 1
            # Check for explicit retry-after hint
            hint = parse_retry_after(str(exc))
            if hint:
                sleep_seconds = hint + random.uniform(0.5, 2.0)
            else:
                sleep_seconds = (base_backoff * (2 ** (attempt - 1))) + random.uniform(
                    0.2, 1.5
                )

            logger.warning(
                f"[Rate Limiter 429] Resource exhausted or rate limited. "
                f"Retrying in {sleep_seconds:.1f}s (attempt {attempt}/{max_retries})..."
            )
            await asyncio.sleep(sleep_seconds)
