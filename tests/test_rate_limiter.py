import asyncio
import time

from research_agent.llm.rate_limiter import (
    AsyncTokenBucketRateLimiter,
    parse_retry_after,
)


def test_parse_retry_after():
    assert parse_retry_after("Rate limited. Please retry after 25s.") == 25.0
    assert parse_retry_after("Retry after 3.5 seconds") == 3.5
    assert parse_retry_after("No retry time here") is None


def test_rate_limiter_token_acquisition():
    # 60 RPM = 1 request per second
    limiter = AsyncTokenBucketRateLimiter(rpm=60, rpd=100)
    assert limiter.remaining_daily_quota == 100

    async def acquire_tokens():
        # First token should be immediate since bucket starts full
        start = time.monotonic()
        await limiter.acquire()
        assert limiter.remaining_daily_quota == 99
        return time.monotonic() - start

    elapsed = asyncio.run(acquire_tokens())
    assert elapsed < 0.2
