import asyncio
import logging

import httpx
import trafilatura
from pydantic import BaseModel, Field

from research_agent.tools.cache import DiskCache

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 (AutonomousResearchAgent/1.0)"
)


class FetchedPage(BaseModel):
    url: str = Field(description="URL of the fetched page")
    title: str | None = Field(default=None, description="Page title")
    text: str = Field(default="", description="Clean extracted text content")
    success: bool = Field(
        default=True, description="Whether fetching and extraction succeeded"
    )
    error: str | None = Field(
        default=None, description="Error reason if fetch or extraction failed"
    )
    from_cache: bool = Field(
        default=False, description="Whether content came from disk cache"
    )


class PageFetcher:
    """Fetches web pages via httpx with disk caching and cleanly extracts readable text with trafilatura."""

    def __init__(
        self,
        timeout: float = 12.0,
        max_retries: int = 1,
        max_content_length: int = 30000,
        user_agent: str = DEFAULT_USER_AGENT,
        disk_cache: DiskCache | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_content_length = max_content_length
        self.user_agent = user_agent
        self.cache = disk_cache or DiskCache()
        self.transport = transport

    async def fetch(self, url: str) -> FetchedPage:
        """Fetches a URL and extracts clean readable text, utilizing disk cache if available.

        Handles timeouts, paywalls/403s, empty content, and non-HTML payloads gracefully.
        """
        # Check disk cache first
        cached = self.cache.get_page(url)
        if cached:
            return FetchedPage(
                url=url,
                title=cached.get("title"),
                text=cached.get("text", ""),
                success=cached.get("success", True),
                error=cached.get("error"),
                from_cache=True,
            )

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        attempt = 0
        backoff_seconds = 1.0

        while attempt <= self.max_retries:
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout,
                    follow_redirects=True,
                    verify=True,
                    transport=self.transport,
                ) as client:
                    response = await client.get(url, headers=headers)

                    if response.status_code == 200:
                        content_type = response.headers.get("content-type", "").lower()
                        if (
                            "text" not in content_type
                            and "html" not in content_type
                            and "xml" not in content_type
                        ):
                            return FetchedPage(
                                url=url,
                                success=False,
                                error=f"Unsupported content-type: {content_type}",
                            )

                        html = response.text
                        if not html or not html.strip():
                            return FetchedPage(
                                url=url,
                                success=False,
                                error="Page returned empty response body.",
                            )

                        # Extract clean text using trafilatura
                        extracted_text = trafilatura.extract(
                            html,
                            include_comments=False,
                            include_tables=True,
                            favor_precision=True,
                        )

                        # Extract metadata for title
                        title = None
                        metadata = trafilatura.extract_metadata(html)
                        if metadata and metadata.title:
                            title = metadata.title

                        if not extracted_text or not extracted_text.strip():
                            return FetchedPage(
                                url=url,
                                title=title,
                                text="",
                                success=False,
                                error="No readable text could be extracted from page (possibly paywalled, dynamic JS, or empty).",
                            )

                        # Truncate if excessively long
                        if len(extracted_text) > self.max_content_length:
                            extracted_text = (
                                extracted_text[: self.max_content_length]
                                + "\n...[truncated]"
                            )

                        result = FetchedPage(
                            url=url,
                            title=title,
                            text=extracted_text,
                            success=True,
                            from_cache=False,
                        )

                        # Cache successful fetch
                        self.cache.set_page(
                            url,
                            {
                                "title": title,
                                "text": extracted_text,
                                "success": True,
                            },
                        )

                        return result

                    elif response.status_code in (401, 403):
                        return FetchedPage(
                            url=url,
                            success=False,
                            error=f"Access denied (HTTP {response.status_code}, likely paywalled or bot-blocked).",
                        )
                    elif response.status_code == 404:
                        return FetchedPage(
                            url=url,
                            success=False,
                            error="Page not found (HTTP 404).",
                        )
                    else:
                        attempt += 1
                        if attempt <= self.max_retries and response.status_code >= 500:
                            await asyncio.sleep(backoff_seconds)
                            backoff_seconds *= 2.0
                            continue
                        return FetchedPage(
                            url=url,
                            success=False,
                            error=f"HTTP request failed with status code {response.status_code}.",
                        )

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                attempt += 1
                if attempt <= self.max_retries:
                    await asyncio.sleep(backoff_seconds)
                    backoff_seconds *= 2.0
                    continue
                return FetchedPage(
                    url=url,
                    success=False,
                    error=f"Fetch failed after retries due to timeout/network error: {exc}",
                )
            except Exception:
                logger.exception("Unexpected error fetching %s", url)
                return FetchedPage(
                    url=url,
                    success=False,
                    error="Unexpected error during fetch",
                )

        return FetchedPage(
            url=url, success=False, error="Fetch failed after max retries."
        )
