import asyncio
import logging

import httpx
from pydantic import BaseModel, Field

from research_agent.config.settings import get_settings
from research_agent.tools.cache import DiskCache

logger = logging.getLogger(__name__)

SERPER_API_URL = "https://google.serper.dev/search"


class SearchResultItem(BaseModel):
    title: str = Field(description="Title of the search result")
    link: str = Field(description="URL of the search result")
    snippet: str = Field(
        default="", description="Snippet or text preview from Google search"
    )
    position: int = Field(default=0, description="Position in search results")


class SearchResponse(BaseModel):
    query: str = Field(description="Original search query")
    results: list[SearchResultItem] = Field(
        default_factory=list, description="List of search result items"
    )
    error: str | None = Field(
        default=None, description="Error message if the search failed or degraded"
    )
    total_results: int = Field(default=0, description="Number of results found")
    from_cache: bool = Field(
        default=False, description="Whether this response came from local disk cache"
    )


class SerperSearchTool:
    """Wrapper for Serper Google Search API with disk caching, retry, rate limit handling, and graceful degradation."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 1,
        disk_cache: DiskCache | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.api_key = api_key if api_key is not None else get_settings().serper_api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache = disk_cache or DiskCache()
        self.transport = transport

    async def search(self, query: str, num_results: int = 5) -> SearchResponse:
        """Executes a search query against Serper API, using disk cache if available.

        Handles rate limits (429) and network errors by retrying once with exponential backoff.
        If failures persist, it degrades gracefully and returns an empty result list with an error note.
        """
        # Check disk cache first to save API calls
        cached = self.cache.get_search(query)
        if cached:
            items = [SearchResultItem(**item) for item in cached.get("results", [])]
            return SearchResponse(
                query=query,
                results=items,
                total_results=len(items),
                from_cache=True,
            )

        if not self.api_key:
            logger.warning("Serper API key is not configured.")
            return SearchResponse(
                query=query,
                error="Serper API key is missing. Set SERPER_API_KEY in environment or .env.",
            )

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": num_results}

        attempt = 0
        backoff_seconds = 1.0

        while attempt <= self.max_retries:
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout, transport=self.transport
                ) as client:
                    response = await client.post(
                        SERPER_API_URL, headers=headers, json=payload
                    )

                    if response.status_code == 200:
                        data = response.json()
                        items: list[SearchResultItem] = []
                        organic = data.get("organic", [])
                        for idx, item in enumerate(organic):
                            items.append(
                                SearchResultItem(
                                    title=item.get("title", "Untitled"),
                                    link=item.get("link", ""),
                                    snippet=item.get("snippet", ""),
                                    position=item.get("position", idx + 1),
                                )
                            )

                        # Save to disk cache for future re-runs
                        self.cache.set_search(
                            query,
                            {
                                "results": [item.model_dump() for item in items],
                                "total_results": len(items),
                            },
                        )

                        return SearchResponse(
                            query=query,
                            results=items,
                            total_results=len(items),
                            from_cache=False,
                        )

                    elif response.status_code == 429:
                        attempt += 1
                        logger.warning(
                            f"Serper rate limit hit (429) for query '{query}'. "
                            f"Attempt {attempt}/{self.max_retries + 1}. Backing off {backoff_seconds}s."
                        )
                        if attempt <= self.max_retries:
                            await asyncio.sleep(backoff_seconds)
                            backoff_seconds *= 2.0
                            continue
                        else:
                            return SearchResponse(
                                query=query,
                                error="Serper rate limit exceeded after retries.",
                                results=[],
                            )
                    else:
                        error_msg = f"Serper API returned HTTP {response.status_code}: {response.text[:200]}"
                        logger.error(error_msg)
                        return SearchResponse(query=query, error=error_msg, results=[])

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                attempt += 1
                logger.warning(
                    f"Network error querying Serper for '{query}': {exc}. "
                    f"Attempt {attempt}/{self.max_retries + 1}."
                )
                if attempt <= self.max_retries:
                    await asyncio.sleep(backoff_seconds)
                    backoff_seconds *= 2.0
                    continue
                else:
                    return SearchResponse(
                        query=query,
                        error=f"Serper request failed after retries due to network/timeout error: {exc}",
                        results=[],
                    )
            except Exception:
                logger.exception("Unexpected error in Serper search")
                return SearchResponse(
                    query=query,
                    error="Unexpected Serper error",
                    results=[],
                )

        return SearchResponse(
            query=query, error="Search failed after max retries.", results=[]
        )
