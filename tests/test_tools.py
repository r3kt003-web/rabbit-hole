import asyncio

import httpx

from research_agent.tools.cache import DiskCache
from research_agent.tools.fetch import PageFetcher
from research_agent.tools.serper import SerperSearchTool

# === Serper Tool Tests ===


def test_serper_missing_api_key(tmp_path):
    cache = DiskCache(cache_dir=tmp_path)
    tool = SerperSearchTool(api_key="", disk_cache=cache)
    result = asyncio.run(tool.search("test query"))
    assert result.error is not None
    assert "missing" in result.error.lower()
    assert len(result.results) == 0


def test_serper_successful_search_and_cache(tmp_path):
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        assert str(request.url) == "https://google.serper.dev/search"
        data = {
            "organic": [
                {
                    "title": "Quantum Supremacy",
                    "link": "https://example.com/quantum",
                    "snippet": "Recent breakthroughs in quantum computing...",
                    "position": 1,
                },
            ]
        }
        return httpx.Response(200, json=data)

    transport = httpx.MockTransport(mock_handler)
    cache = DiskCache(cache_dir=tmp_path)
    tool = SerperSearchTool(api_key="mock_key", disk_cache=cache, transport=transport)

    # First call: misses cache, calls mock transport
    res1 = asyncio.run(tool.search("quantum computing"))
    assert call_count == 1
    assert res1.from_cache is False
    assert len(res1.results) == 1

    # Second call: hits disk cache, no transport call
    res2 = asyncio.run(tool.search("quantum computing"))
    assert call_count == 1  # Unchanged!
    assert res2.from_cache is True
    assert res2.results[0].title == "Quantum Supremacy"


def test_serper_rate_limit_and_retry(tmp_path):
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, json={"message": "Rate limit exceeded"})
        return httpx.Response(
            200,
            json={
                "organic": [
                    {
                        "title": "Recovered",
                        "link": "https://example.com/rec",
                        "snippet": "ok",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(mock_handler)
    cache = DiskCache(cache_dir=tmp_path)
    # Set timeout low and max_retries=1
    tool = SerperSearchTool(
        api_key="mock_key", max_retries=1, disk_cache=cache, transport=transport
    )

    res = asyncio.run(tool.search("rate limit test"))
    assert call_count == 2
    assert res.error is None
    assert len(res.results) == 1
    assert res.results[0].title == "Recovered"


def test_serper_persistent_error_degrades_gracefully(tmp_path):
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    transport = httpx.MockTransport(mock_handler)
    cache = DiskCache(cache_dir=tmp_path)
    tool = SerperSearchTool(
        api_key="mock_key", max_retries=0, disk_cache=cache, transport=transport
    )

    res = asyncio.run(tool.search("failing query"))
    assert res.error is not None
    assert "500" in res.error
    assert len(res.results) == 0


# === Page Fetcher Tests ===


def test_fetcher_successful_extraction_and_cache(tmp_path):
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Scientific Breakthroughs</title></head>
        <body>
            <main>
                <article>
                    <h1>Scientific Breakthroughs in 2024</h1>
                    <p>Scientists have demonstrated viable nuclear fusion net energy gain with repeatable yields across multiple laboratories worldwide.</p>
                </article>
            </main>
        </body>
        </html>
        """
        return httpx.Response(
            200, headers={"content-type": "text/html; charset=utf-8"}, text=html
        )

    transport = httpx.MockTransport(mock_handler)
    cache = DiskCache(cache_dir=tmp_path)
    fetcher = PageFetcher(disk_cache=cache, transport=transport)

    # First fetch: calls mock transport
    page1 = asyncio.run(fetcher.fetch("https://example.com/fusion"))
    assert call_count == 1
    assert page1.success is True
    assert page1.from_cache is False
    assert "fusion net energy gain" in page1.text

    # Second fetch: from disk cache
    page2 = asyncio.run(fetcher.fetch("https://example.com/fusion"))
    assert call_count == 1
    assert page2.success is True
    assert page2.from_cache is True
    assert "fusion net energy gain" in page2.text


def test_fetcher_paywalled_or_403(tmp_path):
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Forbidden")

    transport = httpx.MockTransport(mock_handler)
    cache = DiskCache(cache_dir=tmp_path)
    fetcher = PageFetcher(disk_cache=cache, transport=transport)

    page = asyncio.run(fetcher.fetch("https://paywall.com/article"))
    assert page.success is False
    assert "403" in page.error


def test_fetcher_unsupported_content_type(tmp_path):
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4..."
        )

    transport = httpx.MockTransport(mock_handler)
    cache = DiskCache(cache_dir=tmp_path)
    fetcher = PageFetcher(disk_cache=cache, transport=transport)

    page = asyncio.run(fetcher.fetch("https://example.com/doc.pdf"))
    assert page.success is False
    assert "Unsupported content-type" in page.error
