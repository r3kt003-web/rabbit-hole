import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_agent.config.settings import get_settings
from research_agent.graph.state import Finding, SubQuestion
from research_agent.llm.client import get_chat_model
from research_agent.llm.rate_limiter import call_with_rate_limit
from research_agent.tools.fetch import PageFetcher
from research_agent.tools.serper import SerperSearchTool

logger = logging.getLogger(__name__)

# Global concurrency semaphore for research workers
_worker_semaphore: asyncio.Semaphore | None = None


def get_worker_semaphore() -> asyncio.Semaphore:
    global _worker_semaphore
    if _worker_semaphore is None:
        settings = get_settings()
        _worker_semaphore = asyncio.Semaphore(settings.concurrency_limit)
    return _worker_semaphore


class WorkerInput(BaseModel):
    sub_question: SubQuestion
    trace_id: str = ""


class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(
        min_length=1,
        max_length=3,
        description="1 to 3 search queries to search Google for evidence",
    )


class ExtractedFindingItem(BaseModel):
    claim: str = Field(description="Factual claim answering part of the sub-question")
    evidence_quote: str = Field(
        description="Exact or near-exact quote from the source text supporting the claim"
    )
    source_url: str = Field(description="URL of the page where the evidence was found")
    source_title: str = Field(description="Title of the source webpage")
    confidence: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0"
    )


class WorkerFindingsOutput(BaseModel):
    findings: list[ExtractedFindingItem] = Field(
        default_factory=list,
        description="List of factual findings extracted from the retrieved pages",
    )


QUERY_PROMPT = """You are a research worker assistant.
Given a specific sub-question and search intent, generate 1 to 3 targeted Google search queries to find authoritative evidence.
Keep queries concise and keyword-focused."""

EXTRACT_FINDINGS_PROMPT = """You are an expert fact extractor.
Analyze the provided web page text and extract factual findings that directly address the sub-question.
Every finding MUST include:
1. claim: the specific factual statement.
2. evidence_quote: exact verbatim text from the source supporting the claim.
3. source_url: exact URL of the source page.
4. source_title: title of the page.
5. confidence: a score from 0.0 to 1.0 indicating factual confidence.

Criteria to satisfy: {answer_criteria}

If no relevant information answering the question is present in the text, return an empty findings list."""


class ResearchWorker:
    """Worker that handles a single sub-question: generates queries, searches Serper, fetches pages, extracts findings."""

    def __init__(
        self,
        search_tool: SerperSearchTool | None = None,
        fetch_tool: PageFetcher | None = None,
        model_name: str | None = None,
    ):
        settings = get_settings()
        self.search_tool = search_tool or SerperSearchTool()
        self.fetch_tool = fetch_tool or PageFetcher(
            timeout=min(15.0, settings.worker_timeout_seconds)
        )
        self.model_name = model_name or settings.researcher_model
        self.timeout = settings.worker_timeout_seconds
        self.max_retries = settings.worker_max_retries

    async def generate_queries(self, sub_question: SubQuestion) -> list[str]:
        """Generates 1 to 3 search queries for the sub-question."""
        try:
            chat_model = get_chat_model(self.model_name, temperature=0.0)
            structured_model = chat_model.with_structured_output(QueryGenerationOutput)

            messages = [
                SystemMessage(content=QUERY_PROMPT),
                HumanMessage(
                    content=f"Sub-question: {sub_question.question}\nSearch Intent: {sub_question.search_intent}"
                ),
            ]

            result = await call_with_rate_limit(structured_model.ainvoke, messages)
            if isinstance(result, QueryGenerationOutput) and result.queries:
                return result.queries[:3]
            elif isinstance(result, dict) and "queries" in result:
                return result["queries"][:3]
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.warning(f"Error generating queries for {sub_question.id}: {exc}")

        # Fallback to sub-question and search intent as queries
        return [sub_question.question, sub_question.search_intent][:2]

    async def execute(self, sub_question: SubQuestion) -> list[Finding]:
        """Executes full research for a single sub-question with timeout and retries."""
        sem = get_worker_semaphore()
        async with sem:
            for attempt in range(self.max_retries + 1):
                try:
                    return await asyncio.wait_for(
                        self._research_pipeline(sub_question),
                        timeout=self.timeout,
                    )
                except TimeoutError:
                    logger.warning(
                        f"Worker timeout on {sub_question.id} (attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    if attempt >= self.max_retries:
                        return []
                    await asyncio.sleep(1.0 * (2**attempt))
                except (RuntimeError, ValueError, TypeError, httpx.HTTPError) as exc:
                    logger.error(
                        f"Worker error on {sub_question.id} (attempt {attempt + 1}/{self.max_retries + 1}): {exc}"
                    )
                    if attempt >= self.max_retries:
                        return []
                    await asyncio.sleep(1.0 * (2**attempt))
            return []

    async def _research_pipeline(self, sub_question: SubQuestion) -> list[Finding]:
        # 1. Generate search queries (1 LLM call)
        queries = await self.generate_queries(sub_question)

        # 2. Search Serper for each query
        search_tasks = [self.search_tool.search(q, num_results=4) for q in queries]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Collect unique candidate URLs and their snippets
        urls_to_fetch: list[tuple[str, str]] = []  # (url, title)
        seen_urls: set[str] = set()

        for res in search_results:
            if isinstance(res, Exception) or res.error:
                continue
            for item in res.results:
                if item.link and item.link not in seen_urls:
                    seen_urls.add(item.link)
                    urls_to_fetch.append((item.link, item.title))
                if len(urls_to_fetch) >= 3:
                    break
            if len(urls_to_fetch) >= 3:
                break

        if not urls_to_fetch:
            logger.warning(f"No search results found for {sub_question.id}")
            return []

        # 3. Fetch top pages
        fetch_tasks = [self.fetch_tool.fetch(url) for url, _ in urls_to_fetch]
        pages = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        valid_pages: list[tuple[str, str, str]] = []  # (url, title, text)
        for idx, p in enumerate(pages):
            if isinstance(p, Exception) or not p.success or not p.text.strip():
                continue
            url = p.url
            title = p.title or urls_to_fetch[idx][1] or "Source"
            valid_pages.append((url, title, p.text))

        if not valid_pages:
            logger.warning(f"No pages could be extracted for {sub_question.id}")
            return []

        # 4. Extract findings (1 LLM call combining valid page texts)
        combined_text_snippets = []
        for url, title, text in valid_pages:
            snippet = text[:4000]  # Take top relevant text from each page
            combined_text_snippets.append(
                f"Source: {title} ({url})\nContent:\n{snippet}\n---"
            )

        page_corpus = "\n\n".join(combined_text_snippets)

        chat_model = get_chat_model(self.model_name, temperature=0.0)
        structured_model = chat_model.with_structured_output(WorkerFindingsOutput)

        prompt_content = (
            f"Sub-Question: {sub_question.question}\n"
            f"Search Intent: {sub_question.search_intent}\n"
            f"Answer Criteria: {sub_question.answer_criteria}\n\n"
            f"Retrieved Sources:\n{page_corpus}"
        )

        messages = [
            SystemMessage(
                content=EXTRACT_FINDINGS_PROMPT.format(
                    answer_criteria=sub_question.answer_criteria
                )
            ),
            HumanMessage(content=prompt_content),
        ]

        findings: list[Finding] = []
        now_iso = datetime.now(UTC).isoformat()

        try:
            extraction = await call_with_rate_limit(structured_model.ainvoke, messages)
            extracted_items = []
            if isinstance(extraction, WorkerFindingsOutput):
                extracted_items = extraction.findings
            elif isinstance(extraction, dict) and "findings" in extraction:
                extracted_items = extraction["findings"]

            for item in extracted_items:
                if isinstance(item, dict):
                    item = ExtractedFindingItem(**item)
                findings.append(
                    Finding(
                        sub_question_id=sub_question.id,
                        claim=item.claim,
                        evidence_quote=item.evidence_quote,
                        source_url=item.source_url,
                        source_title=item.source_title,
                        confidence=item.confidence,
                        retrieved_at=now_iso,
                    )
                )
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.error(
                f"Error during findings extraction for {sub_question.id}: {exc}"
            )

        return findings


async def researcher_node(worker_input: dict[str, Any] | WorkerInput) -> dict[str, Any]:
    """LangGraph node for a single research worker, run in parallel via Send API."""
    if isinstance(worker_input, WorkerInput):
        sub_question = worker_input.sub_question
    elif isinstance(worker_input, dict) and "sub_question" in worker_input:
        sq_data = worker_input["sub_question"]
        if isinstance(sq_data, SubQuestion):
            sub_question = sq_data
        else:
            sub_question = SubQuestion(**sq_data)
    else:
        raise ValueError(f"Invalid input to researcher_node: {worker_input}")

    worker = ResearchWorker()
    findings = await worker.execute(sub_question)
    return {"findings": findings}
