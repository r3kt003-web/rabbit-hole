import asyncio
from unittest.mock import AsyncMock, patch

import httpx

from research_agent.graph.nodes.researcher import (
    ExtractedFindingItem,
    QueryGenerationOutput,
    ResearchWorker,
    WorkerFindingsOutput,
    researcher_node,
)
from research_agent.graph.state import SubQuestion
from research_agent.tools.fetch import PageFetcher
from research_agent.tools.serper import (
    SerperSearchTool,
)


def test_researcher_generate_queries_mock():
    worker = ResearchWorker()
    sq = SubQuestion(
        id="sq_1",
        question="What is the efficiency of perovskite solar cells?",
        search_intent="Perovskite solar cell certified laboratory efficiency",
        answer_criteria="Efficiency percentages and year certified",
    )

    with patch(
        "research_agent.graph.nodes.researcher.get_chat_model"
    ) as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = QueryGenerationOutput(
            queries=[
                "perovskite solar cell efficiency 2024",
                "certified perovskite lab records",
            ]
        )
        mock_get_model.return_value.with_structured_output.return_value = mock_model

        queries = asyncio.run(worker.generate_queries(sq))
        assert len(queries) == 2
        assert "perovskite" in queries[0]


def test_researcher_full_pipeline(tmp_path):
    sq = SubQuestion(
        id="sq_1",
        question="What are recent fusion milestones?",
        search_intent="Net energy gain fusion records in 2024",
        answer_criteria="Mention energy output in megajoules",
    )

    # Mock Serper
    def mock_serper(request: httpx.Request) -> httpx.Response:
        data = {
            "organic": [
                {
                    "title": "Fusion Milestone 2024",
                    "link": "https://example.com/fusion",
                    "snippet": "Researchers achieved 3.88 MJ yield",
                }
            ]
        }
        return httpx.Response(200, json=data)

    # Mock Fetcher
    def mock_fetch(request: httpx.Request) -> httpx.Response:
        html = """
        <html><body>
        <article>
        <h1>Fusion Milestone</h1>
        <p>In early 2024, scientists achieved a net energy gain of 3.88 megajoules from nuclear fusion.</p>
        </article>
        </body></html>
        """
        return httpx.Response(
            200, headers={"content-type": "text/html; charset=utf-8"}, text=html
        )

    serper_tool = SerperSearchTool(
        api_key="mock", transport=httpx.MockTransport(mock_serper)
    )
    fetch_tool = PageFetcher(transport=httpx.MockTransport(mock_fetch))

    worker = ResearchWorker(search_tool=serper_tool, fetch_tool=fetch_tool)

    # Mock LLM for query generation and findings extraction
    with patch(
        "research_agent.graph.nodes.researcher.get_chat_model"
    ) as mock_get_model:
        mock_structured = AsyncMock()

        # Call 1: queries, Call 2: findings extraction
        mock_structured.ainvoke.side_effect = [
            QueryGenerationOutput(queries=["nuclear fusion 3.88 MJ milestone"]),
            WorkerFindingsOutput(
                findings=[
                    ExtractedFindingItem(
                        claim="Scientists achieved 3.88 megajoules net energy gain from fusion in 2024.",
                        evidence_quote="scientists achieved a net energy gain of 3.88 megajoules from nuclear fusion",
                        source_url="https://example.com/fusion",
                        source_title="Fusion Milestone",
                        confidence=0.98,
                    )
                ]
            ),
        ]
        mock_get_model.return_value.with_structured_output.return_value = (
            mock_structured
        )

        findings = asyncio.run(worker.execute(sq))
        assert len(findings) == 1
        assert findings[0].sub_question_id == "sq_1"
        assert "3.88 megajoules" in findings[0].claim
        assert findings[0].confidence == 0.98
        assert findings[0].source_url == "https://example.com/fusion"


def test_researcher_node_invocation():
    sq = SubQuestion(
        id="sq_2",
        question="What is the speed of light in vacuum?",
        search_intent="Speed of light exact meters per second",
        answer_criteria="Should state 299,792,458 m/s",
    )

    with patch.object(ResearchWorker, "execute", new_callable=AsyncMock) as mock_exec:
        from research_agent.graph.state import Finding

        mock_exec.return_value = [
            Finding(
                sub_question_id="sq_2",
                claim="The speed of light in vacuum is exactly 299,792,458 m/s.",
                evidence_quote="The speed of light in vacuum is exactly 299,792,458 m/s.",
                source_url="https://physics.nist.gov/constants",
                source_title="NIST Physics Constants",
                confidence=1.0,
            )
        ]

        result = asyncio.run(researcher_node({"sub_question": sq}))
        assert "findings" in result
        assert len(result["findings"]) == 1
        assert result["findings"][0].sub_question_id == "sq_2"
