import asyncio
from unittest.mock import AsyncMock, patch

from research_agent.graph.nodes.synthesizer import (
    build_source_manifest,
    synthesize_report,
    synthesizer_node,
)
from research_agent.graph.state import Finding, ResearchState


def test_build_source_manifest():
    findings = [
        Finding(
            sub_question_id="sq_1",
            claim="Claim 1",
            evidence_quote="Quote 1",
            source_url="https://example.com/source1",
            source_title="Source 1",
        ),
        Finding(
            sub_question_id="sq_2",
            claim="Claim 2",
            evidence_quote="Quote 2",
            source_url="https://example.com/source2",
            source_title="Source 2",
        ),
        Finding(
            sub_question_id="sq_1",
            claim="Claim 1 duplicate",
            evidence_quote="Quote 1 variant",
            source_url="https://example.com/source1",  # Same source URL
            source_title="Source 1",
        ),
    ]

    manifest, citations = build_source_manifest(findings)
    assert len(citations) == 2
    assert citations[0].index == 1
    assert citations[0].source_url == "https://example.com/source1"
    assert citations[1].index == 2
    assert citations[1].source_url == "https://example.com/source2"
    assert "[1] Title: Source 1" in manifest
    assert "[2] Title: Source 2" in manifest


def test_synthesize_empty_findings():
    draft, citations = asyncio.run(synthesize_report("What is quantum computing?", []))
    assert "No verifiable evidence was found" in draft
    assert len(citations) == 0


def test_synthesizer_node():
    findings = [
        Finding(
            sub_question_id="sq_1",
            claim="Perovskite tandem cells reached 33.9% efficiency.",
            evidence_quote="tandem solar cells demonstrated 33.9% certified efficiency",
            source_url="https://nrel.gov/pv",
            source_title="NREL Chart",
        )
    ]
    state = ResearchState(
        question="What is the latest solar efficiency record?", findings=findings
    )

    mock_report = (
        "# Solar Cell Efficiency Report\n\n"
        "Perovskite tandem cells have reached an unprecedented certified efficiency of 33.9% [1].\n\n"
        "## Sources\n"
        "[1] NREL Chart - https://nrel.gov/pv\n"
    )

    with patch(
        "research_agent.graph.nodes.synthesizer.get_chat_model"
    ) as mock_get_model:
        mock_chat = AsyncMock()
        mock_chat.ainvoke.return_value.content = mock_report
        mock_get_model.return_value = mock_chat

        update = asyncio.run(synthesizer_node(state))
        assert "draft" in update
        assert "[1]" in update["draft"]
        assert len(update["citations"]) == 1
        assert update["citations"][0].source_url == "https://nrel.gov/pv"
