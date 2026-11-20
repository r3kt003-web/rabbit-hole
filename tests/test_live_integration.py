import os

import pytest

from research_agent.service.runner import ResearchOptions, run_research


@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_TESTS")
    or not os.getenv("GOOGLE_API_KEY")
    or not os.getenv("SERPER_API_KEY"),
    reason="Optional live integration test skipped unless RUN_LIVE_TESTS=1 and API keys are present",
)
@pytest.mark.asyncio
async def test_live_agent_run():
    """Optional live integration test exercising real Gemini and Serper calls."""
    question = "What is the capital of France and what is its population?"
    options = ResearchOptions(max_revisions=1, max_cost_usd=0.50)

    events = []
    async for event in run_research(question=question, options=options):
        events.append(event)
        if event.event_type in ("run_completed", "run_failed"):
            break

    assert len(events) >= 3
    event_types = {e.event_type for e in events}
    assert "run_started" in event_types
