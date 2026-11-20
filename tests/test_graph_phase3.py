import asyncio
from unittest.mock import AsyncMock, patch

from research_agent.graph.builder import build_phase3_graph
from research_agent.graph.nodes.planner import PlannerOutput
from research_agent.graph.state import Finding, ResearchState, SubQuestion


def test_phase3_end_to_end_graph():
    """Validates full execution from Planner -> Researchers -> Synthesizer -> Citation Verifier."""
    graph = build_phase3_graph()

    mock_plan = [
        SubQuestion(
            id="sq_1",
            question="What are LFP battery benefits?",
            search_intent="LFP battery safety and cycle life",
            answer_criteria="Cycle life and thermal stability",
        ),
        SubQuestion(
            id="sq_2",
            question="What are the energy density limitations of LFP?",
            search_intent="LFP energy density Wh/kg compared to NMC",
            answer_criteria="Specific energy density figures",
        ),
        SubQuestion(
            id="sq_3",
            question="What is the cost comparison between LFP and NMC?",
            search_intent="LFP vs NMC dollar per kilowatt-hour cost",
            answer_criteria="Cost parity metrics",
        ),
    ]

    async def mock_worker_execute(self, sub_question: SubQuestion):
        return [
            Finding(
                sub_question_id="sq_1",
                claim="LFP batteries offer over 3,000 charge cycles and superior thermal safety.",
                evidence_quote="LFP chemistry provides over 3000 cycles and high thermal stability",
                source_url="https://batteryuniversity.com/lfp",
                source_title="Battery University LFP Guide",
            )
        ]

    mock_draft = (
        "# LFP Battery Analysis\n\n"
        "LFP batteries offer over 3,000 charge cycles and superior thermal safety [1].\n\n"
        "## Sources\n"
        "[1] Battery University LFP Guide - https://batteryuniversity.com/lfp\n"
    )

    with (
        patch("research_agent.graph.nodes.planner.plan_research") as mock_plan_fn,
        patch(
            "research_agent.graph.nodes.researcher.ResearchWorker.execute",
            new=mock_worker_execute,
        ),
        patch(
            "research_agent.graph.nodes.synthesizer.get_chat_model"
        ) as mock_get_model,
    ):
        mock_plan_fn.return_value = PlannerOutput(sub_questions=mock_plan)

        mock_chat = AsyncMock()
        mock_chat.ainvoke.return_value.content = mock_draft
        mock_get_model.return_value = mock_chat

        initial_state = ResearchState(question="Assess LFP battery technology")
        final_state = asyncio.run(graph.ainvoke(initial_state))

        # Assertions
        assert len(final_state["plan"]) == 3
        assert len(final_state["findings"]) >= 1
        assert final_state["draft"] is not None
        assert "[1]" in final_state["draft"]

        # Check citations and verifier output
        assert len(final_state["citations"]) == 1
        assert final_state["citations"][0].verified is True
        assert final_state["citations"][0].index == 1
        assert len(final_state["errors"]) == 0
