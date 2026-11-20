import asyncio
from unittest.mock import patch

from research_agent.graph.builder import build_phase2_graph
from research_agent.graph.state import Finding, ResearchState, SubQuestion


def test_parallel_fanout_merges_findings_correctly():
    """Validates that LangGraph Send API parallel fan-out fans out across sub-questions
    and that the merge_findings reducer properly aggregates findings without duplicates.
    """
    graph = build_phase2_graph()

    # Plan with 3 sub-questions
    mock_plan = [
        SubQuestion(
            id="sq_1",
            question="Topic 1",
            search_intent="Intent 1",
            answer_criteria="Crit 1",
        ),
        SubQuestion(
            id="sq_2",
            question="Topic 2",
            search_intent="Intent 2",
            answer_criteria="Crit 2",
        ),
        SubQuestion(
            id="sq_3",
            question="Topic 3",
            search_intent="Intent 3",
            answer_criteria="Crit 3",
        ),
    ]

    async def mock_worker_execute(self, sub_question: SubQuestion):
        # Return unique findings per sub-question, plus one intentionally duplicated finding
        findings = [
            Finding(
                sub_question_id=sub_question.id,
                claim=f"Claim for {sub_question.id}",
                evidence_quote=f"Quote for {sub_question.id}",
                source_url=f"https://example.com/{sub_question.id}",
                source_title=f"Title for {sub_question.id}",
            )
        ]
        # Simulate duplicate finding on sq_1
        if sub_question.id == "sq_1":
            findings.append(
                Finding(
                    sub_question_id=sub_question.id,
                    claim=f"Claim for {sub_question.id}",  # Duplicate claim
                    evidence_quote="Alternate quote",
                    source_url=f"https://example.com/{sub_question.id}",
                    source_title=f"Title for {sub_question.id}",
                )
            )
        return findings

    with (
        patch("research_agent.graph.nodes.planner.plan_research") as mock_plan_fn,
        patch(
            "research_agent.graph.nodes.researcher.ResearchWorker.execute",
            new=mock_worker_execute,
        ),
    ):
        from research_agent.graph.nodes.planner import PlannerOutput

        mock_plan_fn.return_value = PlannerOutput(sub_questions=mock_plan)

        initial_state = ResearchState(
            question="Assess the state of renewable energy in 2024"
        )
        final_state = asyncio.run(graph.ainvoke(initial_state))

        # Check that plan was populated
        assert len(final_state["plan"]) == 3

        # Check that findings from all 3 workers were merged
        findings = final_state["findings"]
        assert len(findings) == 3  # The duplicate was dropped by the reducer!

        sub_ids = {f.sub_question_id for f in findings}
        assert sub_ids == {"sq_1", "sq_2", "sq_3"}
