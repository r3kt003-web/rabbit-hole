import asyncio
from unittest.mock import AsyncMock, patch

from research_agent.graph.nodes.planner import (
    PlannerOutput,
    plan_research,
    planner_node,
)
from research_agent.graph.state import ResearchState, SubQuestion


def test_planner_output_schema():
    sub_qs = [
        SubQuestion(
            id=f"sq_{i}",
            question=f"Sub-question {i}",
            search_intent=f"Intent {i}",
            answer_criteria=f"Criteria {i}",
        )
        for i in range(1, 4)
    ]
    output = PlannerOutput(sub_questions=sub_qs)
    assert len(output.sub_questions) == 3
    assert output.sub_questions[0].id == "sq_1"


def test_planner_mock_subquestions():
    mock_qs = [
        SubQuestion(id="sq_1", question="Q1", search_intent="I1", answer_criteria="C1"),
        SubQuestion(id="sq_2", question="Q2", search_intent="I2", answer_criteria="C2"),
        SubQuestion(id="sq_3", question="Q3", search_intent="I3", answer_criteria="C3"),
    ]
    res = asyncio.run(plan_research("Test question", mock_subquestions=mock_qs))
    assert len(res.sub_questions) == 3
    assert res.sub_questions[0].question == "Q1"


def test_planner_node_execution():
    mock_qs = [
        SubQuestion(id="sq_1", question="Q1", search_intent="I1", answer_criteria="C1"),
        SubQuestion(id="sq_2", question="Q2", search_intent="I2", answer_criteria="C2"),
        SubQuestion(id="sq_3", question="Q3", search_intent="I3", answer_criteria="C3"),
    ]
    with patch(
        "research_agent.graph.nodes.planner.plan_research", new_callable=AsyncMock
    ) as mock_plan:
        mock_plan.return_value = PlannerOutput(sub_questions=mock_qs)
        state = ResearchState(question="Explain solid-state battery tech")
        update = asyncio.run(planner_node(state))
        assert "plan" in update
        assert len(update["plan"]) == 3
        assert update["plan"][0].id == "sq_1"


def test_planner_graceful_fallback_on_llm_failure():
    # When LLM raises an error, planner should degrade gracefully to 3 sub-questions
    with patch("research_agent.graph.nodes.planner.get_chat_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.side_effect = RuntimeError(
            "Safety block: content blocked by policy"
        )
        mock_get_model.return_value.with_structured_output.return_value = mock_model

        res = asyncio.run(plan_research("Future of Quantum Computing"))
        assert len(res.sub_questions) >= 3
        assert res.sub_questions[0].id == "sq_1"
        assert "Quantum Computing" in res.sub_questions[0].question
