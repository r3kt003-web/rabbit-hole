import asyncio
from unittest.mock import patch

from research_agent.graph.builder import build_graph, route_after_evaluator
from research_agent.graph.nodes.planner import PlannerOutput
from research_agent.graph.state import (
    Citation,
    EvalResult,
    EvalScores,
    Finding,
    ResearchState,
    SubQuestion,
)


def test_route_after_evaluator_stops_on_pass():
    state = ResearchState(
        question="Pass test",
        revision_count=0,
        eval_result=EvalResult(
            scores=EvalScores(
                groundedness=4.5, coverage=4.5, citation_accuracy=4.5, clarity=4.5
            ),
            passed=True,
        ),
    )
    res = route_after_evaluator(state)
    assert res == "__end__"


def test_route_after_evaluator_stops_at_max_revisions():
    # Even if eval failed, when revision_count >= MAX_REVISIONS (default 2), it must stop at __end__!
    state = ResearchState(
        question="Max revisions test",
        revision_count=2,
        eval_result=EvalResult(
            scores=EvalScores(
                groundedness=2.0, coverage=2.0, citation_accuracy=2.0, clarity=2.0
            ),
            passed=False,
            feedback="Still bad",
        ),
    )
    res = route_after_evaluator(state)
    assert res == "__end__"


def test_route_after_evaluator_missing_sub_questions():
    # If missing sub-questions exist, routes back to research_worker for only those missing sub-questions
    sq1 = SubQuestion(
        id="sq_1", question="Q1", search_intent="I1", answer_criteria="C1"
    )
    sq2 = SubQuestion(
        id="sq_2", question="Q2", search_intent="I2", answer_criteria="C2"
    )
    state = ResearchState(
        question="Missing SQ test",
        plan=[sq1, sq2],
        revision_count=0,
        eval_result=EvalResult(
            scores=EvalScores(
                groundedness=3.0, coverage=2.0, citation_accuracy=3.0, clarity=3.0
            ),
            missing_sub_questions=["sq_2"],  # Only sq_2 was missed
            passed=False,
        ),
    )
    res = route_after_evaluator(state)
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0].node == "research_worker"
    assert res[0].arg["sub_question"]["id"] == "sq_2"


def test_route_after_evaluator_synthesis_issues_routes_to_synthesizer():
    # If no missing sub-questions, routes directly to synthesizer
    state = ResearchState(
        question="Synthesis issue test",
        plan=[
            SubQuestion(
                id="sq_1", question="Q1", search_intent="I1", answer_criteria="C1"
            )
        ],
        revision_count=0,
        eval_result=EvalResult(
            scores=EvalScores(
                groundedness=2.5, coverage=4.0, citation_accuracy=2.5, clarity=3.0
            ),
            unsupported_claims=["Claim X is not supported"],
            missing_sub_questions=[],
            passed=False,
        ),
    )
    res = route_after_evaluator(state)
    assert res == "synthesizer"


def test_full_graph_revision_loop_stops_at_max_revisions():
    """REQUIRED TEST: Validates that an evaluator failure triggers a revision loop
    that stops strictly at MAX_REVISIONS (default 2).
    """
    graph = build_graph()

    mock_plan = [
        SubQuestion(
            id="sq_1", question="Topic 1", search_intent="I1", answer_criteria="C1"
        ),
        SubQuestion(
            id="sq_2", question="Topic 2", search_intent="I2", answer_criteria="C2"
        ),
        SubQuestion(
            id="sq_3", question="Topic 3", search_intent="I3", answer_criteria="C3"
        ),
    ]

    async def mock_worker_exec(self, sub_question: SubQuestion):
        return [
            Finding(
                sub_question_id=sub_question.id,
                claim=f"Claim {sub_question.id}",
                evidence_quote=f"Quote {sub_question.id}",
                source_url=f"https://example.com/{sub_question.id}",
                source_title=f"Source {sub_question.id}",
            )
        ]

    eval_calls = 0

    async def mock_evaluate(state: ResearchState, **kwargs):
        nonlocal eval_calls
        eval_calls += 1
        # Always fail evaluation to test the loop termination
        return EvalResult(
            scores=EvalScores(
                groundedness=2.0, coverage=2.0, citation_accuracy=2.0, clarity=2.0
            ),
            unsupported_claims=["Always failing claim"],
            missing_sub_questions=[],  # routes to synthesizer
            passed=False,
            feedback=f"Failing on evaluation call #{eval_calls}",
        )

    with (
        patch("research_agent.graph.nodes.planner.plan_research") as mock_plan_fn,
        patch(
            "research_agent.graph.nodes.researcher.ResearchWorker.execute",
            new=mock_worker_exec,
        ),
        patch(
            "research_agent.graph.nodes.synthesizer.synthesize_report"
        ) as mock_synth_fn,
        patch(
            "research_agent.graph.nodes.evaluator.evaluate_report", new=mock_evaluate
        ),
    ):
        mock_plan_fn.return_value = PlannerOutput(sub_questions=mock_plan)

        # Mock synthesizer returning a draft with citation [1]
        mock_citations = [
            Citation(
                index=1,
                source_url="https://example.com/sq_1",
                source_title="Source sq_1",
                retrieved_at="2024-01-01T00:00:00Z",
                evidence_quote="Quote sq_1",
            )
        ]
        mock_synth_fn.return_value = (
            "This is a drafted report [1].",
            mock_citations,
        )

        initial_state = ResearchState(question="Test revision loop limit")
        final_state = asyncio.run(graph.ainvoke(initial_state))

        # Check revision count and termination
        assert final_state["revision_count"] == 2  # Hit MAX_REVISIONS (2)
        # 1 initial evaluation + 2 revision evaluations = 3 evaluation calls
        assert eval_calls == 3
        assert final_state["eval_result"] is not None
        assert final_state["eval_result"].passed is False
