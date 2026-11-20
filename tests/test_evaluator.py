import asyncio
from unittest.mock import AsyncMock, patch

from research_agent.graph.nodes.evaluator import (
    EvaluatorLLMOutput,
    evaluate_report,
    evaluator_node,
)
from research_agent.graph.state import (
    Citation,
    EvalResult,
    EvalScores,
    Finding,
    ResearchState,
    SubQuestion,
)


def test_evaluator_empty_draft():
    state = ResearchState(question="Empty draft test", draft="", plan=[])
    res = asyncio.run(evaluate_report(state))
    assert res.passed is False
    assert res.scores.groundedness == 1.0
    assert "empty" in res.feedback.lower()


def test_evaluator_passes_high_quality_draft():
    sqs = [
        SubQuestion(
            id="sq_1",
            question="What is fusion?",
            search_intent="Fusion definition",
            answer_criteria="Define nuclear fusion",
        )
    ]
    findings = [
        Finding(
            sub_question_id="sq_1",
            claim="Fusion powers stars.",
            evidence_quote="Nuclear fusion is the process that powers active stars.",
            source_url="https://example.com/fusion",
            source_title="Fusion 101",
        )
    ]
    citations = [
        Citation(
            index=1,
            source_url="https://example.com/fusion",
            source_title="Fusion 101",
            retrieved_at="2024-01-01T00:00:00Z",
            verified=True,
        )
    ]
    draft = "Nuclear fusion is the fundamental process that powers stars [1]."
    state = ResearchState(
        question="What is fusion?",
        plan=sqs,
        findings=findings,
        citations=citations,
        draft=draft,
    )

    with patch("research_agent.graph.nodes.evaluator.get_chat_model") as mock_get_model:
        mock_chat = AsyncMock()
        mock_chat.ainvoke.return_value = EvaluatorLLMOutput(
            groundedness=4.8,
            coverage=4.5,
            citation_accuracy=4.9,
            clarity=4.5,
            unsupported_claims=[],
            missing_sub_questions=[],
            feedback="Excellent draft.",
        )
        mock_get_model.return_value.with_structured_output.return_value = mock_chat

        res = asyncio.run(evaluate_report(state))
        assert res.passed is True
        assert res.scores.groundedness == 4.8
        assert len(res.missing_sub_questions) == 0


def test_evaluator_penalizes_unverified_citations():
    # Deterministic verifier flagged an unverified citation
    citations = [
        Citation(
            index=99,
            source_url="",
            source_title="Unknown",
            retrieved_at="",
            verified=False,
            verification_error="Citation [99] does not resolve to any retrieved source.",
        )
    ]
    draft = "Magic energy exists [99]."
    state = ResearchState(
        question="Magic test", plan=[], findings=[], citations=citations, draft=draft
    )

    with patch("research_agent.graph.nodes.evaluator.get_chat_model") as mock_get_model:
        mock_chat = AsyncMock()
        # Even if the LLM mistakenly gave high scores, our deterministic guard penalizes it!
        mock_chat.ainvoke.return_value = EvaluatorLLMOutput(
            groundedness=4.0,
            coverage=4.0,
            citation_accuracy=4.5,
            clarity=4.0,
            unsupported_claims=[],
            missing_sub_questions=[],
            feedback="Looks fine",
        )
        mock_get_model.return_value.with_structured_output.return_value = mock_chat

        res = asyncio.run(evaluate_report(state))
        assert res.passed is False  # Cannot pass with unverified citations!
        assert res.scores.citation_accuracy <= 2.5


def test_evaluator_node_increments_revision_count():
    state = ResearchState(
        question="Revision test",
        draft="Low quality draft",
        revision_count=0,
    )
    with patch(
        "research_agent.graph.nodes.evaluator.evaluate_report", new_callable=AsyncMock
    ) as mock_eval:
        mock_eval.return_value = EvalResult(
            scores=EvalScores(
                groundedness=2.0, coverage=2.0, citation_accuracy=2.0, clarity=2.0
            ),
            unsupported_claims=["Unsupported assertion"],
            missing_sub_questions=["sq_1"],
            passed=False,
            feedback="Needs major fixes",
        )

        update = asyncio.run(evaluator_node(state))
        assert update["eval_result"].passed is False
