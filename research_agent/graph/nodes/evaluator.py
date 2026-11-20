import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_agent.config.settings import get_settings
from research_agent.graph.state import EvalResult, EvalScores, ResearchState
from research_agent.llm.client import get_chat_model
from research_agent.llm.rate_limiter import call_with_rate_limit

logger = logging.getLogger(__name__)

EVALUATOR_SYSTEM_PROMPT = """You are an independent, highly critical research evaluator.
Your role is to rigorously evaluate a drafted research report against the research question, planned sub-questions, and verified findings.
You must not be lenient. High scores (4.0 - 5.0) require demonstrable factual grounding and thorough coverage.

Scoring Criteria (1.0 to 5.0):
1. groundedness: Are all claims supported by evidence quotes from retrieved sources? Deduct heavily for unverified assertions.
2. coverage: Does the report adequately and accurately answer every planned sub-question?
3. citation_accuracy: Are inline citations [n] used properly for every claim, and does every [n] reference a legitimate source?
4. clarity: Is the report well-structured, clear, objective, and readable?

Required Output:
- scores: groundedness, coverage, citation_accuracy, clarity (each 1.0 to 5.0)
- unsupported_claims: List of specific claims in the draft that lack evidence or citations.
- missing_sub_questions: List of sub-question IDs (e.g. ['sq_2']) that were not answered or need more research.
- passed: Boolean indicating whether the report meets the required quality bar.
- feedback: Actionable instructions for revising the report."""


class EvaluatorLLMOutput(BaseModel):
    groundedness: float = Field(
        ge=1.0, le=5.0, description="Score 1.0-5.0 on evidence grounding"
    )
    coverage: float = Field(
        ge=1.0, le=5.0, description="Score 1.0-5.0 on answering all sub-questions"
    )
    citation_accuracy: float = Field(
        ge=1.0, le=5.0, description="Score 1.0-5.0 on citation correctness"
    )
    clarity: float = Field(
        ge=1.0, le=5.0, description="Score 1.0-5.0 on clarity and flow"
    )
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims lacking cited evidence"
    )
    missing_sub_questions: list[str] = Field(
        default_factory=list, description="Sub-question IDs needing more research"
    )
    feedback: str = Field(default="", description="Detailed qualitative feedback")


async def evaluate_report(
    state: ResearchState,
    model_name: str | None = None,
    mock_eval: EvalResult | None = None,
) -> EvalResult:
    """Evaluates the report draft against sub-questions and evidence."""
    if mock_eval is not None:
        return mock_eval

    settings = get_settings()
    pass_threshold = settings.pass_threshold

    if not state.draft or not state.draft.strip():
        return EvalResult(
            scores=EvalScores(
                groundedness=1.0, coverage=1.0, citation_accuracy=1.0, clarity=1.0
            ),
            unsupported_claims=["Draft is completely empty."],
            missing_sub_questions=[sq.id for sq in state.plan],
            passed=False,
            feedback="The draft is empty. Research and synthesis must be run.",
        )

    # Check deterministic citation verifier errors
    has_unverified_citations = any(not c.verified for c in state.citations)
    citation_error_notes = [
        c.verification_error
        for c in state.citations
        if not c.verified and c.verification_error
    ]

    model = model_name or settings.evaluator_model
    chat_model = get_chat_model(model, temperature=0.0)
    structured_model = chat_model.with_structured_output(EvaluatorLLMOutput)

    # Prepare sub-questions summary
    plan_summary = "\n".join(
        f"- [{sq.id}] {sq.question} (Criteria: {sq.answer_criteria})"
        for sq in state.plan
    )

    # Prepare evidence summary
    findings_summary = "\n".join(
        f'- Source [{idx + 1}]: {f.source_title} ({f.source_url})\n  Claim: {f.claim}\n  Quote: "{f.evidence_quote}"'
        for idx, f in enumerate(state.findings[:15])
    )

    verification_note = ""
    if has_unverified_citations:
        verification_note = (
            f"\nNOTE: Deterministic verification identified {len(citation_error_notes)} citation issue(s):\n"
            + "\n".join(f"- {err}" for err in citation_error_notes)
            + "\nPlease penalize citation_accuracy accordingly.\n"
        )

    prompt = (
        f"Primary Research Question: {state.question}\n\n"
        f"Planned Sub-Questions:\n{plan_summary}\n\n"
        f"Retrieved Findings & Evidence:\n{findings_summary}\n"
        f"{verification_note}\n"
        f"Draft Report to Evaluate:\n===================\n{state.draft}\n==================="
    )

    messages = [
        SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        raw_res = await call_with_rate_limit(structured_model.ainvoke, messages)
        if isinstance(raw_res, EvaluatorLLMOutput):
            output = raw_res
        elif isinstance(raw_res, dict):
            output = EvaluatorLLMOutput(**raw_res)
        else:
            raise TypeError(f"Unexpected evaluator output type: {type(raw_res)}")

        # Penalize citation accuracy if deterministic check failed
        citation_score = output.citation_accuracy
        if has_unverified_citations and citation_score > 3.0:
            citation_score = min(citation_score, 2.5)

        avg_score = (
            output.groundedness + output.coverage + citation_score + output.clarity
        ) / 4.0

        # Pass condition: average score >= threshold, no missing sub-questions, and no citation failures
        passed = (
            avg_score >= pass_threshold
            and len(output.missing_sub_questions) == 0
            and not has_unverified_citations
        )

        return EvalResult(
            scores=EvalScores(
                groundedness=output.groundedness,
                coverage=output.coverage,
                citation_accuracy=citation_score,
                clarity=output.clarity,
            ),
            unsupported_claims=output.unsupported_claims,
            missing_sub_questions=output.missing_sub_questions,
            passed=passed,
            feedback=output.feedback,
        )

    except (RuntimeError, ValueError, TypeError) as exc:
        logger.error(f"Evaluator structured call failed: {exc}")
        # Deterministic fallback evaluation
        avg_score = 3.0 if not has_unverified_citations else 2.0
        return EvalResult(
            scores=EvalScores(
                groundedness=3.0,
                coverage=3.0,
                citation_accuracy=2.0 if has_unverified_citations else 3.5,
                clarity=3.0,
            ),
            unsupported_claims=[],
            missing_sub_questions=[],
            passed=(avg_score >= pass_threshold and not has_unverified_citations),
            feedback=f"Fallback evaluation triggered due to: {exc}",
        )


async def evaluator_node(state: ResearchState) -> dict[str, Any]:
    """LangGraph node for report evaluation."""
    eval_result = await evaluate_report(state)
    logger.info(
        f"Evaluation finished: passed={eval_result.passed}, "
        f"avg_score={(eval_result.scores.groundedness + eval_result.scores.coverage + eval_result.scores.citation_accuracy + eval_result.scores.clarity) / 4:.2f}, "
        f"current_revision={state.revision_count}"
    )

    return {"eval_result": eval_result}
