from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, Field


def merge_findings(
    existing: list[Finding], new_findings: list[Finding]
) -> list[Finding]:
    """Reducer that merges findings and drops exact duplicates based on sub_question_id, claim and source_url."""
    seen = {
        (f.sub_question_id, f.claim.strip().lower(), f.source_url) for f in existing
    }
    result = list(existing)
    for f in new_findings:
        key = (f.sub_question_id, f.claim.strip().lower(), f.source_url)
        if key not in seen:
            seen.add(key)
            result.append(f)
    return result


class SubQuestion(BaseModel):
    id: str = Field(description="Unique identifier for the sub-question, e.g. 'sq_1'")
    question: str = Field(description="The specific sub-question to answer")
    search_intent: str = Field(
        description="What kind of information needs to be searched for"
    )
    answer_criteria: str = Field(
        description="Criteria for what constitutes a complete and factual answer"
    )
    status: str = Field(
        default="pending", description="Status: pending, completed, or failed"
    )


class Finding(BaseModel):
    sub_question_id: str = Field(
        description="The id of the sub-question this finding addresses"
    )
    claim: str = Field(description="The factual claim extracted from evidence")
    evidence_quote: str = Field(
        description="Exact or near-exact quote from the source supporting the claim"
    )
    source_url: str = Field(description="URL of the source")
    source_title: str = Field(description="Title of the source webpage")
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO timestamp when the source was retrieved",
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0"
    )
    raw_content: str | None = Field(
        default=None,
        description="Optional raw text content of the page for verification",
    )


class Citation(BaseModel):
    index: int = Field(description="Numbered citation marker, e.g. 1 for [1]")
    source_url: str = Field(description="URL of the cited source")
    source_title: str = Field(description="Title of the cited source")
    retrieved_at: str = Field(description="ISO timestamp when retrieved")
    claim: str | None = Field(
        default=None, description="The claim associated with this citation"
    )
    evidence_quote: str | None = Field(
        default=None, description="Quoted evidence backing this citation"
    )
    verified: bool = Field(
        default=True, description="Whether citation verification passed"
    )
    status: str = Field(
        default="verified",
        description="Verification status: verified, quote_not_found, unsupported_figure, unresolved_source",
    )
    matched_span: str | None = Field(
        default=None, description="The matched text span from the source"
    )
    verification_error: str | None = Field(
        default=None, description="Error reason if verification failed"
    )


class EvalScores(BaseModel):
    groundedness: float = Field(
        ge=1.0, le=5.0, description="Are claims supported by cited evidence (1-5)"
    )
    coverage: float = Field(
        ge=1.0, le=5.0, description="Does it answer every sub-question (1-5)"
    )
    citation_accuracy: float = Field(
        ge=1.0, le=5.0, description="Are inline citations accurate and resolvable (1-5)"
    )
    clarity: float = Field(
        ge=1.0, le=5.0, description="Clarity, structure and readability (1-5)"
    )


class EvalResult(BaseModel):
    scores: EvalScores = Field(description="Category scores 1-5")
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims that lack cited evidence"
    )
    missing_sub_questions: list[str] = Field(
        default_factory=list,
        description="Sub-questions that were missed or insufficiently addressed",
    )
    passed: bool = Field(
        default=False, description="Whether the report passed evaluation threshold"
    )
    feedback: str = Field(
        default="", description="Detailed qualitative feedback for revision"
    )


class ResearchState(BaseModel):
    question: str = Field(description="The primary research question")
    plan: list[SubQuestion] = Field(
        default_factory=list, description="Planned list of sub-questions"
    )
    findings: Annotated[list[Finding], merge_findings] = Field(
        default_factory=list, description="Extracted findings"
    )
    draft: str | None = Field(default=None, description="Current report draft")
    citations: list[Citation] = Field(
        default_factory=list, description="List of verified citations"
    )
    eval_result: EvalResult | None = Field(
        default=None, description="Evaluation score and feedback"
    )
    revision_count: int = Field(
        default=0, description="Number of revisions attempted so far"
    )
    trace_id: str = Field(default="", description="Unique run/trace ID")
    budget_exceeded: bool = Field(
        default=False, description="Flag indicating if cost budget was exceeded"
    )
    errors: list[str] = Field(
        default_factory=list, description="List of recorded runtime errors"
    )
