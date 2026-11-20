from research_agent.graph.state import (
    EvalResult,
    EvalScores,
    Finding,
    ResearchState,
    SubQuestion,
    merge_findings,
)


def test_sub_question_creation():
    sq = SubQuestion(
        id="sq_1",
        question="What is the impact of quantum computing on cryptography?",
        search_intent="Find modern post-quantum cryptography standards and timelines",
        answer_criteria="Should mention NIST standards and lattice-based algorithms",
    )
    assert sq.id == "sq_1"
    assert sq.status == "pending"


def test_finding_creation():
    f = Finding(
        sub_question_id="sq_1",
        claim="NIST announced post-quantum encryption standards.",
        evidence_quote="In August 2024, NIST released its first set of finalized post-quantum standards.",
        source_url="https://example.com/nist-pqc",
        source_title="NIST Finalizes Post-Quantum Standards",
        confidence=0.95,
    )
    assert f.sub_question_id == "sq_1"
    assert f.confidence == 0.95
    assert f.retrieved_at is not None


def test_merge_findings_reducer_deduplication():
    f1 = Finding(
        sub_question_id="sq_1",
        claim="Claim A",
        evidence_quote="Quote A",
        source_url="https://example.com/a",
        source_title="Title A",
    )
    f2 = Finding(
        sub_question_id="sq_1",
        claim="claim a",  # duplicate case-insensitively
        evidence_quote="Quote A variation",
        source_url="https://example.com/a",
        source_title="Title A",
    )
    f3 = Finding(
        sub_question_id="sq_2",
        claim="Claim B",
        evidence_quote="Quote B",
        source_url="https://example.com/b",
        source_title="Title B",
    )

    merged = merge_findings([f1], [f2, f3])
    assert len(merged) == 2
    assert merged[0].claim == "Claim A"
    assert merged[1].claim == "Claim B"


def test_research_state_initialization():
    state = ResearchState(question="Explain fusion energy breakthroughs in 2024")
    assert state.question == "Explain fusion energy breakthroughs in 2024"
    assert len(state.plan) == 0
    assert len(state.findings) == 0
    assert state.draft is None
    assert state.revision_count == 0
    assert state.budget_exceeded is False


def test_eval_result_schema():
    scores = EvalScores(
        groundedness=4.5,
        coverage=4.0,
        citation_accuracy=5.0,
        clarity=4.2,
    )
    eval_res = EvalResult(
        scores=scores,
        unsupported_claims=[],
        missing_sub_questions=[],
        passed=True,
        feedback="High quality draft.",
    )
    assert eval_res.passed is True
    assert eval_res.scores.citation_accuracy == 5.0
