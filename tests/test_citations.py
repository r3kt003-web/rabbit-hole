from research_agent.graph.state import Citation, Finding, ResearchState
from research_agent.verification.citations import (
    CitationVerifier,
    fuzzy_substring_match,
    verify_citations_node,
)


def test_fuzzy_substring_match():
    text = "Scientists at the National Ignition Facility achieved net energy gain with 3.88 MJ yield in 2024."

    # Exact match
    matched, score = fuzzy_substring_match(
        "achieved net energy gain with 3.88 MJ yield", text
    )
    assert matched is True
    assert score == 1.0

    # Near match with minor deviation
    matched, score = fuzzy_substring_match(
        "achieved net energy gain with 3.88 MJ", text, threshold=0.7
    )
    assert matched is True
    assert score >= 0.7

    # Completely false quote
    matched, score = fuzzy_substring_match(
        "Aliens landed on Mars with flying saucers", text, threshold=0.7
    )
    assert matched is False
    assert score < 0.5


def test_citation_verifier_valid():
    findings = [
        Finding(
            sub_question_id="sq_1",
            claim="Fusion net energy gain achieved 3.88 MJ.",
            evidence_quote="achieved net energy gain with 3.88 MJ yield",
            source_url="https://llnl.gov/nif",
            source_title="NIF Announcement",
        )
    ]
    citations = [
        Citation(
            index=1,
            source_url="https://llnl.gov/nif",
            source_title="NIF Announcement",
            retrieved_at="2024-01-01T00:00:00Z",
            claim="Fusion net energy gain achieved 3.88 MJ.",
            evidence_quote="achieved net energy gain with 3.88 MJ yield",
        )
    ]
    draft = "Recent experiments showed viable net energy gain in nuclear fusion [1]."

    verifier = CitationVerifier()
    res = verifier.verify(draft, findings, citations)

    assert res.all_verified is True
    assert len(res.unverified_citations) == 0
    assert len(res.verified_citations) == 1
    assert res.verified_citations[0].verified is True


def test_citation_verifier_catches_unretrieved_source():
    # Draft cites [1] and [99], but [99] does not exist in citations or findings!
    findings = [
        Finding(
            sub_question_id="sq_1",
            claim="Quantum supremacy demonstrated.",
            evidence_quote="Quantum computer performs calculation in 200 seconds",
            source_url="https://nature.com/articles/quantum",
            source_title="Nature Quantum",
        )
    ]
    citations = [
        Citation(
            index=1,
            source_url="https://nature.com/articles/quantum",
            source_title="Nature Quantum",
            retrieved_at="2024-01-01T00:00:00Z",
            evidence_quote="Quantum computer performs calculation in 200 seconds",
        )
    ]
    draft = "Quantum supremacy was proven [1], and faster-than-light travel was developed [99]."

    verifier = CitationVerifier()
    res = verifier.verify(draft, findings, citations)

    assert res.all_verified is False
    assert len(res.unverified_citations) == 1
    assert res.unverified_citations[0].index == 99
    assert "does not resolve" in res.unverified_citations[0].verification_error


def test_citation_verifier_catches_fabricated_quote():
    # Source was retrieved, but cited quote does NOT exist in the source content
    findings = [
        Finding(
            sub_question_id="sq_1",
            claim="Solar panel efficiency data.",
            evidence_quote="Silicon solar panels reach 24% typical commercial efficiency.",
            source_url="https://example.com/solar",
            source_title="Solar Study",
        )
    ]
    citations = [
        Citation(
            index=1,
            source_url="https://example.com/solar",
            source_title="Solar Study",
            retrieved_at="2024-01-01T00:00:00Z",
            # Fabricated quote that cannot be found in the evidence quote
            evidence_quote="Solar panels now generate unlimited power at midnight without light.",
        )
    ]
    draft = "Solar panels generate unlimited power at midnight [1]."

    verifier = CitationVerifier(fuzzy_threshold=0.7)
    res = verifier.verify(draft, findings, citations)

    assert res.all_verified is False
    assert len(res.unverified_citations) == 1
    assert "failed fuzzy verification" in res.unverified_citations[0].verification_error


def test_verify_citations_node_flags_failures_in_state():
    state = ResearchState(
        question="Check fake citation handling",
        draft="This statement is unsupported [42].",
        findings=[],
        citations=[],
    )

    update = verify_citations_node(state)
    assert len(update["citations"]) == 1
    assert update["citations"][0].verified is False
    assert len(update["errors"]) > 0
    assert "Citation [42] does not resolve" in update["errors"][0]
