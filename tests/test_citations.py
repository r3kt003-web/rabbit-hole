from research_agent.graph.state import Citation, Finding, ResearchState
from research_agent.verification.citations import (
    CitationVerifier,
    fuzzy_substring_match,
    verify_citations_node,
)


def test_fuzzy_substring_match():
    text = "Scientists at the National Ignition Facility achieved net energy gain with 3.88 MJ yield in 2024."

    # Exact match
    matched, score, span = fuzzy_substring_match(
        "achieved net energy gain with 3.88 MJ yield", text
    )
    assert matched is True
    assert score == 1.0
    assert span is not None
    assert "achieved net energy gain with 3.88 mj yield" in span

    # Near match with minor deviation
    matched, score, span = fuzzy_substring_match(
        "achieved net energy gain with 3.88 MJ", text, threshold=0.7
    )
    assert matched is True
    assert score >= 0.7

    # Completely false quote
    matched, score, span = fuzzy_substring_match(
        "Aliens landed on Mars with flying saucers", text, threshold=0.7
    )
    assert matched is False
    assert score < 0.5
    assert span is None


def test_citation_verifier_valid():
    findings = [
        Finding(
            sub_question_id="sq_1",
            claim="Fusion net energy gain achieved 3.88 MJ in 2024.",
            evidence_quote="achieved net energy gain with 3.88 MJ yield in 2024",
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
            claim="Fusion net energy gain achieved 3.88 MJ in 2024.",
            evidence_quote="achieved net energy gain with 3.88 MJ yield in 2024",
        )
    ]
    draft = "Recent experiments in 2024 achieved viable net energy gain in nuclear fusion [1]."

    verifier = CitationVerifier(fuzzy_threshold=0.85)
    res = verifier.verify(draft, findings, citations)

    assert res.all_verified is True
    assert len(res.unverified_citations) == 0
    assert len(res.verified_citations) == 1
    assert res.verified_citations[0].verified is True
    assert res.verified_citations[0].status == "verified"
    assert res.verified_citations[0].matched_span is not None


def test_citation_verifier_fabricated_quote_fails():
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
            # Fabricated quote not in source
            evidence_quote="Solar panels now generate unlimited power at midnight without light.",
        )
    ]
    draft = "Solar panels generate unlimited power at midnight [1]."

    verifier = CitationVerifier(fuzzy_threshold=0.85)
    res = verifier.verify(draft, findings, citations)

    assert res.all_verified is False
    assert len(res.unverified_citations) == 1
    failed_c = res.unverified_citations[0]
    assert failed_c.status == "quote_not_found"
    assert failed_c.verified is False
    assert "evidence quote not found in source text" in failed_c.verification_error


def test_citation_verifier_unsupported_figure_fails():
    # Real quote exists in source, but attached claim contains different/fabricated numbers
    findings = [
        Finding(
            sub_question_id="sq_1",
            claim="Market share was 15% in 2020.",
            evidence_quote="Electric vehicle market share reached 15% globally in 2020.",
            source_url="https://example.com/ev-stats",
            source_title="EV Annual Report",
            raw_content="Electric vehicle market share reached 15% globally in 2020.",
        )
    ]
    citations = [
        Citation(
            index=1,
            source_url="https://example.com/ev-stats",
            source_title="EV Annual Report",
            retrieved_at="2024-01-01T00:00:00Z",
            claim="Electric vehicle market share surged to 95% by 2035 [1].",
            evidence_quote="Electric vehicle market share reached 15% globally in 2020.",
        )
    ]
    # Claim introduces unsupported figures: 95% and 2035
    draft = "Electric vehicle market share surged to 95% by 2035 [1]."

    verifier = CitationVerifier(fuzzy_threshold=0.85)
    res = verifier.verify(draft, findings, citations)

    assert res.all_verified is False
    assert len(res.unverified_citations) == 1
    failed_c = res.unverified_citations[0]
    assert failed_c.status == "unsupported_figure"
    assert failed_c.verified is False
    assert "claim contains unsupported figure(s)" in failed_c.verification_error
    assert "95%" in failed_c.verification_error or "2035" in failed_c.verification_error


def test_citation_verifier_whitespace_and_case_variations_pass():
    # Source has standard text, quote has mixed case, tabs, multiple spaces, smart quotes
    source_text = 'Solid-state batteries feature “lithium-metal anodes” yielding 450 Wh/kg in 2026.'
    findings = [
        Finding(
            sub_question_id="sq_1",
            claim="Solid-state batteries yield 450 Wh/kg in 2026.",
            evidence_quote="solid-state batteries feature lithium-metal anodes yielding 450 wh/kg in 2026",
            source_url="https://example.com/battery",
            source_title="Battery Tech",
            raw_content=source_text,
        )
    ]
    citations = [
        Citation(
            index=1,
            source_url="https://example.com/battery",
            source_title="Battery Tech",
            retrieved_at="2026-01-01T00:00:00Z",
            claim="Solid-state batteries yield 450 Wh/kg in 2026.",
            evidence_quote="SOLID-STATE   BATTERIES feature   \"lithium-metal anodes\"\n  yielding 450 Wh/kg in 2026.",
        )
    ]
    draft = "Solid-state batteries yield 450 Wh/kg in 2026 [1]."

    verifier = CitationVerifier(fuzzy_threshold=0.85)
    res = verifier.verify(draft, findings, citations)

    assert res.all_verified is True
    assert len(res.verified_citations) == 1
    assert res.verified_citations[0].status == "verified"
    assert res.verified_citations[0].verified is True


def test_citation_verifier_phantom_index_fails():
    findings = [
        Finding(
            sub_question_id="sq_1",
            claim="Valid finding",
            evidence_quote="Valid evidence quote",
            source_url="https://example.com/valid",
            source_title="Valid Source",
        )
    ]
    citations = [
        Citation(
            index=1,
            source_url="https://example.com/valid",
            source_title="Valid Source",
            retrieved_at="2026-01-01T00:00:00Z",
            claim="Valid claim",
            evidence_quote="Valid evidence quote",
        )
    ]
    # Draft cites [1] and non-existent [99]
    draft = "Valid point here [1]. Another unreferenced phantom claim [99]."

    verifier = CitationVerifier(fuzzy_threshold=0.85)
    res = verifier.verify(draft, findings, citations)

    assert res.all_verified is False
    assert len(res.unverified_citations) == 1
    phantom = res.unverified_citations[0]
    assert phantom.index == 99
    assert phantom.status == "unresolved_source"
    assert phantom.verified is False
    assert "does not resolve to any retrieved source" in phantom.verification_error


def test_citation_threshold_boundary_behavior():
    source_text = "The quick brown fox jumps over the lazy dog in London."
    findings = [
        Finding(
            sub_question_id="sq_1",
            claim="Fox jumping behavior",
            evidence_quote="The quick brown fox jumps over the lazy dog",
            source_url="https://example.com/fox",
            source_title="Fox Story",
            raw_content=source_text,
        )
    ]
    # Quote with slight difference: score is ~0.93
    quote_near = "The quick brown fox leaps over the lazy dog"
    # Quote with moderate deviation: score is exactly 0.75
    quote_mid = "The quick red fox hops over a lazy hound"

    c1 = Citation(
        index=1,
        source_url="https://example.com/fox",
        source_title="Fox Story",
        retrieved_at="2026-01-01T00:00:00Z",
        evidence_quote=quote_near,
    )
    c2 = Citation(
        index=2,
        source_url="https://example.com/fox",
        source_title="Fox Story",
        retrieved_at="2026-01-01T00:00:00Z",
        evidence_quote=quote_mid,
    )

    verifier_strict = CitationVerifier(fuzzy_threshold=0.85)
    # quote_near passes 0.85
    r1 = verifier_strict.verify("Fox leap [1].", findings, [c1])
    assert r1.all_verified is True
    assert r1.verified_citations[0].status == "verified"

    # quote_mid fails at strict 0.85
    r2 = verifier_strict.verify("Fox jump [2].", findings, [c2])
    assert r2.all_verified is False
    assert r2.unverified_citations[0].status == "quote_not_found"

    # With relaxed threshold (0.70), quote_mid passes
    verifier_relaxed = CitationVerifier(fuzzy_threshold=0.70)
    r3 = verifier_relaxed.verify("Fox jump [2].", findings, [c2])
    assert r3.all_verified is True
    assert r3.verified_citations[0].status == "verified"


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
    assert update["citations"][0].status == "unresolved_source"
    assert len(update["errors"]) > 0
    assert "Citation [42] does not resolve" in update["errors"][0]
