import logging
import re
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from research_agent.config.settings import get_settings
from research_agent.graph.state import Citation, Finding, ResearchState

logger = logging.getLogger(__name__)


class VerificationResult(BaseModel):
    all_verified: bool = Field(
        description="True if all inline citations are valid and verified"
    )
    verified_citations: list[Citation] = Field(
        default_factory=list, description="List of verified citations"
    )
    unverified_citations: list[Citation] = Field(
        default_factory=list, description="List of failed/unverified citations"
    )
    errors: list[str] = Field(
        default_factory=list, description="Descriptions of citation verification errors"
    )


def fuzzy_substring_match(
    quote: str, text: str, threshold: float = 0.7
) -> tuple[bool, float]:
    """Checks if a quote is contained in or closely matches a portion of the source text.

    Returns (matched, similarity_score).
    """
    cleaned_quote = quote.strip().lower()
    cleaned_text = text.strip().lower()

    if not cleaned_quote:
        return True, 1.0

    # Direct exact substring match
    if cleaned_quote in cleaned_text:
        return True, 1.0

    # Sliding window fuzzy match over text blocks of similar length
    q_len = len(cleaned_quote)
    if q_len == 0 or len(cleaned_text) == 0:
        return False, 0.0

    step = max(1, q_len // 4)
    best_ratio = 0.0

    for i in range(0, len(cleaned_text) - q_len + 1, step):
        window = cleaned_text[i : i + q_len]
        ratio = SequenceMatcher(None, cleaned_quote, window).ratio()
        best_ratio = max(best_ratio, ratio)
        if best_ratio >= threshold:
            return True, best_ratio

    # Also test whole-text ratio if text is short
    if len(cleaned_text) <= q_len * 2:
        ratio = SequenceMatcher(None, cleaned_quote, cleaned_text).ratio()
        best_ratio = max(best_ratio, ratio)

    return best_ratio >= threshold, best_ratio


class CitationVerifier:
    """Deterministic citation verifier.

    Verifies that:
    1. Every [n] inline marker in the draft resolves to an actually retrieved source in state.findings.
    2. The evidence quote associated with the citation appears in the source text (using configurable fuzzy matching).
    3. Flags fabricated or unresolved citations in state.
    """

    def __init__(self, fuzzy_threshold: float | None = None):
        settings = get_settings()
        self.fuzzy_threshold = (
            fuzzy_threshold
            if fuzzy_threshold is not None
            else settings.citation_fuzzy_threshold
        )

    def verify(
        self,
        draft: str,
        findings: list[Finding],
        existing_citations: list[Citation] | None = None,
    ) -> VerificationResult:
        if not draft or not draft.strip():
            return VerificationResult(
                all_verified=False,
                errors=["Draft is empty. Cannot verify citations."],
            )

        # Extract all inline citation numbers, e.g. [1], [2], [12]
        # Ignore markdown link references like [text](url)
        # Match [n] where n is digits and not followed by (
        pattern = r"\[(\d+)\](?!\()"
        matches = re.findall(pattern, draft)
        cited_indices = sorted({int(m) for m in matches})

        # Map citation index -> Citation object
        citations_by_index = {c.index: c for c in (existing_citations or [])}

        # Build lookup of retrieved findings by URL and by sub_question
        findings_by_url = {f.source_url: f for f in findings if f.source_url}

        verified_list: list[Citation] = []
        unverified_list: list[Citation] = []
        errors: list[str] = []

        for idx in cited_indices:
            citation = citations_by_index.get(idx)

            if citation is None:
                # Citation number does not resolve to any known source
                err = f"Citation [{idx}] does not resolve to any retrieved source in this run."
                logger.warning(err)
                unverified_c = Citation(
                    index=idx,
                    source_url="",
                    source_title="Unknown / Hallucinated Source",
                    retrieved_at="",
                    verified=False,
                    verification_error=err,
                )
                unverified_list.append(unverified_c)
                errors.append(err)
                continue

            # Check if source URL was actually retrieved
            matching_finding = findings_by_url.get(citation.source_url)
            if not matching_finding:
                err = (
                    f"Citation [{idx}] cites unretrieved URL: '{citation.source_url}'."
                )
                logger.warning(err)
                citation.verified = False
                citation.verification_error = err
                unverified_list.append(citation)
                errors.append(err)
                continue

            # Verify evidence quote against finding or page raw content
            evidence = citation.evidence_quote or matching_finding.evidence_quote
            source_content = (
                matching_finding.raw_content or matching_finding.evidence_quote
            )

            matched, score = fuzzy_substring_match(
                evidence, source_content, threshold=self.fuzzy_threshold
            )

            if not matched:
                err = (
                    f"Citation [{idx}] evidence quote failed fuzzy verification (similarity: {score:.2f} < {self.fuzzy_threshold}). "
                    f"Quote: '{evidence[:80]}...'"
                )
                logger.warning(err)
                citation.verified = False
                citation.verification_error = err
                unverified_list.append(citation)
                errors.append(err)
            else:
                citation.verified = True
                citation.verification_error = None
                verified_list.append(citation)

        all_verified = len(unverified_list) == 0 and len(errors) == 0

        return VerificationResult(
            all_verified=all_verified,
            verified_citations=verified_list,
            unverified_citations=unverified_list,
            errors=errors,
        )


def verify_citations_node(state: ResearchState) -> dict:
    """LangGraph node running the deterministic citation verifier."""
    verifier = CitationVerifier()
    res = verifier.verify(
        draft=state.draft or "",
        findings=state.findings,
        existing_citations=state.citations,
    )

    all_citations = res.verified_citations + res.unverified_citations
    # Sort by index
    all_citations.sort(key=lambda c: c.index)

    update = {"citations": all_citations}
    if not res.all_verified:
        logger.warning(f"Citation verification failed with {len(res.errors)} error(s).")
        update["errors"] = state.errors + res.errors

    return update
