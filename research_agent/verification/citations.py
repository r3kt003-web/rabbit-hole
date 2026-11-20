import logging
import re
import unicodedata

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

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


def normalize_text(text: str) -> str:
    """Normalizes text for robust fuzzy comparison:
    - Normalizes unicode (NFKC)
    - Replaces smart quotes / typographic punctuation with ASCII equivalents
    - Lowercases
    - Collapses multiple whitespace and newlines into a single space
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # Smart quotes & dashes
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    text = text.replace("—", "-").replace("–", "-")
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def extract_claim_figures(text: str) -> list[str]:
    """Extracts numbers, percentages, years, currency, and quantities from text,
    ignoring citation markers like [1], [2].
    """
    # Remove citation markers like [1], [23]
    cleaned = re.sub(r"\[\d+\]", "", text)
    # Match currencies ($100, $2.5M), percentages (85%, 12.5%), years (1900-2099), decimals/integers (350, 1,000, 3.88)
    pattern = r"(?:\$[\d,]+(?:\.\d+)?|\b\d+(?:,\d{3})*(?:\.\d+)?%?|\b(?:19|20)\d{2}\b)"
    tokens = re.findall(pattern, cleaned)
    figures: list[str] = []
    for t in tokens:
        val = t.strip().lower()
        if val and val not in figures:
            figures.append(val)
    return figures


def check_unsupported_figures(claim: str, source_text: str) -> list[str]:
    """Entailment guard (deterministic, no LLM):
    Finds any numbers, percentages, units, or years in the claim that do NOT
    appear anywhere in the cited quote or source text.
    """
    claim_figures = extract_claim_figures(claim)
    if not claim_figures:
        return []

    norm_source = normalize_text(source_text).replace(",", "")
    unsupported = []
    for fig in claim_figures:
        norm_fig = fig.replace(",", "")
        # Extract digits-only to check if the numeric value exists in source
        digits = re.sub(r"[^\d.]", "", norm_fig)
        if norm_fig in norm_source:
            continue
        if digits and digits in norm_source:
            continue
        unsupported.append(fig)
    return unsupported


def extract_claim_for_citation(draft: str, index: int) -> str:
    """Extracts the sentence or statement containing the [index] citation marker."""
    sentences = re.split(r"(?<=[.!?\n])\s+", draft)
    marker = f"[{index}]"
    matching = [s.strip() for s in sentences if marker in s]
    return " ".join(matching) if matching else ""


def fuzzy_substring_match(
    quote: str, text: str, threshold: float = 0.85
) -> tuple[bool, float, str | None]:
    """Checks if a quote is contained in or closely matches a portion of the source text.

    Uses text normalization and rapidfuzz.fuzz.partial_ratio_alignment.
    Returns (matched, similarity_score_0_to_1, matched_span).
    """
    norm_quote = normalize_text(quote)
    norm_text = normalize_text(text)

    if not norm_quote:
        return True, 1.0, ""

    if not norm_text:
        return False, 0.0, None

    # Direct exact substring match
    if norm_quote in norm_text:
        start_idx = norm_text.find(norm_quote)
        end_idx = start_idx + len(norm_quote)
        span = norm_text[start_idx:end_idx]
        return True, 1.0, span

    # Rapidfuzz partial ratio alignment (avoids whole-page SequenceMatcher degradation)
    alignment = fuzz.partial_ratio_alignment(norm_quote, norm_text)
    if alignment is None:
        return False, 0.0, None

    score_0_to_1 = alignment.score / 100.0
    matched = score_0_to_1 >= threshold
    matched_span = (
        norm_text[alignment.dest_start : alignment.dest_end] if matched else None
    )
    return matched, score_0_to_1, matched_span


class CitationVerifier:
    """Deterministic citation verifier.

    Verifies that:
    1. Every [n] inline marker in the draft resolves to an actually retrieved source URL.
    2. The evidence quote associated with the citation appears in the source text
       (using normalized rapidfuzz windowed partial ratio matching with configurable threshold).
    3. Entailment Guard: every number, percentage, unit, and year in the cited claim appears
       in the source text or quote; flags mismatches as 'unsupported_figure'.
    4. Categorizes each citation status into:
       {verified, quote_not_found, unsupported_figure, unresolved_source}
       along with the extracted matched text span.
    """

    def __init__(self, fuzzy_threshold: float | None = None):
        settings = get_settings()
        self.fuzzy_threshold = (
            fuzzy_threshold
            if fuzzy_threshold is not None
            else settings.verify_quote_threshold
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

        pattern = r"\[(\d+)\](?!\()"
        matches = re.findall(pattern, draft)
        cited_indices = sorted({int(m) for m in matches})

        citations_by_index = {c.index: c for c in (existing_citations or [])}
        findings_by_url = {f.source_url: f for f in findings if f.source_url}

        verified_list: list[Citation] = []
        unverified_list: list[Citation] = []
        errors: list[str] = []

        for idx in cited_indices:
            citation = citations_by_index.get(idx)
            extracted_claim = extract_claim_for_citation(draft, idx)

            if citation is None:
                err = f"Citation [{idx}] does not resolve to any retrieved source in this run."
                logger.warning(err)
                unverified_c = Citation(
                    index=idx,
                    source_url="",
                    source_title="Unknown / Hallucinated Source",
                    retrieved_at="",
                    verified=False,
                    status="unresolved_source",
                    matched_span=None,
                    verification_error=err,
                )
                unverified_list.append(unverified_c)
                errors.append(err)
                continue

            matching_finding = findings_by_url.get(citation.source_url)
            if not matching_finding:
                err = (
                    f"Citation [{idx}] cites unretrieved URL: '{citation.source_url}'."
                )
                logger.warning(err)
                citation.verified = False
                citation.status = "unresolved_source"
                citation.matched_span = None
                citation.verification_error = err
                unverified_list.append(citation)
                errors.append(err)
                continue

            evidence = citation.evidence_quote or matching_finding.evidence_quote or ""
            source_content = (
                matching_finding.raw_content
                or matching_finding.evidence_quote
                or ""
            )

            matched, score, span = fuzzy_substring_match(
                evidence, source_content, threshold=self.fuzzy_threshold
            )

            if not matched:
                err = (
                    f"Citation [{idx}] evidence quote not found in source text (similarity: {score:.2f} < {self.fuzzy_threshold}). "
                    f"Quote: '{evidence[:80]}...'"
                )
                logger.warning(err)
                citation.verified = False
                citation.status = "quote_not_found"
                citation.matched_span = None
                citation.verification_error = err
                unverified_list.append(citation)
                errors.append(err)
                continue

            # Quote found: run Entailment Guard on figures (numbers, percentages, years)
            claim_text = citation.claim or extracted_claim
            unsupported_figs = check_unsupported_figures(claim_text, source_content)

            if unsupported_figs:
                err = (
                    f"Citation [{idx}] claim contains unsupported figure(s): "
                    f"{', '.join(unsupported_figs)} not found in source text."
                )
                logger.warning(err)
                citation.verified = False
                citation.status = "unsupported_figure"
                citation.matched_span = span
                citation.verification_error = err
                unverified_list.append(citation)
                errors.append(err)
            else:
                citation.verified = True
                citation.status = "verified"
                citation.matched_span = span
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
    all_citations.sort(key=lambda c: c.index)

    update = {"citations": all_citations}
    if not res.all_verified:
        logger.warning(f"Citation verification failed with {len(res.errors)} error(s).")
        update["errors"] = state.errors + res.errors

    return update
