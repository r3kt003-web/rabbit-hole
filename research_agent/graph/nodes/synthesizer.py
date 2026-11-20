import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.config.settings import get_settings
from research_agent.graph.state import Citation, Finding, ResearchState
from research_agent.llm.client import get_chat_model
from research_agent.llm.rate_limiter import call_with_rate_limit

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM_PROMPT = """You are an elite scientific and technical research analyst.
Your task is to write a comprehensive, rigorous, and well-structured research report answering the primary question.
You must synthesize the provided retrieved findings into an authoritative report.

STRICT CITATION RULES:
1. Every factual statement or claim MUST have a numbered inline citation marker, e.g., [1], [2], or [1][3].
2. Use ONLY the citation numbers from the provided Sources Manifest. NEVER invent a citation number or URL.
3. Every cited claim must be grounded directly in the provided evidence.
4. Structure the report logically with clear Markdown headers (e.g. ## Executive Summary, ## Key Findings, ## Analysis & Discussion, ## Conclusion).
5. You MUST include a '## Sources' section at the end listing all cited sources in numerical order, using the format:
   [n] Source Title - Source URL (Retrieved: timestamp)
"""


def build_source_manifest(findings: list[Finding]) -> tuple[str, list[Citation]]:
    """Builds a numbered list of sources and creates corresponding Citation objects."""
    # Deduplicate findings by (source_url, claim)
    unique_findings: list[Finding] = []
    seen = set()
    for f in findings:
        key = (f.source_url.strip(), f.claim.strip().lower())
        if key not in seen:
            seen.add(key)
            unique_findings.append(f)

    citations: list[Citation] = []
    manifest_lines: list[str] = []

    # Map each distinct source URL to an index, or index each key finding
    url_to_index: dict[str, int] = {}
    current_idx = 1

    for f in unique_findings:
        url = f.source_url
        if url not in url_to_index:
            url_to_index[url] = current_idx
            citation = Citation(
                index=current_idx,
                source_url=f.source_url,
                source_title=f.source_title,
                retrieved_at=f.retrieved_at,
                claim=f.claim,
                evidence_quote=f.evidence_quote,
                verified=True,
            )
            citations.append(citation)
            manifest_lines.append(
                f"[{current_idx}] Title: {f.source_title}\n"
                f"    URL: {f.source_url}\n"
                f"    Retrieved: {f.retrieved_at}\n"
                f"    Claim: {f.claim}\n"
                f'    Evidence Quote: "{f.evidence_quote}"\n'
            )
            current_idx += 1
        else:
            # Append additional finding under existing source citation
            existing_idx = url_to_index[url]
            manifest_lines.append(
                f"[{existing_idx}] (Additional Finding from same source)\n"
                f"    Claim: {f.claim}\n"
                f'    Evidence Quote: "{f.evidence_quote}"\n'
            )

    return "\n".join(manifest_lines), citations


async def synthesize_report(
    question: str,
    findings: list[Finding],
    model_name: str | None = None,
    mock_draft: str | None = None,
    eval_result: Any | None = None,
    revision_count: int = 0,
) -> tuple[str, list[Citation]]:
    """Synthesizes findings into a cited markdown report."""
    if not findings:
        empty_draft = (
            f"# Research Report: {question}\n\n"
            "## Executive Summary\n"
            "No verifiable evidence was found to answer this research question.\n\n"
            "## Sources\n"
            "No sources retrieved.\n"
        )
        return empty_draft, []

    manifest_text, citations = build_source_manifest(findings)

    if mock_draft is not None:
        return mock_draft, citations

    settings = get_settings()
    model = model_name or settings.writer_model
    chat_model = get_chat_model(model, temperature=0.1)

    revision_directive = ""
    if eval_result and revision_count > 0:
        feedback = getattr(eval_result, "feedback", "")
        unsupported = getattr(eval_result, "unsupported_claims", [])
        missing = getattr(eval_result, "missing_sub_questions", [])
        revision_directive = (
            f"\nREVISION INSTRUCTIONS (Revision #{revision_count}):\n"
            f"The previous draft was evaluated and requires the following corrections:\n"
            f"- Feedback: {feedback}\n"
            f"- Unsupported claims to fix or remove: {', '.join(unsupported) if unsupported else 'None'}\n"
            f"- Missing sub-questions to address: {', '.join(missing) if missing else 'None'}\n"
            "Ensure all issues are fully resolved and every claim has a valid inline citation.\n"
        )

    prompt = (
        f"Primary Research Question: {question}\n\n"
        f"Sources Manifest (Numbered for Citations):\n"
        f"=========================================\n"
        f"{manifest_text}\n"
        f"=========================================\n"
        f"{revision_directive}\n"
        "Write the complete, comprehensive research report with inline citations [n] referencing the manifest above."
    )

    messages = [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = await call_with_rate_limit(chat_model.ainvoke, messages)
        draft = response.content
        if isinstance(draft, list):
            draft = "\n".join(str(item) for item in draft)
    except (RuntimeError, ValueError, TypeError) as exc:
        logger.error(f"Synthesizer call failed: {exc}")
        # Construct fallback draft from findings
        fallback_lines = [
            f"# Research Report: {question}\n\n",
            "## Key Findings\n",
        ]
        for c in citations:
            fallback_lines.append(f"- {c.claim} [{c.index}]\n")
        fallback_lines.append("\n## Sources\n")
        for c in citations:
            fallback_lines.append(
                f"[{c.index}] {c.source_title} - {c.source_url} (Retrieved: {c.retrieved_at})\n"
            )
        draft = "".join(fallback_lines)

    return draft, citations


async def synthesizer_node(state: ResearchState) -> dict[str, Any]:
    """LangGraph node for report synthesis."""
    # If revising due to failed evaluation, increment revision_count
    next_revision = state.revision_count
    if state.eval_result is not None and not state.eval_result.passed:
        next_revision = state.revision_count + 1

    draft, citations = await synthesize_report(
        question=state.question,
        findings=state.findings,
        eval_result=state.eval_result,
        revision_count=next_revision,
    )
    return {
        "draft": draft,
        "citations": citations,
        "revision_count": next_revision,
    }
