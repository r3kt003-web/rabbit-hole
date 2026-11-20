import asyncio
import re
from unittest.mock import AsyncMock, patch

import pytest

from research_agent.config.settings import get_settings
from research_agent.graph.builder import route_to_researchers
from research_agent.graph.nodes.planner import PlannerOutput
from research_agent.graph.nodes.synthesizer import build_source_manifest
from research_agent.graph.state import (
    EvalResult,
    EvalScores,
    Finding,
    ResearchState,
    SubQuestion,
)
from research_agent.llm.rate_limiter import get_rate_limiter, reset_rate_limiter
from research_agent.observability.metrics import calculate_run_metrics
from research_agent.observability.tracer import Span
from research_agent.service.runner import run_research
from research_agent.verification.citations import CitationVerifier


def extract_urls_from_sources_section(report_markdown: str) -> list[str]:
    """Helper to extract all URLs found in the '## Sources' section of a report."""
    parts = re.split(r"##\s+Sources", report_markdown, flags=re.IGNORECASE)
    if len(parts) < 2:
        return []
    sources_text = parts[-1]
    return re.findall(r"https?://[^\s)\]]+", sources_text)


def test_end_to_end_mocked_run_internal_consistency():
    """TASK 4 CORE TEST:
    Executes a complete mocked run and asserts all 5 required internal-consistency invariants:
    (a) number of research workers equals number of planned sub-questions on the first pass;
    (b) every URL in the report's Sources section appears in findings;
    (c) total tokens in metrics = sum of input + output across all nodes;
    (d) per-node costs sum to the total;
    (e) the 'remaining quota' figure equals RPD minus the recorded LLM calls.
    """
    settings = get_settings()
    reset_rate_limiter()

    mock_plan = [
        SubQuestion(
            id="sq_1",
            question="What are solid-state electrolyte safety mechanisms?",
            search_intent="solid state electrolyte ceramic dendrite suppression safety",
            answer_criteria="Thermal runaway resistance and dendrite suppression",
        ),
        SubQuestion(
            id="sq_2",
            question="What is the ionic conductivity of sulfide vs oxide electrolytes?",
            search_intent="sulfide vs oxide electrolyte ionic conductivity mS/cm comparison",
            answer_criteria="Room-temperature ionic conductivity values in mS/cm",
        ),
        SubQuestion(
            id="sq_3",
            question="What are current commercial manufacturing bottlenecks for solid-state batteries?",
            search_intent="commercial solid state battery roll to roll manufacturing bottlenecks",
            answer_criteria="Manufacturing scalability and interfacial impedance",
        ),
    ]

    mock_findings = [
        Finding(
            sub_question_id="sq_1",
            claim="Solid-state ceramic electrolytes eliminate flammable liquid solvents and withstand over 200°C.",
            evidence_quote="Solid-state ceramic electrolytes eliminate flammable liquid solvents and withstand over 200°C",
            source_url="https://nature.com/articles/solid-state-safety-2024",
            source_title="Nature Energy: Solid State Safety",
            raw_content="Solid-state ceramic electrolytes eliminate flammable liquid solvents and withstand over 200°C under thermal stress tests.",
        ),
        Finding(
            sub_question_id="sq_2",
            claim="Sulfide-based electrolytes achieve room-temperature ionic conductivity exceeding 10 mS/cm.",
            evidence_quote="Sulfide-based electrolytes achieve room-temperature ionic conductivity exceeding 10 mS/cm",
            source_url="https://sciencedirect.com/science/article/pii/sulfide-conduct",
            source_title="Journal of Power Sources: Sulfide Conductivity",
            raw_content="Sulfide-based electrolytes achieve room-temperature ionic conductivity exceeding 10 mS/cm, matching liquid electrolytes.",
        ),
        Finding(
            sub_question_id="sq_3",
            claim="Interfacial resistance and stack pressure requirements remain principal barriers to roll-to-roll production.",
            evidence_quote="Interfacial resistance and stack pressure requirements remain principal barriers to roll-to-roll production",
            source_url="https://energy.gov/eere/vehicles/solid-state-manufacturing",
            source_title="DOE Vehicle Technologies Office Report",
            raw_content="Interfacial resistance and stack pressure requirements remain principal barriers to roll-to-roll production in gigafactories.",
        ),
    ]

    finding_map = {f.sub_question_id: f for f in mock_findings}

    async def mock_worker_exec(self, sub_question: SubQuestion):
        f = finding_map.get(sub_question.id)
        return [f] if f else []

    mock_report = (
        "# Assessment of Solid-State Electrolytes\n\n"
        "## Executive Summary\n"
        "Solid-state ceramic electrolytes eliminate flammable liquid solvents and withstand over 200°C [1]. "
        "Sulfide-based electrolytes achieve room-temperature ionic conductivity exceeding 10 mS/cm [2]. "
        "Interfacial resistance and stack pressure requirements remain principal barriers to roll-to-roll production [3].\n\n"
        "## Sources\n"
        "[1] Nature Energy: Solid State Safety - https://nature.com/articles/solid-state-safety-2024\n"
        "[2] Journal of Power Sources: Sulfide Conductivity - https://sciencedirect.com/science/article/pii/sulfide-conduct\n"
        "[3] DOE Vehicle Technologies Office Report - https://energy.gov/eere/vehicles/solid-state-manufacturing\n"
    )

    with (
        patch("research_agent.service.runner.validate_configured_models"),
        patch(
            "research_agent.graph.nodes.planner.plan_research",
            new=AsyncMock(return_value=PlannerOutput(sub_questions=mock_plan)),
        ),
        patch(
            "research_agent.graph.nodes.researcher.ResearchWorker.execute",
            new=mock_worker_exec,
        ),
        patch(
            "research_agent.graph.nodes.synthesizer.get_chat_model"
        ) as mock_get_chat,
        patch(
            "research_agent.graph.nodes.evaluator.evaluate_report",
            new=AsyncMock(
                return_value=EvalResult(
                    passed=True,
                    scores=EvalScores(
                        groundedness=4.8,
                        coverage=4.7,
                        citation_accuracy=5.0,
                        clarity=4.6,
                    ),
                    feedback="Comprehensive and well-grounded report.",
                )
            ),
        ),
    ):
        mock_chat = AsyncMock()
        mock_chat.ainvoke.return_value.content = mock_report
        mock_get_chat.return_value = mock_chat

        async def _execute_mock_run():
            evts = []
            async for evt in run_research("Assess next-generation solid-state batteries"):
                evts.append(evt)
            return evts

        events = asyncio.run(_execute_mock_run())

        # Collect event payloads
        plan_event = next(e for e in events if e.event_type == "plan_ready")
        worker_events = [e for e in events if e.event_type == "worker_finished"]
        completed_event = next(e for e in events if e.event_type == "run_completed")

        planned_subqs = plan_event.payload["sub_questions"]
        report_draft = completed_event.payload["draft"]
        metrics = completed_event.payload["metrics"]

        # --- (a) Number of research workers equals number of planned sub-questions on first pass ---
        assert len(planned_subqs) == 3
        assert len(worker_events) == len(planned_subqs), (
            f"Expected {len(planned_subqs)} worker executions on first pass, got {len(worker_events)}"
        )

        # --- (b) Every URL in the report's Sources section appears in findings ---
        sources_urls = extract_urls_from_sources_section(report_draft)
        assert len(sources_urls) == 3
        findings_urls = {f.source_url for f in mock_findings}
        for u in sources_urls:
            assert u in findings_urls, f"URL {u} in Sources section was not found in findings: {findings_urls}"

        # --- (c) Total tokens in metrics = sum of input + output across all nodes ---
        tokens_by_node = metrics["tokens_by_node"]
        sum_node_input = sum(node["input"] for node in tokens_by_node.values())
        sum_node_output = sum(node["output"] for node in tokens_by_node.values())
        sum_node_total = sum(node["input"] + node["output"] for node in tokens_by_node.values())

        assert metrics["total_input_tokens"] == sum_node_input
        assert metrics["total_output_tokens"] == sum_node_output
        assert metrics["total_tokens"] == sum_node_total
        assert metrics["total_tokens"] == metrics["total_input_tokens"] + metrics["total_output_tokens"]

        # --- (d) Per-node costs sum to the total ---
        cost_by_node = metrics["cost_by_node"]
        sum_node_costs = sum(cost_by_node.values())
        assert round(sum_node_costs, 6) == metrics["total_cost_usd"], (
            f"Sum of per-node costs ({sum_node_costs:.6f}) does not equal total_cost_usd ({metrics['total_cost_usd']:.6f})"
        )
        assert metrics["total_cost_usd"] == pytest.approx(sum_node_costs, abs=1e-6)

        # --- (e) The 'remaining quota' figure equals RPD minus the recorded LLM calls ---
        recorded_llm_calls = metrics["number_of_llm_calls"]
        assert recorded_llm_calls > 0
        expected_remaining = max(0, settings.llm_rpd - recorded_llm_calls)
        assert metrics["remaining_daily_quota"] == expected_remaining, (
            f"Remaining quota {metrics['remaining_daily_quota']} != RPD ({settings.llm_rpd}) - calls ({recorded_llm_calls})"
        )


def test_worker_fanout_matches_planned_subquestions():
    """INVARIANT (a): Tests that route_to_researchers dispatches an exact 1:1 match
    of Send('research_worker', ...) for each planned sub-question on the first pass.
    """
    for count in (1, 2, 4, 6):
        sub_qs = [
            SubQuestion(
                id=f"sq_{i}",
                question=f"Question {i}?",
                search_intent=f"Intent {i}",
                answer_criteria=f"Criteria {i}",
            )
            for i in range(count)
        ]
        state = ResearchState(question="Fanout test", plan=sub_qs, trace_id="trace_test")
        sends = route_to_researchers(state)

        assert len(sends) == count
        for send_cmd, expected_sq in zip(sends, sub_qs, strict=True):
            assert send_cmd.node == "research_worker"
            assert send_cmd.arg["sub_question"]["id"] == expected_sq.id


def test_sources_manifest_and_report_url_consistency():
    """INVARIANT (b): Tests that every URL appearing in the report's Sources section
    originates from the retrieved findings, and tests that unretrieved/hallucinated
    URLs fail the consistency check and citation verification.
    """
    findings = [
        Finding(
            sub_question_id="sq_1",
            claim="Claim A",
            evidence_quote="Quote A",
            source_url="https://verified-source-a.org/paper1",
            source_title="Source A",
            raw_content="Quote A in source content.",
        ),
        Finding(
            sub_question_id="sq_2",
            claim="Claim B",
            evidence_quote="Quote B",
            source_url="https://verified-source-b.com/data",
            source_title="Source B",
            raw_content="Quote B in source content.",
        ),
    ]

    _manifest_text, citations = build_source_manifest(findings)
    finding_urls = {f.source_url for f in findings}

    for c in citations:
        assert c.source_url in finding_urls

    valid_report = (
        "# Valid Report\n\n"
        "Claim A [1] and Claim B [2].\n\n"
        "## Sources\n"
        "[1] Source A - https://verified-source-a.org/paper1\n"
        "[2] Source B - https://verified-source-b.com/data\n"
    )
    urls = extract_urls_from_sources_section(valid_report)
    assert len(urls) == 2
    for u in urls:
        assert u in finding_urls

    # Negative case: report citing a hallucinated URL not present in findings
    hallucinated_report = (
        "# Hallucinated Report\n\n"
        "Fabricated claim [3].\n\n"
        "## Sources\n"
        "[3] Phantom Source - https://hallucinated-unretrieved-source.org/fake\n"
    )
    bad_urls = extract_urls_from_sources_section(hallucinated_report)
    assert len(bad_urls) == 1
    assert bad_urls[0] not in finding_urls

    # Citation verifier must flag the unretrieved source
    verifier = CitationVerifier()
    v_res = verifier.verify(hallucinated_report, findings, existing_citations=citations)
    assert v_res.all_verified is False
    assert any(c.status == "unresolved_source" for c in v_res.unverified_citations)


def test_token_accounting_across_all_nodes():
    """INVARIANT (c): Tests that total tokens in RunMetrics equals the exact sum
    of input + output tokens across every node in tokens_by_node.
    """
    spans = [
        Span(
            run_id="test_tok",
            span_id="s1",
            name="planner",
            input_tokens=420,
            output_tokens=180,
            model_name="gemini-2.5-flash",
        ),
        Span(
            run_id="test_tok",
            span_id="s2",
            name="research_worker_0",
            input_tokens=1250,
            output_tokens=310,
            model_name="gemini-2.5-flash-lite",
        ),
        Span(
            run_id="test_tok",
            span_id="s3",
            name="research_worker_1",
            input_tokens=980,
            output_tokens=270,
            model_name="gemini-2.5-flash-lite",
        ),
        Span(
            run_id="test_tok",
            span_id="s4",
            name="serper_search",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.001,
        ),
        Span(
            run_id="test_tok",
            span_id="s5",
            name="synthesizer",
            input_tokens=1600,
            output_tokens=850,
            model_name="gemini-2.5-flash",
        ),
        Span(
            run_id="test_tok",
            span_id="s6",
            name="evaluator",
            input_tokens=750,
            output_tokens=220,
            model_name="gemini-2.5-flash",
        ),
    ]

    metrics = calculate_run_metrics(run_id="test_tok", spans=spans)

    node_inp_sum = sum(node["input"] for node in metrics.tokens_by_node.values())
    node_out_sum = sum(node["output"] for node in metrics.tokens_by_node.values())
    node_tot_sum = sum(node["input"] + node["output"] for node in metrics.tokens_by_node.values())

    assert metrics.total_input_tokens == 420 + 1250 + 980 + 1600 + 750 == 5000
    assert metrics.total_output_tokens == 180 + 310 + 270 + 850 + 220 == 1830
    assert metrics.total_tokens == 6830
    assert metrics.total_input_tokens == node_inp_sum
    assert metrics.total_output_tokens == node_out_sum
    assert metrics.total_tokens == node_tot_sum


def test_per_node_costs_sum_to_total_cost():
    """INVARIANT (d): Tests that per-node costs sum to total_cost_usd without
    rounding discrepancy or missed costs across heterogeneous model and tool spans.
    """
    spans = [
        Span(
            run_id="test_cost",
            span_id="s1",
            name="planner",
            model_name="gemini-2.5-flash",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.000300,
        ),
        Span(
            run_id="test_cost",
            span_id="s2",
            name="research_worker_0",
            model_name="gemini-2.5-flash-lite",
            input_tokens=2000,
            output_tokens=600,
            cost_usd=0.000390,
        ),
        Span(
            run_id="test_cost",
            span_id="s3",
            name="serper_search",
            cost_usd=0.001000,
        ),
        Span(
            run_id="test_cost",
            span_id="s4",
            name="synthesizer",
            model_name="gemini-2.5-flash",
            input_tokens=3000,
            output_tokens=1200,
            cost_usd=0.000780,
        ),
        Span(
            run_id="test_cost",
            span_id="s5",
            name="evaluator",
            model_name="gemini-2.5-flash",
            input_tokens=1500,
            output_tokens=400,
            cost_usd=0.000310,
        ),
    ]

    metrics = calculate_run_metrics(run_id="test_cost", spans=spans)

    sum_costs = sum(metrics.cost_by_node.values())
    assert round(sum_costs, 6) == metrics.total_cost_usd
    assert metrics.total_cost_usd == pytest.approx(sum_costs, abs=1e-6)

    # Also test spans with cost_usd=0 (inferred from model rate calculation)
    zero_cost_spans = [
        Span(
            run_id="test_inferred",
            span_id="s1",
            name="planner",
            model_name="gemini-2.5-flash",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cost_usd=0.0,
        ),
        Span(
            run_id="test_inferred",
            span_id="s2",
            name="serper_search",
            cost_usd=0.0,
        ),
    ]
    inferred_metrics = calculate_run_metrics(run_id="test_inferred", spans=zero_cost_spans)
    inferred_sum = sum(inferred_metrics.cost_by_node.values())
    assert round(inferred_sum, 6) == inferred_metrics.total_cost_usd
    assert inferred_metrics.total_cost_usd > 0.0


def test_remaining_quota_equals_rpd_minus_recorded_llm_calls():
    """INVARIANT (e): Tests that remaining_daily_quota equals RPD minus the recorded LLM calls."""
    settings = get_settings()
    reset_rate_limiter()
    limiter = get_rate_limiter()

    initial_rpd = limiter.rpd
    assert initial_rpd == settings.llm_rpd
    assert limiter.remaining_daily_quota == initial_rpd

    # Create 4 LLM spans
    spans = [
        Span(run_id="quota_run", span_id=f"sq_{i}", name=f"llm_node_{i}", model_name="gemini-2.5-flash")
        for i in range(4)
    ]
    # Plus 1 tool span with no model
    spans.append(Span(run_id="quota_run", span_id="tool_1", name="serper_search"))

    metrics = calculate_run_metrics(run_id="quota_run", spans=spans, rate_limiter=limiter)

    assert metrics.number_of_llm_calls == 4
    assert metrics.remaining_daily_quota == initial_rpd - 4
    assert limiter.remaining_daily_quota == initial_rpd - 4

    # Simulate 3 more calls
    more_spans = list(spans) + [
        Span(run_id="quota_run", span_id=f"extra_{i}", name=f"extra_llm_{i}", model_name="gemini-2.5-flash")
        for i in range(3)
    ]
    metrics2 = calculate_run_metrics(run_id="quota_run", spans=more_spans, rate_limiter=limiter)
    assert metrics2.number_of_llm_calls == 7
    assert metrics2.remaining_daily_quota == initial_rpd - 7
