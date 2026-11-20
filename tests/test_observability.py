import json

from research_agent.graph.state import ResearchState
from research_agent.observability.metrics import (
    calculate_run_metrics,
    get_cost_calculator,
)
from research_agent.observability.tracer import LocalTracer, Span, format_span_tree


def test_local_tracer_writes_jsonl(tmp_path):
    tracer = LocalTracer(run_id="test_run_1", runs_dir=str(tmp_path))
    span = tracer.start_span(
        "planner", span_type="node", input_summary="Planning research"
    )
    tracer.end_span(
        span,
        output_summary="Plan ready",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
    )

    trace_file = tmp_path / "test_run_1" / "trace.jsonl"
    assert trace_file.exists()

    with open(trace_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["name"] == "planner"
        assert record["input_tokens"] == 100
        assert record["latency_ms"] >= 0.0


def test_format_span_tree():
    s1 = Span(
        run_id="r1",
        span_id="s1",
        name="planner",
        latency_ms=120.0,
        input_tokens=50,
        output_tokens=50,
        cost_usd=0.0005,
    )
    s2 = Span(
        run_id="r1",
        span_id="s2",
        parent_span_id="s1",
        name="gemini_call",
        latency_ms=100.0,
        input_tokens=50,
        output_tokens=50,
        cost_usd=0.0005,
    )
    tree_str = format_span_tree([s1, s2])
    assert "planner" in tree_str
    assert "gemini_call" in tree_str
    assert "120ms" in tree_str


def test_cost_calculator_pricing():
    calc = get_cost_calculator()
    # gemini-2.0-flash is $0.10 / $0.40 per million
    cost = calc.calculate_token_cost(
        "gemini-2.0-flash", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert abs(cost - 0.50) < 0.001

    # Serper cost
    serper_cost = calc.calculate_serper_cost(num_calls=10)
    assert abs(serper_cost - 0.01) < 0.001


def test_budget_guard_halts_run(monkeypatch):
    """REQUIRED TEST: Verifies that when run cost exceeds MAX_RUN_COST_USD,
    budget guard detects it and flags budget_exceeded.
    """
    from research_agent.config.settings import reload_settings

    monkeypatch.setenv("MAX_RUN_COST_USD", "0.005")
    reload_settings()
    try:
        # High token span exceeding the tiny budget
        s = Span(
            run_id="budget_test",
            span_id="b1",
            name="synthesizer",
            model_name="gemini-1.5-pro",  # expensive model
            input_tokens=10_000,
            output_tokens=5_000,
            cost_usd=0.0375,
        )
        state = ResearchState(question="Budget test")

        metrics = calculate_run_metrics(
            run_id="budget_test",
            spans=[s],
            state=state,
            total_duration_seconds=1.5,
        )
        assert metrics.total_cost_usd >= 0.03
        assert metrics.budget_exceeded is True
    finally:
        reload_settings()
