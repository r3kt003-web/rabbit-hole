import json
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from research_agent.config.settings import get_settings
from research_agent.graph.state import ResearchState
from research_agent.llm.rate_limiter import get_rate_limiter
from research_agent.observability.tracer import Span

logger = logging.getLogger(__name__)


class ModelPricing(BaseModel):
    input_cost_per_million: float = 0.10
    output_cost_per_million: float = 0.40


class CostCalculator:
    """Computes exact cost for LLM calls and Serper searches from pricing.yaml."""

    def __init__(self, pricing_file: Path | None = None):
        settings = get_settings()
        self.pricing_file = pricing_file or settings.pricing_file
        self.pricing_data = self._load_pricing()

    def _load_pricing(self) -> dict[str, Any]:
        if not self.pricing_file.exists():
            logger.warning(
                f"Pricing file {self.pricing_file} not found. Using defaults."
            )
            return {
                "models": {
                    "default": {
                        "input_cost_per_million": 0.10,
                        "output_cost_per_million": 0.40,
                    }
                },
                "serper": {"cost_per_call": 0.001},
            }
        try:
            with open(self.pricing_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            logger.error(f"Error loading pricing yaml: {e}")
            return {}

    def calculate_token_cost(
        self, model_name: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculates token cost using per-million rates in pricing table."""
        models_table = self.pricing_data.get("models", {})
        rates = models_table.get(model_name) or models_table.get("default", {})
        input_rate = float(rates.get("input_cost_per_million", 0.10))
        output_rate = float(rates.get("output_cost_per_million", 0.40))

        cost = (input_tokens / 1_000_000.0) * input_rate + (
            output_tokens / 1_000_000.0
        ) * output_rate
        return round(cost, 6)

    def calculate_serper_cost(self, num_calls: int) -> float:
        settings = get_settings()
        serper_cfg = self.pricing_data.get("serper", {})
        per_call = float(
            serper_cfg.get("cost_per_call", settings.serper_price_per_call)
        )
        return round(num_calls * per_call, 6)


_cost_calc: CostCalculator | None = None


def get_cost_calculator() -> CostCalculator:
    global _cost_calc
    if _cost_calc is None:
        _cost_calc = CostCalculator()
    return _cost_calc


class RunMetrics(BaseModel):
    run_id: str
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    tokens_by_node: dict[str, dict[str, int]] = Field(default_factory=dict)
    cost_by_node: dict[str, float] = Field(default_factory=dict)
    tokens_by_model: dict[str, dict[str, int]] = Field(default_factory=dict)
    cost_by_model: dict[str, float] = Field(default_factory=dict)
    number_of_searches: int = 0
    serper_cost_usd: float = 0.0
    number_of_pages_fetched: int = 0
    total_latency_seconds: float = 0.0
    cost_per_cited_claim: float = 0.0
    remaining_daily_quota: int = 0
    budget_exceeded: bool = False

    def save_to_file(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)

    def format_summary_table(self) -> str:
        """Returns a formatted ASCII / Markdown summary table of metrics."""
        lines = [
            f"=== Execution Metrics for Run: {self.run_id} ===",
            f"Total Cost:            ${self.total_cost_usd:.5f} USD",
            f"Total Tokens:          {self.total_tokens:,} (Input: {self.total_input_tokens:,}, Output: {self.total_output_tokens:,})",
            f"Total Latency:         {self.total_latency_seconds:.2f}s",
            f"Google Searches:       {self.number_of_searches} (${self.serper_cost_usd:.4f})",
            f"Pages Fetched:         {self.number_of_pages_fetched}",
            f"Cost per Cited Claim:  ${self.cost_per_cited_claim:.5f}",
            f"Remaining Daily Quota: {self.remaining_daily_quota} requests",
            f"Budget Exceeded:       {self.budget_exceeded}",
            "",
            "--- Cost & Tokens by Node ---",
            f"{'Node Name':<20} | {'Input Tok':<10} | {'Output Tok':<10} | {'Cost (USD)':<10}",
            "-" * 58,
        ]
        for node, cost in sorted(self.cost_by_node.items()):
            t = self.tokens_by_node.get(node, {})
            inp = t.get("input", 0)
            out = t.get("output", 0)
            lines.append(f"{node:<20} | {inp:<10} | {out:<10} | ${cost:.5f}")

        lines.extend(
            [
                "",
                "--- Cost & Tokens by Model ---",
                f"{'Model Name':<25} | {'Input Tok':<10} | {'Output Tok':<10} | {'Cost (USD)':<10}",
                "-" * 63,
            ]
        )
        for model, cost in sorted(self.cost_by_model.items()):
            t = self.tokens_by_model.get(model, {})
            inp = t.get("input", 0)
            out = t.get("output", 0)
            lines.append(f"{model:<25} | {inp:<10} | {out:<10} | ${cost:.5f}")

        return "\n".join(lines)


def calculate_run_metrics(
    run_id: str,
    spans: list[Span],
    state: ResearchState | None = None,
    total_duration_seconds: float = 0.0,
) -> RunMetrics:
    """Aggregates all spans and state into a comprehensive RunMetrics object."""
    calculator = get_cost_calculator()
    rate_limiter = get_rate_limiter()
    settings = get_settings()

    total_input = 0
    total_output = 0
    node_tokens: dict[str, dict[str, int]] = {}
    node_cost: dict[str, float] = {}
    model_tokens: dict[str, dict[str, int]] = {}
    model_cost: dict[str, float] = {}
    searches_count = 0
    pages_count = 0

    for s in spans:
        # Tally tokens
        inp = s.input_tokens
        out = s.output_tokens
        total_input += inp
        total_output += out

        # Node grouping
        node_entry = node_tokens.setdefault(s.name, {"input": 0, "output": 0})
        node_entry["input"] += inp
        node_entry["output"] += out
        node_cost[s.name] = node_cost.get(s.name, 0.0) + s.cost_usd

        # Model grouping
        if s.model_name:
            m_entry = model_tokens.setdefault(s.model_name, {"input": 0, "output": 0})
            m_entry["input"] += inp
            m_entry["output"] += out
            m_cost = calculator.calculate_token_cost(s.model_name, inp, out)
            model_cost[s.model_name] = model_cost.get(s.model_name, 0.0) + m_cost

        # Tool count
        if "serper" in s.name.lower() or "search" in s.name.lower():
            searches_count += 1
        if "fetch" in s.name.lower():
            pages_count += 1

    # Serper cost
    serper_cost = calculator.calculate_serper_cost(searches_count)

    # Total cost = sum of model token costs + serper cost
    total_cost = sum(model_cost.values()) + serper_cost

    # Number of cited claims
    num_claims = len(state.citations) if state and state.citations else 1
    cost_per_claim = round(total_cost / max(1, num_claims), 6)

    budget_exceeded = total_cost > settings.max_run_cost_usd

    metrics = RunMetrics(
        run_id=run_id,
        total_cost_usd=round(total_cost, 6),
        total_tokens=total_input + total_output,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        tokens_by_node=node_tokens,
        cost_by_node={k: round(v, 6) for k, v in node_cost.items()},
        tokens_by_model=model_tokens,
        cost_by_model={k: round(v, 6) for k, v in model_cost.items()},
        number_of_searches=searches_count,
        serper_cost_usd=serper_cost,
        number_of_pages_fetched=pages_count,
        total_latency_seconds=round(total_duration_seconds, 2),
        cost_per_cited_claim=cost_per_claim,
        remaining_daily_quota=rate_limiter.remaining_daily_quota,
        budget_exceeded=budget_exceeded,
    )

    # Save to ./runs/<run_id>/metrics.json
    metrics_path = Path("runs") / run_id / "metrics.json"
    metrics.save_to_file(metrics_path)

    return metrics
