from typing import Any

from pydantic import BaseModel, Field


class RunCreateRequest(BaseModel):
    question: str = Field(description="Primary research question")
    max_revisions: int | None = Field(
        default=None, description="Max revisions allowed (defaults to config)"
    )
    max_cost_usd: float | None = Field(
        default=None, description="Max budget in USD (defaults to config)"
    )
    replay_run_id: str | None = Field(
        default=None, description="Optional past run_id to replay from cache"
    )
    pass_threshold: float | None = Field(
        default=None, description="Configurable pass threshold for evaluator"
    )


class RunCreateResponse(BaseModel):
    run_id: str
    status: str = "started"
    created_at: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str  # running, completed, failed, cancelled
    question: str
    final_report: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    evaluator_scores: dict[str, Any] | None = None
    metrics_summary: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)


class RunSummaryItem(BaseModel):
    run_id: str
    question: str
    status: str
    created_at: str
    cost_usd: float = 0.0


class CancelResponse(BaseModel):
    run_id: str
    success: bool
    message: str
