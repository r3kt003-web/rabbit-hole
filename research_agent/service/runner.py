import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from research_agent.config.settings import get_settings
from research_agent.graph.builder import build_graph
from research_agent.graph.state import ResearchState
from research_agent.observability.metrics import calculate_run_metrics
from research_agent.observability.tracer import get_tracer

logger = logging.getLogger(__name__)


class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    run_id: str
    event_type: str
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    payload: dict[str, Any] = Field(default_factory=dict)


class ResearchOptions(BaseModel):
    max_revisions: int | None = None
    max_cost_usd: float | None = None
    replay_run_id: str | None = None
    pass_threshold: float | None = None


# In-memory registry of active cancellation events
_active_cancellations: dict[str, asyncio.Event] = {}


def get_cancellation_event(run_id: str) -> asyncio.Event:
    if run_id not in _active_cancellations:
        _active_cancellations[run_id] = asyncio.Event()
    return _active_cancellations[run_id]


def cancel_run(run_id: str) -> bool:
    if run_id in _active_cancellations:
        _active_cancellations[run_id].set()
        logger.info(f"Cancellation event set for run: {run_id}")
        return True
    return False


async def run_research(
    question: str,
    options: ResearchOptions | None = None,
    run_id: str | None = None,
    cancel_event: asyncio.Event | None = None,
) -> AsyncIterator[Event]:
    """Single execution path for Autonomous Research Agent.

    Wraps the compiled graph and yields structured Events in real time for CLI, API (SSE), and UI.
    """
    settings = get_settings()
    options = options or ResearchOptions()

    actual_run_id = (
        run_id
        or f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"
    )
    c_event = cancel_event or get_cancellation_event(actual_run_id)

    max_revisions = (
        options.max_revisions
        if options.max_revisions is not None
        else settings.max_revisions
    )
    max_cost_usd = (
        options.max_cost_usd
        if options.max_cost_usd is not None
        else settings.max_run_cost_usd
    )

    tracer = get_tracer(actual_run_id)
    queue: asyncio.Queue[Event] = asyncio.Queue()

    # Event emitter hook connecting tracer to SSE / event queue
    def on_tracer_event(event_type: str, payload: dict[str, Any]) -> None:
        evt = Event(run_id=actual_run_id, event_type=event_type, payload=payload)
        queue.put_nowait(evt)

    tracer.add_event_listener(on_tracer_event)

    # Initial event: run_started
    yield Event(
        run_id=actual_run_id,
        event_type="run_started",
        payload={
            "question": question,
            "max_revisions": max_revisions,
            "max_cost_usd": max_cost_usd,
            "replay_run_id": options.replay_run_id,
        },
    )

    initial_state = ResearchState(
        question=question,
        trace_id=actual_run_id,
    )

    t0 = time.monotonic()
    graph = build_graph()

    # Async task running graph
    async def execute_graph():
        try:
            current_state = initial_state
            # Stream graph updates step by step
            async for step_output in graph.astream(initial_state):
                if c_event.is_set():
                    logger.warning(f"Run {actual_run_id} cancelled cooperatively.")
                    queue.put_nowait(
                        Event(
                            run_id=actual_run_id,
                            event_type="run_failed",
                            payload={"error": "Run cancelled cooperatively by user."},
                        )
                    )
                    return

                # Check budget guard on accumulated cost
                metrics = calculate_run_metrics(
                    actual_run_id, tracer.spans, current_state, time.monotonic() - t0
                )
                if metrics.total_cost_usd >= max_cost_usd:
                    logger.warning(
                        f"Hard budget limit exceeded: ${metrics.total_cost_usd:.4f} >= ${max_cost_usd}"
                    )
                    queue.put_nowait(
                        Event(
                            run_id=actual_run_id,
                            event_type="budget_warning",
                            payload={
                                "current_cost": metrics.total_cost_usd,
                                "max_budget": max_cost_usd,
                                "message": f"Cost reached budget guard limit of ${max_cost_usd:.2f} USD. Halting gracefully.",
                            },
                        )
                    )
                    current_state.budget_exceeded = True
                    # Append note to draft and stop
                    if current_state.draft:
                        current_state.draft += (
                            f"\n\n> **[BUDGET GUARD HALT]**: Execution halted gracefully as total cost "
                            f"reached ${metrics.total_cost_usd:.4f} USD (budget: ${max_cost_usd:.2f} USD)."
                        )
                    break

                # Extract and map node events
                for node_name, node_update in step_output.items():
                    if node_name == "planner" and "plan" in node_update:
                        plan = node_update["plan"]
                        current_state.plan = plan
                        queue.put_nowait(
                            Event(
                                run_id=actual_run_id,
                                event_type="plan_ready",
                                payload={
                                    "sub_questions": [sq.model_dump() for sq in plan]
                                },
                            )
                        )
                    elif node_name == "research_worker" and "findings" in node_update:
                        new_findings = node_update["findings"]
                        queue.put_nowait(
                            Event(
                                run_id=actual_run_id,
                                event_type="worker_finished",
                                payload={
                                    "findings_count": len(new_findings),
                                    "sub_question_id": new_findings[0].sub_question_id
                                    if new_findings
                                    else "unknown",
                                },
                            )
                        )
                    elif node_name == "synthesizer" and "draft" in node_update:
                        current_state.draft = node_update["draft"]
                        current_state.citations = node_update.get("citations", [])
                        current_state.revision_count = node_update.get(
                            "revision_count", current_state.revision_count
                        )
                        if current_state.revision_count > 0:
                            queue.put_nowait(
                                Event(
                                    run_id=actual_run_id,
                                    event_type="revision_started",
                                    payload={
                                        "revision_count": current_state.revision_count
                                    },
                                )
                            )
                        queue.put_nowait(
                            Event(
                                run_id=actual_run_id,
                                event_type="draft_ready",
                                payload={
                                    "draft_preview": current_state.draft[:500],
                                    "citations_count": len(current_state.citations),
                                },
                            )
                        )
                    elif node_name == "evaluator" and "eval_result" in node_update:
                        eval_res = node_update["eval_result"]
                        current_state.eval_result = eval_res
                        queue.put_nowait(
                            Event(
                                run_id=actual_run_id,
                                event_type="evaluation_ready",
                                payload={
                                    "passed": eval_res.passed,
                                    "scores": eval_res.scores.model_dump(),
                                    "missing_sub_questions": eval_res.missing_sub_questions,
                                    "unsupported_claims": eval_res.unsupported_claims,
                                    "feedback": eval_res.feedback,
                                },
                            )
                        )

            # Finalize metrics
            total_duration = time.monotonic() - t0
            final_metrics = calculate_run_metrics(
                actual_run_id, tracer.spans, current_state, total_duration
            )

            # Save state to ./runs/<run_id>/state.json
            run_dir = Path("runs") / actual_run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            report_path = run_dir / "final_report.md"
            content = current_state.draft or "No report generated."
            await asyncio.to_thread(report_path.write_text, content, encoding="utf-8")

            queue.put_nowait(
                Event(
                    run_id=actual_run_id,
                    event_type="run_completed",
                    payload={
                        "draft": current_state.draft,
                        "citations": [c.model_dump() for c in current_state.citations],
                        "eval_result": current_state.eval_result.model_dump()
                        if current_state.eval_result
                        else None,
                        "metrics": final_metrics.model_dump(),
                    },
                )
            )

        except Exception as exc:
            logger.exception(f"Unhandled error in run {actual_run_id}")
            queue.put_nowait(
                Event(
                    run_id=actual_run_id,
                    event_type="run_failed",
                    payload={"error": str(exc)},
                )
            )
        finally:
            # Sentinel to finish event queue
            queue.put_nowait(None)

    graph_task = asyncio.create_task(execute_graph())

    # Stream out events until None sentinel
    while True:
        event = await queue.get()
        if event is None:
            break
        yield event

    await graph_task
