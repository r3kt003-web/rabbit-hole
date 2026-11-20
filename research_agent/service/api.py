import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import (
    BackgroundTasks,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Security,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader

from research_agent.config.settings import get_settings
from research_agent.service.runner import (
    Event,
    ResearchOptions,
    cancel_run,
    run_research,
)
from research_agent.service.schemas import (
    CancelResponse,
    RunCreateRequest,
    RunCreateResponse,
    RunStatusResponse,
    RunSummaryItem,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Autonomous Research Agent API",
    description="Single-process FastAPI service with live SSE streaming for autonomous research orchestration.",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional API Key Authentication
api_key_header_scheme = APIKeyHeader(name="X-API-KEY", auto_error=False)


def verify_api_key(api_key: str | None = Security(api_key_header_scheme)) -> None:
    expected_key = get_settings().api_key
    if expected_key and (not api_key or api_key != expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-KEY header.",
        )


# In-memory run tracking (single-process architecture; persistent artifacts live in ./runs/)
_run_memory: dict[str, dict[str, Any]] = {}
_run_event_streams: dict[str, list[Event]] = {}
_run_event_queues: dict[str, list[asyncio.Queue[Event]]] = {}


async def _background_research_task(
    run_id: str, question: str, options: ResearchOptions
) -> None:
    _run_memory[run_id] = {
        "run_id": run_id,
        "question": question,
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "final_report": None,
        "sources": [],
        "evaluator_scores": None,
        "metrics_summary": None,
        "errors": [],
    }
    _run_event_streams[run_id] = []
    _run_event_queues[run_id] = []

    try:
        async for event in run_research(
            question=question, options=options, run_id=run_id
        ):
            _run_event_streams[run_id].append(event)
            # Fan out to connected SSE clients
            for q in list(_run_event_queues.get(run_id, [])):
                q.put_nowait(event)

            if event.event_type == "run_completed":
                _run_memory[run_id]["status"] = "completed"
                _run_memory[run_id]["final_report"] = event.payload.get("draft")
                _run_memory[run_id]["sources"] = event.payload.get("citations", [])
                _run_memory[run_id]["evaluator_scores"] = event.payload.get(
                    "eval_result"
                )
                _run_memory[run_id]["metrics_summary"] = event.payload.get("metrics")
            elif event.event_type == "run_failed":
                _run_memory[run_id]["status"] = "failed"
                err = event.payload.get("error", "Unknown error")
                _run_memory[run_id]["errors"].append(err)

    except Exception:
        logger.exception("Background task exception for %s", run_id)
        _run_memory[run_id]["status"] = "failed"
        _run_memory[run_id]["errors"].append("Background task encountered error")


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Autonomous Research Agent"}


@app.post("/runs", response_model=RunCreateResponse)
async def create_run(
    req: RunCreateRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    verify_api_key(request.headers.get("X-API-KEY"))

    clean_slug = "".join(
        c for c in req.question[:15].replace(" ", "_") if c.isalnum() or c == "_"
    )
    run_id = (
        f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{clean_slug or 'query'}"
    )
    now_iso = datetime.now(UTC).isoformat()

    # Initialize run record in memory immediately
    _run_memory[run_id] = {
        "run_id": run_id,
        "question": req.question,
        "status": "started",
        "created_at": now_iso,
        "final_report": None,
        "sources": [],
        "evaluator_scores": None,
        "metrics_summary": None,
        "errors": [],
    }
    _run_event_streams.setdefault(run_id, [])
    _run_event_queues.setdefault(run_id, [])

    options = ResearchOptions(
        max_revisions=req.max_revisions,
        max_cost_usd=req.max_cost_usd,
        replay_run_id=req.replay_run_id,
        pass_threshold=req.pass_threshold,
    )

    background_tasks.add_task(_background_research_task, run_id, req.question, options)

    return RunCreateResponse(
        run_id=run_id,
        status="started",
        created_at=now_iso,
    )


@app.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: str,
    request: Request,
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
):
    verify_api_key(request.headers.get("X-API-KEY"))

    client_queue: asyncio.Queue[Event] = asyncio.Queue()
    _run_event_queues.setdefault(run_id, []).append(client_queue)

    async def event_generator() -> AsyncIterator[str]:
        # 1. Replay historical events if client reconnecting with Last-Event-ID or starting late
        past_events = _run_event_streams.get(run_id, [])
        replay = not bool(last_event_id)

        for evt in past_events:
            if not replay:
                if evt.event_id == last_event_id:
                    replay = True
                continue
            yield f"id: {evt.event_id}\nevent: {evt.event_type}\ndata: {json.dumps(evt.model_dump())}\n\n"
            if evt.event_type in ("run_completed", "run_failed"):
                return

        # 2. Stream live events
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(client_queue.get(), timeout=15.0)
                    yield f"id: {event.event_id}\nevent: {event.event_type}\ndata: {json.dumps(event.model_dump())}\n\n"
                    if event.event_type in ("run_completed", "run_failed"):
                        break
                except TimeoutError:
                    # SSE keepalive comment
                    yield ": keepalive\n\n"
        finally:
            if (
                run_id in _run_event_queues
                and client_queue in _run_event_queues[run_id]
            ):
                _run_event_queues[run_id].remove(client_queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/runs/{run_id}", response_model=RunStatusResponse)
def get_run_status(run_id: str, request: Request):
    verify_api_key(request.headers.get("X-API-KEY"))

    # Check memory first
    if run_id in _run_memory:
        data = _run_memory[run_id]
        return RunStatusResponse(**data)

    # Check disk
    run_dir = Path("runs") / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    report = None
    report_file = run_dir / "final_report.md"
    if report_file.exists():
        report = report_file.read_text(encoding="utf-8")

    metrics = None
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        try:
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"Could not read metrics file {metrics_file}: {e}")

    return RunStatusResponse(
        run_id=run_id,
        status="completed" if report else "unknown",
        question="Historical run from disk",
        final_report=report,
        metrics_summary=metrics,
    )


@app.get("/runs/{run_id}/trace")
def get_run_trace(run_id: str, request: Request):
    verify_api_key(request.headers.get("X-API-KEY"))

    trace_file = Path("runs") / run_id / "trace.jsonl"
    if not trace_file.exists():
        raise HTTPException(
            status_code=404, detail=f"Trace for run '{run_id}' not found."
        )

    spans = []
    try:
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    spans.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Error reading trace: {e}") from e

    return {"run_id": run_id, "spans": spans}


@app.get("/runs", response_model=list[RunSummaryItem])
def list_runs(request: Request, limit: int = Query(default=20, ge=1, le=100)):
    verify_api_key(request.headers.get("X-API-KEY"))

    runs_dir = Path("runs")
    if not runs_dir.exists():
        return []

    items: list[RunSummaryItem] = []
    subdirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    for d in subdirs[:limit]:
        rid = d.name
        status_val = "completed" if (d / "final_report.md").exists() else "unknown"
        cost = 0.0

        metrics_file = d / "metrics.json"
        if metrics_file.exists():
            try:
                m = json.loads(metrics_file.read_text(encoding="utf-8"))
                cost = m.get("total_cost_usd", 0.0)
            except (OSError, json.JSONDecodeError) as e:
                logger.debug(f"Could not read metrics file {metrics_file}: {e}")

        created_str = datetime.fromtimestamp(d.stat().st_ctime, UTC).isoformat()
        items.append(
            RunSummaryItem(
                run_id=rid,
                question=rid,
                status=status_val,
                created_at=created_str,
                cost_usd=cost,
            )
        )

    return items


@app.post("/runs/{run_id}/cancel", response_model=CancelResponse)
def cancel_active_run(run_id: str, request: Request):
    verify_api_key(request.headers.get("X-API-KEY"))
    success = cancel_run(run_id)
    if run_id in _run_memory:
        _run_memory[run_id]["status"] = "cancelled"
    return CancelResponse(
        run_id=run_id,
        success=success,
        message=f"Run {run_id} cancellation requested."
        if success
        else "Run was not active in memory.",
    )
