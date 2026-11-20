import asyncio
import json
import logging
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from research_agent.config.settings import get_settings

logger = logging.getLogger(__name__)


class Span(BaseModel):
    run_id: str
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_span_id: str | None = None
    name: str
    span_type: str = "node"  # node, llm, tool
    start_time: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    end_time: str | None = None
    latency_ms: float = 0.0
    input_summary: str = ""
    output_summary: str = ""
    model_name: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LocalTracer:
    """Local JSONL tracer that records all nodes, LLM calls, and tool calls to disk.

    Operates with zero external services and provides hooks for live SSE events and span tree rendering.
    """

    def __init__(self, run_id: str, runs_dir: str = "runs"):
        self.run_id = run_id
        self.run_dir = Path(runs_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = self.run_dir / "trace.jsonl"
        self.spans: list[Span] = []
        self._span_stack: list[str] = []
        self._event_listeners: list[Callable[[str, dict[str, Any]], Any]] = []

        settings = get_settings()
        self.langsmith_enabled = settings.langsmith_tracing

    def add_event_listener(
        self, listener: Callable[[str, dict[str, Any]], Any]
    ) -> None:
        """Register a callback for live event streaming."""
        self._event_listeners.append(listener)

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        for listener in self._event_listeners:
            try:
                res = listener(event_type, payload)
                if asyncio.iscoroutine(res):
                    asyncio.create_task(res)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Error in event listener for {event_type}: {e}")

    def start_span(
        self,
        name: str,
        span_type: str = "node",
        input_summary: str = "",
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Span:
        parent_id = self._span_stack[-1] if self._span_stack else None
        span = Span(
            run_id=self.run_id,
            parent_span_id=parent_id,
            name=name,
            span_type=span_type,
            input_summary=input_summary[:300],
            model_name=model_name,
            metadata=metadata or {},
        )
        self._span_stack.append(span.span_id)
        self.spans.append(span)

        # Notify event listeners
        self._emit_event(
            f"{span_type}_started",
            {"span_id": span.span_id, "name": name, "input": span.input_summary},
        )
        return span

    def end_span(
        self,
        span: Span,
        output_summary: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        error: str | None = None,
    ) -> None:
        now = datetime.now(UTC)
        span.end_time = now.isoformat()
        span.output_summary = output_summary[:400]
        span.input_tokens = input_tokens
        span.output_tokens = output_tokens
        span.cost_usd = cost_usd
        span.error = error

        start_dt = datetime.fromisoformat(span.start_time)
        span.latency_ms = (now - start_dt).total_seconds() * 1000.0

        if self._span_stack and self._span_stack[-1] == span.span_id:
            self._span_stack.pop()

        # Append to trace.jsonl immediately
        self._append_to_file(span)

        # Notify event listeners
        self._emit_event(
            f"{span.span_type}_finished",
            {
                "span_id": span.span_id,
                "name": span.name,
                "latency_ms": span.latency_ms,
                "cost_usd": span.cost_usd,
                "error": span.error,
            },
        )

    def _append_to_file(self, span: Span) -> None:
        try:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(span.model_dump(), ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error(
                f"Failed writing span {span.span_id} to {self.trace_file}: {e}"
            )

    @asynccontextmanager
    async def span(
        self,
        name: str,
        span_type: str = "node",
        input_summary: str = "",
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Async context manager for tracing a block or node."""
        s = self.start_span(
            name=name,
            span_type=span_type,
            input_summary=input_summary,
            model_name=model_name,
            metadata=metadata,
        )
        try:
            yield s
        except Exception as exc:
            self.end_span(s, error=str(exc))
            raise
        else:
            self.end_span(s)


_tracers_by_run: dict[str, LocalTracer] = {}


def get_tracer(run_id: str) -> LocalTracer:
    if run_id not in _tracers_by_run:
        _tracers_by_run[run_id] = LocalTracer(run_id=run_id)
    return _tracers_by_run[run_id]


@asynccontextmanager
async def trace_span(
    run_id: str, name: str, span_type: str = "node", input_summary: str = "", **kwargs
):
    tracer = get_tracer(run_id)
    async with tracer.span(
        name, span_type=span_type, input_summary=input_summary, **kwargs
    ) as s:
        yield s


def format_span_tree(spans: list[Span]) -> str:
    """Renders a readable ASCII span tree with latencies and token usage."""
    if not spans:
        return "No spans recorded."

    # Build parent -> children map
    children_map: dict[str | None, list[Span]] = {}
    for s in spans:
        children_map.setdefault(s.parent_span_id, []).append(s)

    lines: list[str] = []

    def render_node(span: Span, prefix: str = "", is_last: bool = True) -> None:
        connector = "└── " if is_last else "├── "
        duration = f"{span.latency_ms:.0f}ms" if span.latency_ms > 0 else "running"
        tokens = (
            f"{span.input_tokens + span.output_tokens} tok"
            if (span.input_tokens or span.output_tokens)
            else ""
        )
        cost = f"${span.cost_usd:.4f}" if span.cost_usd > 0 else ""
        err = f" [FAILED: {span.error}]" if span.error else ""

        info_parts = [p for p in (duration, tokens, cost) if p]
        info_str = f" ({', '.join(info_parts)})" if info_parts else ""

        lines.append(
            f"{prefix}{connector}{span.name} [{span.span_type}]{info_str}{err}"
        )

        child_prefix = prefix + ("    " if is_last else "│   ")
        children = children_map.get(span.span_id, [])
        for i, child in enumerate(children):
            render_node(child, child_prefix, i == len(children) - 1)

    root_spans = children_map.get(None, [])
    for i, root in enumerate(root_spans):
        render_node(root, "", i == len(root_spans) - 1)

    return "\n".join(lines)
