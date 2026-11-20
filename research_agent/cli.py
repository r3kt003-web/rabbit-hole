import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from research_agent.observability.metrics import RunMetrics
from research_agent.observability.tracer import Span, format_span_tree
from research_agent.service.runner import ResearchOptions, run_research

cli_app = typer.Typer(
    name="agent",
    help="Autonomous Research Agent CLI: Run deep autonomous research, view traces, and inspect metrics.",
)
console = Console()


@cli_app.command(name="run")
def run_command(
    question: str = typer.Argument(
        ..., help="The primary research question to investigate"
    ),
    max_revisions: int | None = typer.Option(
        None, "--max-revisions", "-r", help="Maximum revision loops allowed"
    ),
    max_cost: float | None = typer.Option(
        None, "--max-cost", "-c", help="Hard budget limit in USD"
    ),
    replay: str | None = typer.Option(
        None,
        "--replay",
        help="Replay run from past run_id cache without spending quota",
    ),
):
    """Executes the autonomous research agent workflow on a question."""
    console.print(
        Panel.fit(
            f"[bold cyan]Autonomous Research Agent[/bold cyan]\nQuestion: [bold]{question}[/bold]"
        )
    )

    options = ResearchOptions(
        max_revisions=max_revisions,
        max_cost_usd=max_cost,
        replay_run_id=replay,
    )

    async def _execute():
        final_draft = ""
        final_metrics = None

        with console.status("[bold green]Agent working...[/bold green]") as status:
            async for event in run_research(question=question, options=options):
                etype = event.event_type
                payload = event.payload

                if etype == "run_started":
                    status.update(
                        "[bold cyan]Decomposing research question with Planner...[/bold cyan]"
                    )
                    console.print(f"[dim]Run ID: {event.run_id}[/dim]")

                elif etype == "plan_ready":
                    sub_qs = payload.get("sub_questions", [])
                    console.print(
                        f"[bold green]✓ Plan Generated:[/bold green] {len(sub_qs)} sub-questions"
                    )
                    for sq in sub_qs:
                        console.print(
                            f"  • [cyan]{sq.get('id')}[/cyan]: {sq.get('question')}"
                        )
                    status.update(
                        "[bold yellow]Parallel researchers querying Serper & fetching web pages...[/bold yellow]"
                    )

                elif etype == "worker_finished":
                    sq_id = payload.get("sub_question_id")
                    count = payload.get("findings_count", 0)
                    console.print(
                        f"  [dim]Worker [{sq_id}] extracted {count} verified finding(s)[/dim]"
                    )

                elif etype == "draft_ready":
                    citations_cnt = payload.get("citations_count", 0)
                    console.print(
                        f"[bold green]✓ Synthesizer drafted report[/bold green] ({citations_cnt} cited sources)"
                    )
                    status.update(
                        "[bold magenta]Evaluator scoring groundedness, coverage & citations...[/bold magenta]"
                    )

                elif etype == "revision_started":
                    rev = payload.get("revision_count", 1)
                    console.print(
                        f"[bold yellow]↺ Revision loop triggered (#{rev})[/bold yellow]"
                    )
                    status.update(
                        f"[bold yellow]Executing revision #{rev}...[/bold yellow]"
                    )

                elif etype == "evaluation_ready":
                    passed = payload.get("passed", False)
                    scores = payload.get("scores", {})
                    pass_str = (
                        "[bold green]PASS[/bold green]"
                        if passed
                        else "[bold red]REVISE[/bold red]"
                    )
                    console.print(
                        f"[bold]Evaluation Result:[/bold] {pass_str} "
                        f"(Groundedness: {scores.get('groundedness', 0):.1f}, "
                        f"Coverage: {scores.get('coverage', 0):.1f}, "
                        f"Citations: {scores.get('citation_accuracy', 0):.1f}, "
                        f"Clarity: {scores.get('clarity', 0):.1f})"
                    )

                elif etype == "budget_warning":
                    console.print(f"[bold red]⚠️ {payload.get('message')}[/bold red]")

                elif etype == "run_completed":
                    final_draft = payload.get("draft", "")
                    if "metrics" in payload:
                        final_metrics = RunMetrics(**payload["metrics"])

                elif etype == "run_failed":
                    console.print(
                        f"[bold red]Run failed: {payload.get('error')}[/bold red]"
                    )

        # Print Final Report
        if final_draft:
            console.print("\n" + "=" * 60)
            console.print(Markdown(final_draft))
            console.print("=" * 60 + "\n")

        # Print Metrics Summary Table
        if final_metrics:
            console.print(final_metrics.format_summary_table())

    asyncio.run(_execute())


@cli_app.command(name="trace")
def trace_command(
    run_id: str = typer.Argument(..., help="The run_id to inspect"),
):
    """Prints a readable ASCII span tree with latencies and token usage for a given run_id."""
    trace_path = Path("runs") / run_id / "trace.jsonl"
    if not trace_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] Trace file not found at {trace_path}"
        )
        raise typer.Exit(code=1)

    spans: list[Span] = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                spans.append(Span(**json.loads(line)))

    console.print(
        Panel(f"[bold]Span Tree for Run:[/bold] {run_id} ({len(spans)} spans recorded)")
    )
    tree_str = format_span_tree(spans)
    console.print(tree_str)

    # If metrics.json exists, show summary
    metrics_path = Path("runs") / run_id / "metrics.json"
    if metrics_path.exists():
        try:
            m = RunMetrics(**json.loads(metrics_path.read_text(encoding="utf-8")))
            console.print("\n" + m.format_summary_table())
        except (OSError, json.JSONDecodeError, ValueError) as err:
            console.print(f"[dim]Could not load metrics summary: {err}[/dim]")


def main():
    cli_app()


if __name__ == "__main__":
    main()
