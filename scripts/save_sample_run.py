#!/usr/bin/env python3
"""Saves the latest or specified agent run artifacts into docs/sample_run/

and prints a verified span tree and cost summary in Markdown format.

Usage:
    python scripts/save_sample_run.py [RUN_ID]
"""

import json
import shutil
import sys
from pathlib import Path

from research_agent.observability.tracer import Span, format_span_tree


def find_latest_run(runs_dir: Path) -> Path | None:
    """Finds the most recent non-test run directory in runs_dir."""
    if not runs_dir.exists():
        return None

    candidates = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        # Skip unit test run dirs unless explicitly specified
        if d.name.startswith(("test_", "budget_test", "quota_run")):
            continue
        candidates.append(d)

    if not candidates:
        return None

    # Sort by directory modification time descending
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def format_markdown_cost_table(metrics: dict) -> str:
    """Generates a clean Markdown table of node and model costs."""
    lines = [
        "### Cost & Token Metrics",
        "",
        f"- **Run ID**: `{metrics.get('run_id', 'unknown')}`",
        f"- **Total Cost**: `${metrics.get('total_cost_usd', 0.0):.5f} USD`",
        f"- **Total Tokens**: `{metrics.get('total_tokens', 0):,}` (Input: `{metrics.get('total_input_tokens', 0):,}`, Output: `{metrics.get('total_output_tokens', 0):,}`)",
        f"- **Total Duration**: `{metrics.get('total_latency_seconds', 0.0):.2f}s`",
        f"- **Web Searches**: `{metrics.get('number_of_searches', 0)}` (${metrics.get('serper_cost_usd', 0.0):.4f} USD)",
        f"- **Pages Extracted**: `{metrics.get('number_of_pages_fetched', 0)}`",
        f"- **Cost per Cited Claim**: `${metrics.get('cost_per_cited_claim', 0.0):.5f}`",
        f"- **Remaining Daily Quota**: `{metrics.get('remaining_daily_quota', 0)} requests`",
        "",
        "#### Cost Breakdown by Node",
        "",
        "| Node / Step | Input Tokens | Output Tokens | Cost (USD) |",
        "| :--- | :--- | :--- | :--- |",
    ]

    cost_by_node = metrics.get("cost_by_node", {})
    tokens_by_node = metrics.get("tokens_by_node", {})

    for node, cost in sorted(cost_by_node.items()):
        toks = tokens_by_node.get(node, {})
        inp = toks.get("input", 0)
        out = toks.get("output", 0)
        lines.append(f"| `{node}` | {inp:,} | {out:,} | ${cost:.5f} |")

    lines.extend(
        [
            "",
            "#### Cost Breakdown by Model",
            "",
            "| Model | Input Tokens | Output Tokens | Cost (USD) |",
            "| :--- | :--- | :--- | :--- |",
        ]
    )

    cost_by_model = metrics.get("cost_by_model", {})
    tokens_by_model = metrics.get("tokens_by_model", {})

    for model, cost in sorted(cost_by_model.items()):
        toks = tokens_by_model.get(model, {})
        inp = toks.get("input", 0)
        out = toks.get("output", 0)
        lines.append(f"| `{model}` | {inp:,} | {out:,} | ${cost:.5f} |")

    return "\n".join(lines)


def main() -> int:
    runs_dir = Path("runs")
    target_dir: Path | None = None

    if len(sys.argv) > 1 and sys.argv[1] not in ("-h", "--help"):
        specified = Path(sys.argv[1])
        if specified.is_dir():
            target_dir = specified
        elif (runs_dir / sys.argv[1]).is_dir():
            target_dir = runs_dir / sys.argv[1]
        else:
            print(f"Error: Specified run directory not found: {sys.argv[1]}")
            return 1
    elif len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python scripts/save_sample_run.py [RUN_ID]")
        print("Copies report.md, trace.jsonl, and metrics.json into docs/sample_run/")
        return 0
    else:
        target_dir = find_latest_run(runs_dir)

    if not target_dir or not target_dir.exists():
        print(f"Error: No run directories found in {runs_dir.resolve()}.")
        print("Run the agent first using:")
        print('  agent run "Your research question"')
        return 1

    run_id = target_dir.name
    print(f"--> Found run directory: {target_dir}")

    # Check for report file
    report_file = target_dir / "final_report.md"
    if not report_file.exists():
        report_file = target_dir / "report.md"

    trace_file = target_dir / "trace.jsonl"
    metrics_file = target_dir / "metrics.json"

    missing = []
    if not report_file.exists():
        missing.append("final_report.md / report.md")
    if not trace_file.exists():
        missing.append("trace.jsonl")
    if not metrics_file.exists():
        missing.append("metrics.json")

    if missing:
        print(f"Warning: The following files are missing in {target_dir}: {', '.join(missing)}")

    # Destination directory: docs/sample_run/
    dest_dir = Path("docs") / "sample_run"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if report_file.exists():
        dest_report = dest_dir / "report.md"
        shutil.copy2(report_file, dest_report)
        print(f"  [+] Copied report to {dest_report}")

    if trace_file.exists():
        dest_trace = dest_dir / "trace.jsonl"
        shutil.copy2(trace_file, dest_trace)
        print(f"  [+] Copied trace to {dest_trace}")

    if metrics_file.exists():
        dest_metrics = dest_dir / "metrics.json"
        shutil.copy2(metrics_file, dest_metrics)
        print(f"  [+] Copied metrics to {dest_metrics}")

    print("\n" + "=" * 70)
    print("VERIFIED SAMPLE RUN MARKDOWN SNIPPET (For README.md)")
    print("=" * 70 + "\n")

    # Render Span Tree
    spans: list[Span] = []
    if trace_file.exists():
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        spans.append(Span(**json.loads(line)))
                    except (json.JSONDecodeError, TypeError, ValueError) as err:
                        print(f"Warning: could not parse span line: {err}")

    if spans:
        print("```text")
        print(f"Execution Span Tree ({run_id})")
        print(format_span_tree(spans))
        print("```\n")

    # Render Cost Table
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        print(format_markdown_cost_table(metrics_data))

    print("\n" + "=" * 70)
    print(f"Artifacts successfully saved to: {dest_dir.resolve()}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
