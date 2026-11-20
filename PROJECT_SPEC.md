ROLE
You are a senior Python engineer building a portfolio-grade Autonomous Research Agent. Work in phases. After each phase, run the code and tests, show me the result, and stop for my confirmation before the next phase. Do not stub things silently. If something is mocked, label it MOCK in code and in the README.

GOAL
Given a research question, the agent plans sub-questions, researches them in parallel using web search, writes a cited report, has the report scored by an evaluator, and revises it if the score is too low. Every run produces a trace and a cost report.

STACK (fixed)
- Python 3.11+, LangGraph for orchestration, Pydantic v2 for all state and schemas
- LLM: Google Gemini via `langchain-google-genai`. Model names come from .env (PLANNER_MODEL, RESEARCHER_MODEL, WRITER_MODEL, EVALUATOR_MODEL) so I can mix a cheaper model for research and a stronger one for planning and evaluation
- Search: Serper (https://google.serper.dev/search) through a tool wrapper. Page reading: `httpx` + `trafilatura` for clean text extraction
- Config: python-dotenv, .env.example committed, no secrets in code
- Tests: pytest. Lint: ruff. CLI: typer
- API: FastAPI + Uvicorn, with Server-Sent Events for live progress
- UI: Streamlit (thin client that talks to the FastAPI over HTTP only; it must never import the graph or call Gemini/Serper directly)

ARCHITECTURE (LangGraph)
State = Pydantic model containing: question, plan (list of SubQuestion), findings (list of Finding, merged with a reducer), draft, citations, eval_result, revision_count, trace_id.

Nodes:
1. planner: decomposes the question into 3-6 independent sub-questions, each with a search intent and an "answer criteria" field. Structured output only (Pydantic schema), no free-text parsing.
2. research_worker: runs once per sub-question, in parallel, using LangGraph's `Send` API for fan-out. Each worker: generate 1-3 search queries -> call Serper -> select top URLs -> fetch and extract text -> produce a Finding {sub_question_id, claim, evidence_quote, source_url, source_title, retrieved_at, confidence}. Enforce a concurrency limit (asyncio semaphore, configurable) and per-worker timeouts and retries with backoff.
3. synthesizer: merges the findings and drops duplicates. It writes the report with numbered inline citations [1], [2]. Every factual claim must map to at least one Finding. The agent may not cite a URL that was not actually retrieved in this run.
4. evaluator: a separate LLM call, with a different prompt and ideally a different model from the writer. It scores the draft 1-5 on: groundedness (are claims supported by cited evidence), coverage (does it answer every sub-question), citation accuracy, and clarity. It returns structured JSON: scores, list of unsupported claims, list of missing sub-questions, pass/fail. Pass threshold is configurable.
5. Conditional edge: if fail and revision_count < MAX_REVISIONS (default 2), route back to research_worker for only the missing sub-questions or to the synthesizer for fixes, depending on the evaluator's issue type. Otherwise finish and record the final score.

CITATION RULES
- A deterministic citation verifier runs before the evaluator: check that every [n] resolves to a retrieved source, and that each cited evidence_quote appears in that source's extracted text (fuzzy match is acceptable, with the threshold in config). Flag failures in state. Do not rely on the LLM to self-police this.
- The final output has a "Sources" section with title, URL and retrieval date.

TRACING
- Implement a tracer that wraps every node, LLM call and tool call, recording: run_id, span_id, parent_span_id, node name, start/end time, latency, input/output summary (truncated), model name, token counts, error if any.
- Write traces as JSONL to ./runs/<run_id>/trace.jsonl.
- Add optional LangSmith support behind an env flag, but the local JSONL tracer must work with zero external services.
- Provide a CLI command `agent trace <run_id>` that prints a readable span tree with timings.

COST METRICS
- Read token usage from the Gemini response usage metadata for every call. Compute cost from a pricing table in `config/pricing.yaml` (per 1M input/output tokens per model, editable by me). Do not hardcode prices in code.
- Count Serper calls as a separate cost line (configurable price per call).
- Per run, produce ./runs/<run_id>/metrics.json and print a summary table: tokens and cost per node, per model, total, plus total latency, number of searches, number of pages fetched, and cost per cited claim.
- Add a hard budget guard: if cost exceeds MAX_RUN_COST_USD, stop gracefully and return the partial report with a note.
7. API + UI

service/runner.py
- A single `run_research(question, options) -> AsyncIterator[Event]` function that wraps the compiled graph. Both the CLI and the API call this, so there is exactly one execution path.
- Events: run_started, plan_ready, worker_started/finished (per sub-question), draft_ready, evaluation_ready, revision_started, budget_warning, run_completed, run_failed. Each event is a Pydantic model with run_id, timestamp and payload. Emit them from the tracer hooks, not by duplicating logic.

service/api.py (FastAPI)
- POST /runs -> body {question, max_revisions?, max_cost_usd?}; starts the run as a background task and returns {run_id} immediately.
- GET /runs/{run_id}/events -> SSE stream of the events above; supports reconnect using the Last-Event-ID header by replaying from the saved trace.
- GET /runs/{run_id} -> status, final report, sources, evaluator scores, metrics summary.
- GET /runs/{run_id}/trace -> the span list as JSON.
- GET /runs -> recent runs (read from ./runs/).
- POST /runs/{run_id}/cancel -> cooperative cancellation.
- GET /health.
- Pydantic request/response models in service/schemas.py, CORS configured from env, optional API key header (API_KEY env; if unset, auth is off). Run state lives in memory plus ./runs on disk; document that this is single-process and not production-scale.

ui/app.py (Streamlit)
- Sidebar: API base URL, max revisions, max cost, pass threshold.
- Main: question input, Run button. Consume the SSE stream and show live progress: the plan as a checklist, one status row per parallel worker, the evaluator scores per revision.
- On completion show tabs: Report (markdown with clickable citations), Sources, Evaluation (scores and unsupported claims), Trace (span tree with latencies), Cost (per-node and per-model table plus total).
- A history view listing past runs from GET /runs.
- Handle API-down, run_failed and budget-stop states with clear messages, not stack traces.

Tests
- FastAPI TestClient tests with the runner mocked: POST /runs returns an id, the SSE stream yields events in order, cancel stops a run, auth rejects a bad key when API_KEY is set.
- One test that the Streamlit module has no imports from graph/ or tools/ (grep-style assertion).

Run commands (put in README): `uvicorn research_agent.service.api:app --reload` and `streamlit run research_agent/ui/app.py`.

PROJECT STRUCTURE
research_agent/
  config/ (settings.py, pricing.yaml)
  graph/ (state.py, builder.py, nodes/{planner,researcher,synthesizer,evaluator}.py)
  tools/ (serper.py, fetch.py)
  verification/citations.py
  observability/ (tracer.py, metrics.py)
  service/ (runner.py, api.py, schemas.py)
  ui/app.py
  cli.py
tests/
README.md, .env.example, pyproject.toml

TESTS (required)
- Unit tests with mocked Serper and mocked LLM for: planner schema validity, parallel fan-out merges findings correctly, citation verifier catches a fabricated citation, evaluator failure triggers a revision loop that stops at MAX_REVISIONS, budget guard halts a run.
- One optional live integration test, skipped unless API keys are present.

PHASES
1. Scaffold, config, Pydantic state, Serper + fetch tools with tests.
2. Planner + parallel researchers (Send fan-out) with tests.
3. Synthesizer + citation verifier.
4. Evaluator + revision loop.
5. Tracer + metrics + budget guard + `agent trace` command.
6. README: architecture diagram (Mermaid), graph diagram, sample run output with a real trace and cost table, design decisions, known limitations.

CONSTRAINTS
- Type hints everywhere, no global mutable state, async for I/O.
- Handle failures explicitly: Serper rate limits, fetch timeouts, paywalled or empty pages, Gemini safety blocks or malformed structured output (retry once, then degrade gracefully and log it).
- The graph, CLI, API and UI must share one execution path (service/runner.py). No duplicated pipeline logic anywhere.
- Ask me before adding any dependency not listed here.
- Start with Phase 1 now.FREE-TIER MODE (default)
- Global async rate limiter (token bucket) shared by all LLM calls; RPM and RPD come from .env (LLM_RPM=8, LLM_RPD=200 as defaults). Researcher parallelism must go through it, not around it.
- On 429: exponential backoff with jitter, honoring any retry-after hint; never crash the run.
- Reduce calls per worker: one combined LLM call to generate queries AND pick URLs, one call to extract findings.
- Disk cache for Serper responses and fetched pages (keyed by query/URL hash) so re-runs during development cost zero API calls.
- Record/replay mode: `--replay <run_id>` re-runs the graph from cached LLM and search outputs so I can develop the evaluator, tracer and UI without spending quota.
- Use the cheapest stable Flash-Lite model for researchers and a stronger Flash model for planner/evaluator; all names in .env.
- Surface remaining daily quota in the metrics summary.

“re-read PROJECT_SPEC.md and fix deviations.”