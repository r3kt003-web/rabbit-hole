ROLE
You are a senior Python engineer doing a truthfulness-and-hardening pass on this repo (LangGraph research agent). Priority order: correctness of claims > verifier quality > CI > presentation. Work in the numbered tasks below. After each task: run `ruff check .` and `pytest`, show me the output, make one git commit with a conventional message, then stop and wait for my approval. Never invent output. If something cannot be verified, say so.

RULES
- Every statement in README.md must be verifiable from the code or from a real run artifact. Delete anything you cannot verify.
- Never paste or fabricate sample output. Sample output may only come from a real run whose artifacts are committed under docs/sample_run/.
- Do not commit secrets, .env, .cache/ or runs/ (verify .gitignore covers them; fix if not).
- Ask me before adding any dependency other than `rapidfuzz`.

TASK 1: Audit the README against the code
- List every claim in README.md as a table: claim | file/test proving it | verified yes/no. Show me the table before editing.
- Known suspects: "verified in CI" (no workflow exists), "53 tests" (count them with `pytest --collect-only -q`), "exact evidence quotes" vs fuzzy matching at 0.70, "eliminating hallucinations", the Gemini free-tier numbers (15 RPM / 200 RPD), "Python 3.13 tested", and the project-structure root name "autonomus-research-agent/" (typo, wrong name).
- Fix or remove each unverified claim. Use one CLI name everywhere: add a `[project.scripts] agent = "research_agent.cli:app"` entry so `agent run` and `agent trace` really work, then use `agent ...` throughout the README.

TASK 2: Model and quota validation
- The defaults are gemini-2.0-flash and gemini-2.0-flash-lite. Add scripts/check_models.py that lists the models available to my GEMINI_API_KEY and reports whether each configured model name exists.
- On startup, settings validation must fail fast with a clear message if a configured model is unavailable (not a raw 404 mid-run).
- Update .env.example defaults to models that actually appear in that listing. Ask me which ones to pick after showing me the list.
- In the README, stop stating free-tier numbers as fact. Say "check your limits in AI Studio" and keep LLM_RPM / LLM_RPD as user-set values.

TASK 3: Harden the citation verifier (research_agent/verification/citations.py)
- First read the current implementation and tell me exactly how it matches quotes against page text and what it does NOT catch.
- Matching: normalize both texts (lowercase, collapse whitespace, strip punctuation differences). Use windowed fuzzy matching or rapidfuzz.partial_ratio, never SequenceMatcher on a whole page against a short quote. Make the threshold configurable (VERIFY_QUOTE_THRESHOLD, default 0.85).
- Entailment guard (deterministic, no LLM): every number, percentage, unit and year in a claim must appear in the cited quote or source text; flag mismatches as "unsupported_figure".
- Output per citation: status in {verified, quote_not_found, unsupported_figure, unresolved_source} plus the matched span.
- Tests (add to tests/test_citations.py): fabricated quote fails; real quote attached to an unrelated claim with different numbers fails; whitespace/case variations pass; phantom [n] fails; threshold boundary behaves.
- Update the README verifier section to state precisely what it catches and what it doesn't (it does not prove semantic entailment).

TASK 4: Internal-consistency tests
Add tests/test_consistency.py asserting, using a mocked run: (a) number of research workers equals number of planned sub-questions on the first pass; (b) every URL in the report's Sources section appears in findings; (c) total tokens in metrics = sum of input + output across all nodes; (d) per-node costs sum to the total; (e) the "remaining quota" figure equals RPD minus the recorded LLM calls. Fix any bug these expose.

TASK 5: CI
- Add .github/workflows/ci.yml: on push and pull_request, set up Python 3.11 and 3.13 matrix, install with uv, run `ruff check .` and `pytest` (offline tests only; live test stays skipped without keys). Add a CI status badge to the README.
- Only after CI exists may the README mention CI.

TASK 6: Real sample run
- I will run the agent myself with real keys. Provide a script scripts/save_sample_run.py that copies the latest run's report.md, trace.jsonl, metrics.json into docs/sample_run/ and prints the span tree and cost table in markdown.
- Replace the README "Sample Run Output" section with the real output from docs/sample_run/ once I provide it. Until then, delete the current sample section and leave a clearly marked TODO. Do not generate it.

TASK 7: Hygiene
- Add LICENSE (ask me which license).
- Add a "Limitations" bullet about quote-level (not semantic) verification and free-tier throughput.
- Add docs/ folder placeholder for a Streamlit screenshot; reference it in the README only after I add the image.

DONE WHEN
- pytest and ruff pass in CI.
- Every README claim is backed by code, a test or docs/sample_run/.
- I have approved each task's commit.
Start with Task 1 now.