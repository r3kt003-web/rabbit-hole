import json

import httpx
import streamlit as st

# Set Streamlit page layout
st.set_page_config(
    page_title="Autonomous Research Agent",
    page_icon="🔬",
    layout="wide",
)

# === Sidebar Configuration ===
st.sidebar.title("⚙️ Agent Settings")
api_base_url = st.sidebar.text_input("FastAPI Base URL", value="http://127.0.0.1:8000")
api_key = st.sidebar.text_input("API Key (Optional)", type="password", value="")

st.sidebar.subheader("Execution Parameters")
max_revisions = st.sidebar.slider("Max Revisions", min_value=1, max_value=4, value=2)
max_cost_usd = st.sidebar.slider(
    "Max Budget ($ USD)", min_value=0.5, max_value=10.0, value=2.0, step=0.5
)
pass_threshold = st.sidebar.slider(
    "Pass Threshold (1-5)", min_value=2.0, max_value=5.0, value=3.5, step=0.1
)


# Helper headers
def get_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-KEY"] = api_key
    return headers


# Check API Health
def check_api_health() -> bool:
    try:
        resp = httpx.get(f"{api_base_url}/health", timeout=3.0)
        return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


api_online = check_api_health()
if api_online:
    st.sidebar.success("● API Backend Connected")
else:
    st.sidebar.error("○ API Backend Offline")
    st.sidebar.info("Launch API via: `uvicorn research_agent.service.api:app --reload`")

# === Main Interface ===
st.title("🔬 Autonomous Research Agent")
st.caption(
    "Plan, research, synthesize, verify citations, score, and revise autonomously."
)

nav_choice = st.radio("Navigation", ["New Research", "Run History"], horizontal=True)

if nav_choice == "New Research":
    question = st.text_area(
        "Enter your research question:",
        placeholder="e.g. What are the key breakthroughs and technical hurdles in commercial nuclear fusion in 2024?",
        height=100,
    )

    col1, col2 = st.columns([1, 5])
    start_btn = col1.button(
        "Start Research",
        type="primary",
        disabled=not api_online or not question.strip(),
    )

    if start_btn:
        # Create Run
        payload = {
            "question": question,
            "max_revisions": max_revisions,
            "max_cost_usd": max_cost_usd,
            "pass_threshold": pass_threshold,
        }

        try:
            with st.spinner("Submitting research job to API..."):
                resp = httpx.post(
                    f"{api_base_url}/runs",
                    json=payload,
                    headers=get_headers(),
                    timeout=10.0,
                )
                if resp.status_code != 200:
                    st.error(f"Failed to start run: {resp.text}")
                    st.stop()
                run_data = resp.json()
                run_id = run_data["run_id"]
                st.session_state["active_run_id"] = run_id
        except httpx.ConnectError:
            st.error("Could not reach API server. Ensure the backend is running.")
            st.stop()
        except (httpx.HTTPError, OSError) as e:
            st.error(f"Error starting research: {e}")
            st.stop()

        st.info(f"**Run ID:** `{run_id}`")

        # Live Progress Containers
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        plan_container = st.container()
        workers_container = st.container()
        eval_container = st.container()

        completed_data = None
        plan_subqs = []
        worker_statuses = {}

        # Consume SSE Stream
        try:
            with httpx.stream(
                "GET",
                f"{api_base_url}/runs/{run_id}/events",
                headers=get_headers(),
                timeout=300.0,
            ) as response:
                if response.status_code != 200:
                    st.error(f"SSE connection failed: {response.status_code}")
                    st.stop()

                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            event = json.loads(data_str)
                            etype = event.get("event_type")
                            epayload = event.get("payload", {})

                            if etype == "run_started":
                                progress_bar.progress(0.1)
                                status_text.markdown(
                                    "📋 **Planner** is decomposing the question into sub-questions..."
                                )

                            elif etype == "plan_ready":
                                progress_bar.progress(0.25)
                                plan_subqs = epayload.get("sub_questions", [])
                                status_text.markdown(
                                    f"🔍 **Planner created {len(plan_subqs)} sub-questions**. Fanning out to parallel researchers..."
                                )
                                with plan_container:
                                    st.write("### Research Plan")
                                    for sq in plan_subqs:
                                        st.markdown(
                                            f"- **{sq.get('id')}**: {sq.get('question')}"
                                        )

                            elif etype == "worker_finished":
                                progress_bar.progress(0.55)
                                sq_id = epayload.get("sub_question_id", "worker")
                                count = epayload.get("findings_count", 0)
                                worker_statuses[sq_id] = f"Completed ({count} findings)"
                                with workers_container:
                                    st.write("### Parallel Researcher Workers")
                                    st.dataframe(
                                        [
                                            {"Worker / Sub-Question": k, "Status": v}
                                            for k, v in worker_statuses.items()
                                        ]
                                    )

                            elif etype == "draft_ready":
                                progress_bar.progress(0.75)
                                status_text.markdown(
                                    "✍️ **Synthesizer** generated report draft with inline citations. Verifying citations & running Evaluator..."
                                )

                            elif etype == "revision_started":
                                progress_bar.progress(0.60)
                                rev_num = epayload.get("revision_count", 1)
                                status_text.warning(
                                    f"↺ **Revision loop triggered (#{rev_num})** by Evaluator..."
                                )

                            elif etype == "evaluation_ready":
                                progress_bar.progress(0.90)
                                passed = epayload.get("passed", False)
                                scores = epayload.get("scores", {})
                                with eval_container:
                                    st.write("### Evaluator Scorecard")
                                    st.metric(
                                        "Groundedness (1-5)",
                                        f"{scores.get('groundedness', 0):.1f}",
                                    )
                                    st.metric(
                                        "Coverage (1-5)",
                                        f"{scores.get('coverage', 0):.1f}",
                                    )
                                    st.metric(
                                        "Citation Accuracy (1-5)",
                                        f"{scores.get('citation_accuracy', 0):.1f}",
                                    )
                                    st.metric(
                                        "Clarity (1-5)",
                                        f"{scores.get('clarity', 0):.1f}",
                                    )
                                    if passed:
                                        st.success("✓ Report meets quality threshold!")
                                    else:
                                        st.warning("⚠️ Evaluator requested revision.")

                            elif etype == "budget_warning":
                                st.error(f"🛑 Budget Guard: {epayload.get('message')}")

                            elif etype == "run_completed":
                                progress_bar.progress(1.0)
                                status_text.success(
                                    "🎉 Research Run Completed Successfully!"
                                )
                                completed_data = epayload
                                break

                            elif etype == "run_failed":
                                st.error(f"❌ Run Failed: {epayload.get('error')}")
                                break

                        except json.JSONDecodeError:
                            continue

        except httpx.ConnectError:
            st.error("Lost connection to API server during execution.")
        except (httpx.HTTPError, OSError) as e:
            st.error(f"Error during live streaming: {e}")

        # Render Final Results Tabs if completed
        if completed_data:
            st.markdown("---")
            tabs = st.tabs(
                [
                    "📄 Final Report",
                    "🔗 Sources",
                    "⚖️ Evaluation",
                    "🌳 Trace Spans",
                    "💰 Cost Metrics",
                ]
            )

            with tabs[0]:
                draft_text = completed_data.get("draft", "No report draft.")
                st.markdown(draft_text)
                st.download_button(
                    "Download Report (Markdown)",
                    draft_text,
                    file_name=f"{run_id}_report.md",
                )

            with tabs[1]:
                citations = completed_data.get("citations", [])
                st.write(f"### Cited Sources ({len(citations)})")
                for c in citations:
                    v_icon = (
                        "✓ Verified"
                        if c.get("verified", True)
                        else f"⚠️ Unverified: {c.get('verification_error')}"
                    )
                    st.markdown(
                        f"**[{c.get('index')}] [{c.get('source_title')}]({c.get('source_url')})**"
                    )
                    st.caption(f"Status: {v_icon} | Retrieved: {c.get('retrieved_at')}")
                    if c.get("evidence_quote"):
                        st.caption(f'*Evidence:* "{c.get("evidence_quote")}"')

            with tabs[2]:
                eval_res = completed_data.get("eval_result") or {}
                st.write("### Evaluation Details")
                st.json(eval_res)

            with tabs[3]:
                # Fetch Trace from API
                try:
                    trace_resp = httpx.get(
                        f"{api_base_url}/runs/{run_id}/trace", headers=get_headers()
                    )
                    if trace_resp.status_code == 200:
                        spans_data = trace_resp.json().get("spans", [])
                        st.write(f"### Recorded Spans ({len(spans_data)})")
                        st.dataframe(spans_data)
                except (httpx.HTTPError, OSError):
                    st.info("Trace spans could not be retrieved.")

            with tabs[4]:
                metrics = completed_data.get("metrics") or {}
                st.write("### Cost & Token Metrics")
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric(
                    "Total Cost", f"${metrics.get('total_cost_usd', 0):.4f} USD"
                )
                col_b.metric("Total Tokens", f"{metrics.get('total_tokens', 0):,}")
                col_c.metric(
                    "Latency", f"{metrics.get('total_latency_seconds', 0):.1f}s"
                )
                col_d.metric(
                    "Daily Quota Remaining",
                    f"{metrics.get('remaining_daily_quota', 0)}",
                )

                st.json(metrics)

elif nav_choice == "Run History":
    st.subheader("Recent Research Runs")
    if not api_online:
        st.warning("API backend is offline. Start the backend to view run history.")
    else:
        try:
            resp = httpx.get(f"{api_base_url}/runs", headers=get_headers(), timeout=5.0)
            if resp.status_code == 200:
                runs = resp.json()
                if not runs:
                    st.info("No past research runs found in `./runs/`.")
                else:
                    for r in runs:
                        with st.expander(
                            f"📁 {r.get('run_id')} ({r.get('status')}) - ${r.get('cost_usd', 0):.4f}"
                        ):
                            st.write(f"Created: {r.get('created_at')}")
                            # Inspect run
                            if st.button("Load Details", key=r.get("run_id")):
                                detail_resp = httpx.get(
                                    f"{api_base_url}/runs/{r.get('run_id')}",
                                    headers=get_headers(),
                                )
                                if detail_resp.status_code == 200:
                                    st.json(detail_resp.json())
        except (httpx.HTTPError, OSError) as e:
            st.error(f"Failed to fetch run history: {e}")
