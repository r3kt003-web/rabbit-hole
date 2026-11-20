from unittest.mock import patch

from fastapi.testclient import TestClient

from research_agent.config.settings import reload_settings
from research_agent.service.api import app
from research_agent.service.runner import Event

client = TestClient(app)


async def _mock_run_research(*args, **kwargs):
    yield Event(
        run_id="run_test", event_type="run_started", payload={"question": "mock"}
    )
    yield Event(
        run_id="run_test", event_type="plan_ready", payload={"sub_questions": []}
    )
    yield Event(
        run_id="run_test", event_type="run_completed", payload={"draft": "Mock report."}
    )


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_create_run_and_status():
    with patch(
        "research_agent.service.api.run_research", side_effect=_mock_run_research
    ):
        resp = client.post("/runs", json={"question": "What is perovskite?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert data["status"] == "started"

        # Status of the newly created run
        run_id = data["run_id"]
        status_resp = client.get(f"/runs/{run_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["run_id"] == run_id


def test_api_auth_rejects_bad_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret_passphrase_123")
    reload_settings()

    try:
        with patch(
            "research_agent.service.api.run_research", side_effect=_mock_run_research
        ):
            # Request without key
            resp_no_key = client.post("/runs", json={"question": "Auth test"})
            assert resp_no_key.status_code == 401

            # Request with bad key
            resp_bad_key = client.post(
                "/runs",
                json={"question": "Auth test"},
                headers={"X-API-KEY": "wrong_key"},
            )
            assert resp_bad_key.status_code == 401

            # Request with valid key
            resp_valid = client.post(
                "/runs",
                json={"question": "Auth test"},
                headers={"X-API-KEY": "secret_passphrase_123"},
            )
            assert resp_valid.status_code == 200
    finally:
        monkeypatch.delenv("API_KEY", raising=False)
        reload_settings()


def test_cancel_endpoint():
    with patch(
        "research_agent.service.api.run_research", side_effect=_mock_run_research
    ):
        resp = client.post("/runs", json={"question": "Cancellation test"})
        run_id = resp.json()["run_id"]

        cancel_resp = client.post(f"/runs/{run_id}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["run_id"] == run_id


def test_sse_events_stream_with_mocked_runner():
    with patch(
        "research_agent.service.api.run_research", side_effect=_mock_run_research
    ):
        resp = client.post("/runs", json={"question": "SSE test"})
        run_id = resp.json()["run_id"]

        with client.stream("GET", f"/runs/{run_id}/events") as stream:
            lines = list(stream.iter_lines())
            text = "\n".join(lines)
            assert "event: run_started" in text or "id:" in text
