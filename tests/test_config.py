from unittest.mock import MagicMock, patch

import pytest

from research_agent.config.settings import (
    Settings,
    get_settings,
    validate_configured_models,
)


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.delenv("PLANNER_MODEL", raising=False)
    monkeypatch.delenv("RESEARCHER_MODEL", raising=False)
    monkeypatch.delenv("MAX_REVISIONS", raising=False)
    monkeypatch.delenv("CONCURRENCY_LIMIT", raising=False)
    monkeypatch.delenv("WORKER_TIMEOUT_SECONDS", raising=False)

    settings = Settings()
    assert settings.planner_model == "gemini-2.5-flash"
    assert settings.researcher_model == "gemini-2.5-flash-lite"
    assert settings.max_revisions == 2
    assert settings.max_run_cost_usd == 2.0
    assert settings.concurrency_limit == 3
    assert settings.pass_threshold == 3.5


def test_settings_env_overrides(monkeypatch):
    monkeypatch.setenv("PLANNER_MODEL", "gemini-1.5-pro")
    monkeypatch.setenv("MAX_REVISIONS", "4")
    monkeypatch.setenv("MAX_RUN_COST_USD", "5.5")
    monkeypatch.setenv("SERPER_PRICE_PER_CALL", "0.002")

    settings = Settings()
    assert settings.planner_model == "gemini-1.5-pro"
    assert settings.max_revisions == 4
    assert settings.max_run_cost_usd == 5.5
    assert settings.serper_price_per_call == 0.002


def test_pricing_file_exists():
    settings = get_settings()
    assert settings.pricing_file.exists(), (
        f"Pricing file does not exist at {settings.pricing_file}"
    )


def test_validate_configured_models_skips_when_empty_key():
    settings = Settings(google_api_key="")
    assert validate_configured_models(settings) == []


def test_validate_configured_models_success():
    settings = Settings(
        google_api_key="valid_test_api_key",
        planner_model="gemini-2.5-flash",
        researcher_model="gemini-2.5-flash-lite",
        writer_model="gemini-2.5-flash",
        evaluator_model="gemini-2.5-flash",
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [
            {
                "name": "models/gemini-2.5-flash",
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "models/gemini-2.5-flash-lite",
                "supportedGenerationMethods": ["generateContent"],
            },
        ]
    }
    with patch("httpx.get", return_value=mock_resp):
        res = validate_configured_models(settings)
        assert "gemini-2.5-flash" in res
        assert "gemini-2.5-flash-lite" in res


def test_validate_configured_models_raises_on_missing_model():
    settings = Settings(
        google_api_key="valid_test_api_key",
        planner_model="gemini-2.0-flash",  # deprecated/missing model
        researcher_model="gemini-2.5-flash-lite",
        writer_model="gemini-2.5-flash",
        evaluator_model="gemini-2.5-flash",
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [
            {
                "name": "models/gemini-2.5-flash",
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "models/gemini-2.5-flash-lite",
                "supportedGenerationMethods": ["generateContent"],
            },
        ]
    }
    with patch("httpx.get", return_value=mock_resp), pytest.raises(
        ValueError, match="is unavailable for your Gemini API key"
    ):
        validate_configured_models(settings)
