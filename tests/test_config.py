from research_agent.config.settings import Settings, get_settings


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
