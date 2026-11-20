import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env if present
load_dotenv(override=False)


class Settings(BaseModel):
    google_api_key: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    )
    serper_api_key: str = Field(default_factory=lambda: os.getenv("SERPER_API_KEY", ""))

    # Model configuration - default to cheapest stable flash-lite for researcher, flash for planner/evaluator
    planner_model: str = Field(
        default_factory=lambda: os.getenv("PLANNER_MODEL", "gemini-2.5-flash")
    )
    researcher_model: str = Field(
        default_factory=lambda: os.getenv("RESEARCHER_MODEL", "gemini-2.5-flash-lite")
    )
    writer_model: str = Field(
        default_factory=lambda: os.getenv("WRITER_MODEL", "gemini-2.5-flash")
    )
    evaluator_model: str = Field(
        default_factory=lambda: os.getenv("EVALUATOR_MODEL", "gemini-2.5-flash")
    )

    # Free-tier rate limiting defaults
    llm_rpm: int = Field(default_factory=lambda: int(os.getenv("LLM_RPM", "8")))
    llm_rpd: int = Field(default_factory=lambda: int(os.getenv("LLM_RPD", "200")))

    # Cache configuration
    cache_dir: str = Field(default_factory=lambda: os.getenv("CACHE_DIR", ".cache"))
    enable_cache: bool = Field(
        default_factory=lambda: (
            os.getenv("ENABLE_CACHE", "true").lower() in ("true", "1", "yes")
        )
    )
    replay_run_id: str | None = Field(
        default_factory=lambda: os.getenv("REPLAY_RUN_ID") or None
    )

    # Execution limits & thresholds
    max_revisions: int = Field(
        default_factory=lambda: int(os.getenv("MAX_REVISIONS", "2"))
    )
    max_run_cost_usd: float = Field(
        default_factory=lambda: float(os.getenv("MAX_RUN_COST_USD", "2.0"))
    )
    concurrency_limit: int = Field(
        default_factory=lambda: int(os.getenv("CONCURRENCY_LIMIT", "3"))
    )
    serper_price_per_call: float = Field(
        default_factory=lambda: float(os.getenv("SERPER_PRICE_PER_CALL", "0.001"))
    )
    citation_fuzzy_threshold: float = Field(
        default_factory=lambda: float(os.getenv("CITATION_FUZZY_THRESHOLD", "0.7"))
    )
    pass_threshold: float = Field(
        default_factory=lambda: float(os.getenv("PASS_THRESHOLD", "3.5"))
    )

    worker_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("WORKER_TIMEOUT_SECONDS", "30.0"))
    )
    worker_max_retries: int = Field(
        default_factory=lambda: int(os.getenv("WORKER_MAX_RETRIES", "2"))
    )

    api_key: str | None = Field(default_factory=lambda: os.getenv("API_KEY") or None)
    langsmith_tracing: bool = Field(
        default_factory=lambda: (
            os.getenv("LANGSMITH_TRACING", "false").lower() in ("true", "1")
        )
    )
    langsmith_api_key: str | None = Field(
        default_factory=lambda: os.getenv("LANGSMITH_API_KEY") or None
    )

    pricing_file: Path = Field(
        default_factory=lambda: Path(
            os.getenv("PRICING_FILE", str(Path(__file__).parent / "pricing.yaml"))
        )
    )


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


def reload_settings() -> Settings:
    global _settings_instance
    load_dotenv(override=True)
    _settings_instance = Settings()
    return _settings_instance
