import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

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


def validate_configured_models(settings: Settings | None = None) -> list[str]:
    """Validates that configured model names exist in the Google Gemini API for the current key.

    Fails fast with a clear ValueError if a model is unavailable.
    """
    settings = settings or get_settings()
    api_key = settings.google_api_key

    # Skip validation if key is missing, empty, or mock/test key
    if not api_key or api_key.startswith("mock") or api_key == "test_key":
        return []

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        response = httpx.get(url, timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            available = set()
            for m in models:
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    name = m.get("name", "").removeprefix("models/")
                    available.add(name)

            configured = {
                "PLANNER_MODEL": settings.planner_model,
                "RESEARCHER_MODEL": settings.researcher_model,
                "WRITER_MODEL": settings.writer_model,
                "EVALUATOR_MODEL": settings.evaluator_model,
            }

            for env_var, model_name in configured.items():
                clean_name = model_name.removeprefix("models/")
                if clean_name not in available:
                    sample = ", ".join(sorted(available)[:6])
                    raise ValueError(
                        f"Configured model '{model_name}' ({env_var}) is unavailable for your Gemini API key. "
                        f"Available models supporting content generation include: {sample}... "
                        f"Please update {env_var} in your .env file or run scripts/check_models.py."
                    )
            return sorted(available)
        elif response.status_code in (400, 403):
            raise ValueError(
                f"Failed to authenticate with Google Gemini API ({response.status_code}): {response.text}"
            )
    except httpx.RequestError as exc:
        logger.warning(
            f"Could not reach Google Gemini API for model validation (offline/network error): {exc}"
        )
    return []
