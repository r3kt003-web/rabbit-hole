import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from research_agent.config.settings import get_settings
from research_agent.llm.rate_limiter import call_with_rate_limit

logger = logging.getLogger(__name__)


class LLMDiskCache:
    """Disk cache for LLM invocations to support zero-cost re-runs and replay mode."""

    def __init__(self, cache_dir: Path | None = None):
        settings = get_settings()
        self.enabled = settings.enable_cache
        self.cache_dir = cache_dir or (Path(settings.cache_dir) / "llm")
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _compute_key(model_name: str, messages: list[Any], extra: str = "") -> str:
        serialized = []
        for m in messages:
            if hasattr(m, "content"):
                serialized.append(f"{m.__class__.__name__}:{m.content}")
            else:
                serialized.append(str(m))
        raw = f"{model_name}|{extra}|{'|'.join(serialized)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        file_path = self.cache_dir / f"{key}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    logger.debug(f"[CACHE HIT] LLM response for key: {key}")
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Error reading LLM cache {file_path}: {e}")
        return None

    def set(self, key: str, data: dict[str, Any]) -> None:
        if not self.enabled:
            return
        file_path = self.cache_dir / f"{key}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except (OSError, TypeError) as e:
            logger.warning(f"Error writing LLM cache {file_path}: {e}")


_llm_cache = LLMDiskCache()


def get_chat_model(model_name: str, temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Instantiate a ChatGoogleGenerativeAI client."""
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key,
        temperature=temperature,
    )


async def invoke_llm_with_cache(
    model: ChatGoogleGenerativeAI,
    messages: list[Any],
    extra_cache_key: str = "",
    replay_data: dict[str, Any] | None = None,
) -> AIMessage:
    """Invokes LLM through global rate limiter with disk caching and replay support."""
    key = LLMDiskCache._compute_key(model.model, messages, extra_cache_key)

    # 1. Check replay data first if in replay mode
    if replay_data and key in replay_data:
        logger.info(
            f"[REPLAY] Serving LLM response from replay data (key: {key[:8]}...)"
        )
        entry = replay_data[key]
        return AIMessage(
            content=entry.get("content", ""),
            response_metadata=entry.get("metadata", {}),
            usage_metadata=entry.get("usage", {}),
        )

    # 2. Check disk cache
    cached = _llm_cache.get(key)
    if cached is not None:
        return AIMessage(
            content=cached.get("content", ""),
            response_metadata=cached.get("metadata", {}),
            usage_metadata=cached.get("usage", {}),
        )

    # 3. Call LLM with rate limiting and exponential backoff
    response = await call_with_rate_limit(model.ainvoke, messages)

    # 4. Save to cache
    usage = getattr(response, "usage_metadata", {}) or {}
    metadata = getattr(response, "response_metadata", {}) or {}
    _llm_cache.set(
        key,
        {
            "content": response.content,
            "usage": usage,
            "metadata": metadata,
        },
    )

    return response
