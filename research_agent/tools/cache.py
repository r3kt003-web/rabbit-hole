import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from research_agent.config.settings import get_settings

logger = logging.getLogger(__name__)


class DiskCache:
    """Simple, robust file-based disk cache keyed by SHA256 of the input (URL or query)."""

    def __init__(self, cache_dir: Path | None = None):
        settings = get_settings()
        self.enabled = settings.enable_cache
        self.cache_dir = cache_dir or Path(settings.cache_dir)
        self.serper_cache_dir = self.cache_dir / "serper"
        self.fetch_cache_dir = self.cache_dir / "pages"

        if self.enabled:
            self.serper_cache_dir.mkdir(parents=True, exist_ok=True)
            self.fetch_cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.strip().lower().encode("utf-8")).hexdigest()

    def get_search(self, query: str) -> dict[str, Any] | None:
        """Retrieve cached Serper search response if available."""
        if not self.enabled:
            return None
        file_path = self.serper_cache_dir / f"{self._hash_key(query)}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    logger.debug(f"[CACHE HIT] Serper search query: '{query}'")
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Error reading cache file {file_path}: {e}")
        return None

    def set_search(self, query: str, data: dict[str, Any]) -> None:
        """Save Serper search response to disk cache."""
        if not self.enabled:
            return
        file_path = self.serper_cache_dir / f"{self._hash_key(query)}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except (OSError, TypeError) as e:
            logger.warning(f"Error writing cache file {file_path}: {e}")

    def get_page(self, url: str) -> dict[str, Any] | None:
        """Retrieve cached fetched page if available."""
        if not self.enabled:
            return None
        file_path = self.fetch_cache_dir / f"{self._hash_key(url)}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    logger.debug(f"[CACHE HIT] Fetched page URL: '{url}'")
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Error reading cache file {file_path}: {e}")
        return None

    def set_page(self, url: str, data: dict[str, Any]) -> None:
        """Save fetched page data to disk cache."""
        if not self.enabled:
            return
        file_path = self.fetch_cache_dir / f"{self._hash_key(url)}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except (OSError, TypeError) as e:
            logger.warning(f"Error writing cache file {file_path}: {e}")
