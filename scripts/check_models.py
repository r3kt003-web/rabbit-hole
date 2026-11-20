"""Script to check available Gemini models for the configured API key

and report whether configured models exist.
"""

import os
import sys
from pathlib import Path

import dotenv
import httpx

# Load .env file
dotenv.load_dotenv()

# Add project root to sys.path so research_agent can be imported
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from research_agent.config.settings import get_settings


def fetch_available_models(api_key: str) -> list[str]:
    """Queries the Google Generative Language API for models supporting generateContent."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        response = httpx.get(url, timeout=15.0)
        if response.status_code != 200:
            print(
                f"[ERROR] Failed to query models API (Status {response.status_code}): {response.text}"
            )
            return []
        data = response.json()
        models = data.get("models", [])
        available = []
        for m in models:
            methods = m.get("supportedGenerationMethods", [])
            if "generateContent" in methods:
                name = m.get("name", "").removeprefix("models/")
                available.append(name)
        return sorted(available)
    except (httpx.HTTPError, OSError, ValueError) as exc:
        print(f"[ERROR] Exception while fetching models: {exc}")
        return []


def check_models():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] Neither GOOGLE_API_KEY nor GEMINI_API_KEY found in environment or .env file.")
        sys.exit(1)

    print(f"Checking Gemini models for API key ({api_key[:6]}...{api_key[-4:]})...\n")
    available_models = fetch_available_models(api_key)

    if not available_models:
        print("[ERROR] No models found or failed to authenticate with Google Gemini API.")
        sys.exit(1)

    print("=== Available Gemini Models (supporting generateContent) ===")
    for m in available_models:
        print(f"  • {m}")
    print(f"\nTotal available models: {len(available_models)}\n")

    settings = get_settings()
    configured = {
        "PLANNER_MODEL": settings.planner_model,
        "RESEARCHER_MODEL": settings.researcher_model,
        "WRITER_MODEL": settings.writer_model,
        "EVALUATOR_MODEL": settings.evaluator_model,
    }

    print("=== Configured Model Status ===")
    all_found = True
    for env_var, model_name in configured.items():
        clean_name = model_name.removeprefix("models/")
        if clean_name in available_models:
            print(f"  [FOUND]     {env_var}: '{model_name}'")
        else:
            print(f"  [NOT FOUND] {env_var}: '{model_name}' (Unavailable for this API key!)")
            all_found = False

    print("\nSummary:")
    if all_found:
        print("✓ All configured models are valid and available.")
    else:
        print("✗ One or more configured models were NOT found. Please update .env.")
        sys.exit(1)


if __name__ == "__main__":
    check_models()
