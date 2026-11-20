import re
from pathlib import Path


def test_streamlit_module_has_no_backend_imports():
    """REQUIRED TEST: Grep-style assertion that ui/app.py has NO imports
    from graph/, tools/, or llm/. The UI must communicate strictly over HTTP.
    """
    app_path = Path("research_agent/ui/app.py")
    assert app_path.exists(), f"{app_path} does not exist"

    code = app_path.read_text(encoding="utf-8")
    lines = code.splitlines()

    forbidden_patterns = [
        r"^\s*from\s+research_agent\.graph",
        r"^\s*import\s+research_agent\.graph",
        r"^\s*from\s+research_agent\.tools",
        r"^\s*import\s+research_agent\.tools",
        r"^\s*from\s+research_agent\.llm",
        r"^\s*import\s+research_agent\.llm",
        r"^\s*from\s+research_agent\.verification",
        r"^\s*import\s+research_agent\.verification",
        r"^\s*import\s+langgraph",
        r"^\s*from\s+langgraph",
        r"^\s*import\s+google\.genai",
        r"^\s*from\s+langchain_google_genai",
    ]

    violations = []
    for idx, line in enumerate(lines, start=1):
        for pattern in forbidden_patterns:
            if re.search(pattern, line):
                violations.append(f"Line {idx}: {line.strip()}")

    assert not violations, (
        "Streamlit ui/app.py violates architectural boundary by importing backend modules directly!\n"
        "Violations found:\n" + "\n".join(violations)
    )
