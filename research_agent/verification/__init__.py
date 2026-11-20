"""Verification module for citations and factual grounding."""

from .citations import (
    CitationVerifier,
    VerificationResult,
    fuzzy_substring_match,
    verify_citations_node,
)

__all__ = [
    "CitationVerifier",
    "VerificationResult",
    "fuzzy_substring_match",
    "verify_citations_node",
]
