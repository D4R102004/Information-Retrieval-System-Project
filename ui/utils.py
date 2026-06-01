"""Small UI utility helpers."""

from __future__ import annotations

from typing import Any

from .config import DEFAULT_QUERY_MAX_LENGTH, DEFAULT_QUERY_MIN_LENGTH


def validate_query(query: str) -> tuple[bool, str]:
    """Validate a query string and return a status message."""
    cleaned_query = query.strip() if query else ""
    if len(cleaned_query) < DEFAULT_QUERY_MIN_LENGTH:
        return False, f"Query must contain at least {DEFAULT_QUERY_MIN_LENGTH} characters."
    if len(cleaned_query) > DEFAULT_QUERY_MAX_LENGTH:
        return False, f"Query must not exceed {DEFAULT_QUERY_MAX_LENGTH} characters."
    return True, ""


def build_status_message(title: str, value: Any) -> str:
    """Format a concise status line for the UI."""
    return f"**{title}:** {value}"
