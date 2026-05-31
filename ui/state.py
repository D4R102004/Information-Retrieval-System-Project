"""Session state helpers for the frontend."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from .config import (
    DEFAULT_AUTO_RELOAD,
    DEFAULT_ENABLE_WEB_SEARCH,
    DEFAULT_MAX_LOCAL_RESULTS,
    DEFAULT_MAX_WEB_RESULTS,
)


@dataclass
class UIState:
    """Store the user-facing UI state for a single Gradio session."""

    last_query: str = ""
    retrieved_documents: list[dict[str, Any]] = field(default_factory=list)
    rag_response: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(
        default_factory=lambda: {
            "use_web_search": DEFAULT_ENABLE_WEB_SEARCH,
            "auto_reload_empty": DEFAULT_AUTO_RELOAD,
            "max_local_results": DEFAULT_MAX_LOCAL_RESULTS,
            "max_web_results": DEFAULT_MAX_WEB_RESULTS,
        }
    )


def create_default_state() -> UIState:
    """Create a fresh UI state instance with default settings."""
    return UIState()


def state_to_dict(state: UIState) -> dict[str, Any]:
    """Convert a UI state object to a plain dictionary."""
    return asdict(state)
