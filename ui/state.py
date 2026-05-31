"""Session state helpers for the frontend scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UIState:
    """State container for the frontend session."""

    last_query: str = ""
    retrieved_documents: list[dict[str, Any]] = field(default_factory=list)
    rag_response: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
