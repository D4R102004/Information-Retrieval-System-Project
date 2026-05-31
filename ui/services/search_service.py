"""Search-related helpers for the frontend scaffold."""

from __future__ import annotations

from typing import Any


def build_result_payload(documents: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    """Shape retrieval output for the UI layer."""
    return {"documents": documents, "metadata": metadata}
