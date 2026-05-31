"""Adapter layer for MainOrchestator integration."""

from __future__ import annotations

from typing import Any


class OrchestratorService:
    """Placeholder adapter for backend orchestration calls."""

    def retrieve_documents(self, question: str, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def augment_response(self, question: str, documents: list[dict[str, Any]]) -> Any:
        raise NotImplementedError

    def clear_all_indices(self) -> dict[str, Any]:
        raise NotImplementedError

    def load_documents_from_crawlers(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def evaluate_test(self, test_spec: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError
