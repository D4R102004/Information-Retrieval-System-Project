"""Persistent user search history for automatic recommendations.

The history is intentionally small and file-based so the SRI/RAG project can
personalize recommendations without adding a database dependency. Each search
stores the query text, timestamp, and the document ids returned by retrieval.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class UserSearchHistory:
    """Manage a JSON-backed search history for one or more UI users."""

    def __init__(self, history_path: str | Path = "data/user_history.json", max_entries: int = 100) -> None:
        self.history_path = Path(history_path)
        self.max_entries = max(5, int(max_entries))
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def add_search(
        self,
        query: str,
        retrieved_documents: Iterable[dict[str, Any]] | None = None,
        user_id: str = "default",
    ) -> dict[str, Any]:
        """Persist one search event and keep only the most recent entries.

        Args:
            query: Search query submitted by the user.
            retrieved_documents: Documents returned by the search tab.
            user_id: Logical UI/user profile id. Defaults to ``default``.
        """
        query = (query or "").strip()
        if not query:
            return {"success": False, "message": "Empty query was not stored."}

        history = self._load_all()
        entries = history.setdefault(user_id, [])
        doc_ids = self._extract_doc_ids(retrieved_documents or [])
        entry = {
            "query": query,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "retrieved_doc_ids": doc_ids,
        }
        entries.append(entry)
        history[user_id] = entries[-self.max_entries :]
        self._save_all(history)
        return {"success": True, "entry": entry, "total_entries": len(history[user_id])}

    def latest_searches(self, user_id: str = "default", limit: int = 5) -> list[dict[str, Any]]:
        """Return the most recent searches, newest first."""
        limit = max(1, int(limit))
        entries = self._load_all().get(user_id, [])
        return list(reversed(entries[-limit:]))

    def build_profile(self, user_id: str = "default", limit: int = 5) -> dict[str, Any]:
        """Build a recommendation profile from the latest searches.

        The profile uses the latest ``limit`` queries as text and the retrieved
        document ids as seed/liked ids for the content recommender.
        """
        searches = self.latest_searches(user_id=user_id, limit=limit)
        chronological = list(reversed(searches))
        queries = [item.get("query", "") for item in chronological if item.get("query")]
        seed_doc_ids: list[str] = []
        for item in chronological:
            for doc_id in item.get("retrieved_doc_ids", []) or []:
                if doc_id and doc_id not in seed_doc_ids:
                    seed_doc_ids.append(str(doc_id))

        return {
            "query_profile": " ".join(queries),
            "queries": queries,
            "seed_doc_ids": seed_doc_ids,
            "searches_used": searches,
            "history_limit": limit,
        }

    def clear(self, user_id: str = "default") -> dict[str, Any]:
        """Clear one user's stored search history."""
        history = self._load_all()
        removed = len(history.get(user_id, []))
        history[user_id] = []
        self._save_all(history)
        return {"success": True, "removed_entries": removed}

    @staticmethod
    def _extract_doc_ids(documents: Iterable[dict[str, Any]]) -> list[str]:
        doc_ids: list[str] = []
        for doc in documents:
            if not isinstance(doc, dict):
                continue
            doc_id = doc.get("id") or doc.get("doc_id") or doc.get("source")
            if doc_id and str(doc_id) not in doc_ids:
                doc_ids.append(str(doc_id))
        return doc_ids

    def _load_all(self) -> dict[str, list[dict[str, Any]]]:
        if not self.history_path.exists():
            return {}
        try:
            with self.history_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_all(self, history: dict[str, list[dict[str, Any]]]) -> None:
        with self.history_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=2)
