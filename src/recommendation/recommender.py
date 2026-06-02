"""Content-based recommendation module.

This module adds an optional recommender to the SRI/RAG system. It recommends
technology/software documents using document content, tags, user interests,
previously liked documents, and a small freshness signal.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RecommendationConfig:
    """Weights used by the recommender."""

    content_weight: float = 0.82
    recency_weight: float = 0.12
    source_weight: float = 0.06
    max_features: int = 50_000


class ContentBasedRecommender:
    """Recommend documents from the local corpus.

    The recommender is intentionally independent from Gradio and from the RAG
    module. It can be used by the orchestrator, the CLI, tests, or future APIs.
    """

    def __init__(
        self,
        documents_path: str | Path = "data/documents.json",
        config: RecommendationConfig | None = None,
    ) -> None:
        self.documents_path = Path(documents_path)
        self.config = config or RecommendationConfig()
        self.documents: list[dict[str, Any]] = []
        self.doc_by_id: dict[str, dict[str, Any]] = {}
        self.doc_ids: list[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self._source_scores: dict[str, float] = {}
        self.load_documents()

    # ------------------------------------------------------------------
    # Loading and feature construction
    # ------------------------------------------------------------------

    def load_documents(self, documents: list[dict[str, Any]] | None = None) -> None:
        """Load documents and build the TF-IDF matrix.

        Args:
            documents: Optional in-memory corpus. When omitted, the recommender
                loads ``self.documents_path``.
        """
        if documents is None:
            documents = self._load_from_disk()

        self.documents = [doc for doc in documents if self._valid_document(doc)]
        self.doc_by_id = {str(doc["id"]): doc for doc in self.documents}
        self.doc_ids = list(self.doc_by_id.keys())
        self._source_scores = self._build_source_scores(self.documents)

        texts = [self._document_text(doc) for doc in self.documents]
        if not texts:
            self.vectorizer = None
            self.matrix = None
            return

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.config.max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def _load_from_disk(self) -> list[dict[str, Any]]:
        if not self.documents_path.exists():
            return []
        with self.documents_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []

    @staticmethod
    def _valid_document(doc: Any) -> bool:
        return isinstance(doc, dict) and bool(doc.get("id"))

    @staticmethod
    def _clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return " ".join(str(item) for item in value)
        return str(value)

    def _document_text(self, doc: dict[str, Any]) -> str:
        # Tags and titles are repeated to give them stronger influence than the
        # long article body.
        title = self._clean(doc.get("title"))
        tags = self._clean(doc.get("tags"))
        source = self._clean(doc.get("source"))
        content = self._clean(doc.get("content") or doc.get("snippet"))
        return f"{title} {title} {tags} {tags} {source} {content}"

    # ------------------------------------------------------------------
    # Public recommendation API
    # ------------------------------------------------------------------

    def recommend(
        self,
        query: str | None = None,
        interests: str | Iterable[str] | None = None,
        liked_doc_ids: Iterable[str] | None = None,
        exclude_doc_ids: Iterable[str] | None = None,
        top_k: int = 10,
        content_weight: float | None = None,
        recency_weight: float | None = None,
        source_weight: float | None = None,
    ) -> dict[str, Any]:
        """Return personalized content-based recommendations.

        Args:
            query: Current search or information need.
            interests: Free-text profile, tags, or keywords describing the user.
            liked_doc_ids: Documents the user liked, opened, or selected.
            exclude_doc_ids: Documents to remove from the recommendation list.
            top_k: Number of recommendations.
            content_weight: Optional override for semantic/content similarity.
            recency_weight: Optional override for freshness.
            source_weight: Optional override for source diversity/quality prior.
        """
        if not self.documents or self.vectorizer is None or self.matrix is None:
            return {
                "status": "error",
                "message": "No documents available for recommendation.",
                "recommendations": [],
                "metadata": {"total_documents": 0},
            }

        liked_ids = [str(doc_id).strip() for doc_id in liked_doc_ids or [] if str(doc_id).strip()]
        excluded = {str(doc_id).strip() for doc_id in exclude_doc_ids or [] if str(doc_id).strip()}
        excluded.update(liked_ids)

        profile_text = self._profile_text(query=query, interests=interests)
        similarity_scores = self._content_scores(profile_text, liked_ids)

        cw = self.config.content_weight if content_weight is None else float(content_weight)
        rw = self.config.recency_weight if recency_weight is None else float(recency_weight)
        sw = self.config.source_weight if source_weight is None else float(source_weight)
        total = cw + rw + sw
        if total <= 0:
            cw, rw, sw = self.config.content_weight, self.config.recency_weight, self.config.source_weight
            total = cw + rw + sw
        cw, rw, sw = cw / total, rw / total, sw / total

        candidates: list[dict[str, Any]] = []
        for idx, doc in enumerate(self.documents):
            doc_id = str(doc.get("id"))
            if doc_id in excluded:
                continue
            similarity = float(similarity_scores[idx])
            recency = self._recency_score(doc)
            source = self._source_scores.get(self._clean(doc.get("source")), 0.5)
            final_score = (cw * similarity) + (rw * recency) + (sw * source)
            candidate = self._result_document(doc)
            candidate.update(
                {
                    "recommendation_score": round(final_score, 6),
                    "similarity_score": round(similarity, 6),
                    "recency_score": round(recency, 6),
                    "source_score": round(source, 6),
                    "explanation": self._explain(doc, similarity, recency, profile_text, liked_ids),
                }
            )
            candidates.append(candidate)

        ranked = sorted(candidates, key=lambda item: item["recommendation_score"], reverse=True)[: max(1, int(top_k))]
        return {
            "status": "success",
            "message": f"Generated {len(ranked)} recommendations.",
            "recommendations": ranked,
            "metadata": {
                "total_documents": len(self.documents),
                "candidate_documents": len(candidates),
                "liked_doc_ids_used": [doc_id for doc_id in liked_ids if doc_id in self.doc_by_id],
                "query_used": bool(query and query.strip()),
                "interests_used": bool(self._clean(interests).strip()),
                "weights": {"content": cw, "recency": rw, "source": sw},
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    def similar_to_document(
        self,
        document_id: str,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Recommend documents similar to a specific document."""
        document_id = str(document_id).strip()
        if document_id not in self.doc_by_id:
            return {
                "status": "error",
                "message": f"Document id not found: {document_id}",
                "recommendations": [],
                "metadata": {"document_id": document_id},
            }
        return self.recommend(
            liked_doc_ids=[document_id],
            exclude_doc_ids=[document_id],
            top_k=top_k,
            content_weight=0.9,
            recency_weight=0.07,
            source_weight=0.03,
        )

    def recommend_from_search_results(
        self,
        query: str,
        retrieved_documents: list[dict[str, Any]],
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Recommend extra documents using the current query and retrieved docs."""
        liked_ids = [str(doc.get("id") or doc.get("doc_id")) for doc in retrieved_documents if doc.get("id") or doc.get("doc_id")]
        return self.recommend(
            query=query,
            liked_doc_ids=liked_ids,
            exclude_doc_ids=liked_ids,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _profile_text(self, query: str | None, interests: str | Iterable[str] | None) -> str:
        interests_text = self._clean(interests)
        query_text = self._clean(query)
        return f"{query_text} {interests_text}".strip()

    def _content_scores(self, profile_text: str, liked_doc_ids: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []

        if profile_text:
            vectors.append(self.vectorizer.transform([profile_text]).toarray())

        valid_liked_indices = [self.doc_ids.index(doc_id) for doc_id in liked_doc_ids if doc_id in self.doc_by_id]
        if valid_liked_indices:
            liked_centroid = np.asarray(self.matrix[valid_liked_indices].mean(axis=0))
            vectors.append(liked_centroid)

        if not vectors:
            # Cold start: no profile is available. Ranking then depends on freshness
            # and source prior, but content score remains neutral.
            return np.zeros(len(self.documents), dtype=float)

        # Average all available signals: text profile and liked-document centroid.
        centroid = np.mean(np.vstack(vectors), axis=0, keepdims=True)
        scores = cosine_similarity(centroid, self.matrix).ravel()
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    def _recency_score(self, doc: dict[str, Any]) -> float:
        raw_date = self._clean(doc.get("date") or doc.get("published") or doc.get("created_at"))
        if not raw_date:
            return 0.35
        try:
            normalized = raw_date.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            days = max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds() / 86_400)
            return float(math.exp(-days / 180.0))
        except Exception:
            return 0.35

    @staticmethod
    def _build_source_scores(documents: list[dict[str, Any]]) -> dict[str, float]:
        counts: dict[str, int] = {}
        for doc in documents:
            source = str(doc.get("source") or "unknown")
            counts[source] = counts.get(source, 0) + 1
        if not counts:
            return {}
        max_count = max(counts.values())
        # Smooth source prior: sources with more content get a slightly higher
        # prior, but it is capped so recommendations remain content-driven.
        return {source: 0.4 + 0.6 * (count / max_count) for source, count in counts.items()}

    def _result_document(self, doc: dict[str, Any]) -> dict[str, Any]:
        content = self._clean(doc.get("content") or doc.get("snippet"))
        return {
            "id": str(doc.get("id", "")),
            "title": self._clean(doc.get("title")) or "Untitled document",
            "url": self._clean(doc.get("url")),
            "source": self._clean(doc.get("source")),
            "date": self._clean(doc.get("date")),
            "tags": doc.get("tags", ""),
            "snippet": content[:500] + ("..." if len(content) > 500 else ""),
        }

    def _explain(
        self,
        doc: dict[str, Any],
        similarity: float,
        recency: float,
        profile_text: str,
        liked_doc_ids: list[str],
    ) -> str:
        reasons: list[str] = []
        tags = self._clean(doc.get("tags"))
        if similarity > 0.08 and profile_text:
            reasons.append("matches the provided interests/query")
        if similarity > 0.08 and liked_doc_ids:
            reasons.append("is similar to selected documents")
        if tags:
            reasons.append(f"shares topic tags: {tags[:120]}")
        if recency > 0.65:
            reasons.append("is relatively recent")
        if not reasons:
            reasons.append("has a balanced content, source, and freshness score")
        return "; ".join(reasons)
