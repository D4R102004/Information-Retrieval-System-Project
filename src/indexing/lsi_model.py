from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import json
import os
import pickle

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer


@dataclass
class SearchResult:
    doc_id: str
    title: str
    score: float
    content: str


class LSIRetriever:
    def __init__(
        self,
        n_components: int = 2,
        max_features: int = 5000,
        stop_words: str | None = "english",
    ) -> None:
        self.n_components = n_components
        self.max_features = max_features
        self.stop_words = stop_words

        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.normalizer: Normalizer | None = None

        self.documents: List[Dict[str, Any]] = []
        self.doc_vectors: np.ndarray | None = None

    def fit(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            raise ValueError("La lista de documentos está vacía.")

        self.documents = documents
        corpus = [
            f"{doc.get('title', '')} {doc.get('content', '')}".strip()
            for doc in documents
        ]

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=self.stop_words,
            lowercase=True
        )

        tfidf_matrix = self.vectorizer.fit_transform(corpus)

        max_allowed_components = min(tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        if max_allowed_components < 1:
            raise ValueError("No hay suficientes datos para construir el modelo LSI.")

        n_components = min(self.n_components, max_allowed_components)

        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_matrix = self.svd.fit_transform(tfidf_matrix)

        self.normalizer = Normalizer(copy=False)
        self.doc_vectors = self.normalizer.fit_transform(reduced_matrix)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if not query.strip():
            raise ValueError("La consulta no puede estar vacía.")

        if (
            self.vectorizer is None
            or self.svd is None
            or self.normalizer is None
            or self.doc_vectors is None
        ):
            raise RuntimeError("El modelo LSI no ha sido entrenado todavía.")

        query_tfidf = self.vectorizer.transform([query])
        query_lsi = self.svd.transform(query_tfidf)
        query_lsi = self.normalizer.transform(query_lsi)

        similarities = np.dot(self.doc_vectors, query_lsi.T).ravel()
        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        results: List[SearchResult] = []
        for idx in ranked_indices:
            doc = self.documents[idx]
            results.append(
                SearchResult(
                    doc_id=str(doc.get("id", "")),
                    title=str(doc.get("title", "")),
                    score=float(similarities[idx]),
                    content=str(doc.get("content", "")),
                )
            )

        return results

    def save(self, output_dir: str) -> None:
        if (
            self.vectorizer is None
            or self.svd is None
            or self.normalizer is None
            or self.doc_vectors is None
        ):
            raise RuntimeError("No se puede guardar un modelo no entrenado.")

        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "lsi_model.pkl"), "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "svd": self.svd,
                    "normalizer": self.normalizer,
                    "documents": self.documents,
                    "doc_vectors": self.doc_vectors,
                    "n_components": self.n_components,
                    "max_features": self.max_features,
                    "stop_words": self.stop_words,
                },
                f,
            )

    @classmethod
    def load(cls, model_dir: str) -> "LSIRetriever":
        model_path = os.path.join(model_dir, "lsi_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No existe el modelo en: {model_path}")

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        instance = cls(
            n_components=data["n_components"],
            max_features=data["max_features"],
            stop_words=data["stop_words"],
        )
        instance.vectorizer = data["vectorizer"]
        instance.svd = data["svd"]
        instance.normalizer = data["normalizer"]
        instance.documents = data["documents"]
        instance.doc_vectors = data["doc_vectors"]

        return instance


def load_documents_from_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)