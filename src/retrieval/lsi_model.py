"""
LSI — Latent Semantic Indexing
================================
Implementación del Modelo de Semántica Latente (LSI) para recuperación
de información en el dominio de Tecnología y Software.

Referencia bibliográfica:
  Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990).
  Indexing by latent semantic analysis. Journal of the American Society for
  Information Science, 41(6), 391–407. https://doi.org/10.1002/(SICI)1097-4571
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine


class LSIModel:
    """
    Modelo de Indexación Semántica Latente (LSI / LSA).

    Pasos del modelo:
    1. Construir matriz término-documento (TF-IDF).
    2. Aplicar SVD truncado para reducir a k dimensiones latentes.
    3. Proyectar consultas al espacio latente.
    4. Rankear documentos por similitud coseno.
    """

    def __init__(self, n_components: int = 100, language: str = "spanish"):
        """
        Args:
            n_components: Número de dimensiones latentes (k).
            language:     Idioma para stop words.
        """
        self.n_components = n_components
        self.language = language
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self.doc_matrix: Optional[np.ndarray] = None   # (n_docs, k)
        self.documents: List[Dict] = []
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Entrenamiento / indexación
    # ------------------------------------------------------------------

    def fit(self, documents: List[Dict]) -> None:
        """
        Construye el índice LSI a partir de una lista de documentos.

        Args:
            documents: Lista de dicts con al menos {"id", "title", "content"}.
        """
        if not documents:
            raise ValueError("Se requiere al menos un documento para indexar.")

        self.documents = documents
        corpus = [self._get_text(doc) for doc in documents]

        # Paso 1: Matriz TF-IDF  (términos × documentos)
        self.vectorizer = TfidfVectorizer(
            max_features=20_000,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]{2,}\b",
            min_df=2,
        )
        tfidf_matrix = self.vectorizer.fit_transform(corpus)  # (n_docs, vocab)

        # Paso 2: SVD truncado — espacio semántico latente
        k = min(self.n_components, tfidf_matrix.shape[0] - 1,
                tfidf_matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=42)
        doc_latent = self.svd.fit_transform(tfidf_matrix)    # (n_docs, k)

        # Normalización L2 (facilita similitud coseno vía producto punto)
        self.doc_matrix = normalize(doc_latent, norm="l2")
        self.is_fitted = True

        explained = self.svd.explained_variance_ratio_.sum()
        print(f"[LSI] Índice construido: {len(documents)} docs, "
              f"k={k}, varianza explicada={explained:.2%}")

    # ------------------------------------------------------------------
    # Recuperación
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Dict]:
        """
        Recupera los documentos más relevantes para una consulta.

        Args:
            query_text: Texto de la consulta en lenguaje natural.
            top_k:      Número máximo de resultados.
            threshold:  Similitud mínima para incluir un documento.

        Returns:
            Lista de dicts {"doc_id", "title", "score", "snippet"} ordenada
            por relevancia descendente.
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo no ha sido entrenado. Llame a fit() primero.")

        # Proyectar consulta al espacio latente
        q_tfidf = self.vectorizer.transform([query_text])      # (1, vocab)
        q_latent = self.svd.transform(q_tfidf)                # (1, k)
        q_norm = normalize(q_latent, norm="l2")               # (1, k)

        # Similitud coseno con todos los documentos
        scores = self.doc_matrix @ q_norm.T                   # (n_docs, 1)
        scores = scores.ravel()

        # Rankear
        ranked_indices = np.argsort(scores)[::-1]
        results = []
        for idx in ranked_indices[:top_k]:
            sim = float(scores[idx])
            if sim < threshold:
                break
            doc = self.documents[idx]
            results.append({
                "doc_id": doc.get("id", str(idx)),
                "title":  doc.get("title", "Sin título"),
                "score":  round(sim, 4),
                "snippet": self._snippet(doc.get("content", ""), 200),
                "url":    doc.get("url", ""),
                "tags":   doc.get("tags", []),
            })
        return results

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Guarda el modelo en disco."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "lsi_model.pkl"), "wb") as f:
            pickle.dump({
                "vectorizer":   self.vectorizer,
                "svd":          self.svd,
                "doc_matrix":   self.doc_matrix,
                "documents":    self.documents,
                "n_components": self.n_components,
            }, f)
        print(f"[LSI] Modelo guardado en {path}")

    def load(self, path: str) -> None:
        """Carga el modelo desde disco."""
        with open(os.path.join(path, "lsi_model.pkl"), "rb") as f:
            data = pickle.load(f)
        self.vectorizer   = data["vectorizer"]
        self.svd          = data["svd"]
        self.doc_matrix   = data["doc_matrix"]
        self.documents    = data["documents"]
        self.n_components = data["n_components"]
        self.is_fitted    = True
        print(f"[LSI] Modelo cargado: {len(self.documents)} documentos.")

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    @staticmethod
    def _get_text(doc: Dict) -> str:
        """Concatena título + contenido para indexación."""
        title   = doc.get("title", "") or ""
        content = doc.get("content", "") or ""
        tags    = " ".join(doc.get("tags", [])) if doc.get("tags") else ""
        return f"{title} {title} {content} {tags}"   # doble título = mayor peso

    @staticmethod
    def _snippet(text: str, length: int = 200) -> str:
        """Extrae un fragmento del texto."""
        text = text.strip()
        if len(text) <= length:
            return text
        return text[:length].rsplit(" ", 1)[0] + "…"
