"""
Módulo de Ranking y Posicionamiento
=====================================
Combina señales de LSI, vector store y factores adicionales (frescura,
popularidad, tipo de contenido) para rankear y posicionar resultados
en el dominio de Tecnología y Software.
"""

import math
from datetime import datetime, timezone
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Pesos por defecto del ranking
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "semantic_score":  0.55,   # similitud semántica (LSI / coseno)
    "vector_score":    0.25,   # similitud vectorial (embeddings)
    "freshness":       0.10,   # documentos recientes reciben boost
    "popularity":      0.10,   # vistas / votos / interacciones
}


def _freshness_score(date_str: Optional[str]) -> float:
    """
    Devuelve un score 0-1 basado en cuán reciente es el documento.
    Usa decaimiento exponencial con vida media de 180 días.
    """
    if not date_str:
        return 0.5   # neutro si no hay fecha
    try:
        pub_date = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.5
    now = datetime.now(timezone.utc)
    days_old = max(0, (now - pub_date).days)
    half_life = 180  # días
    return math.exp(-math.log(2) * days_old / half_life)


def _popularity_score(doc: Dict) -> float:
    """
    Normaliza la popularidad (vistas, estrellas, votos) al rango 0-1.
    Usa escala logarítmica para evitar sesgos extremos.
    """
    raw = doc.get("popularity", 0) or 0
    if raw <= 0:
        return 0.0
    return min(1.0, math.log1p(raw) / math.log1p(100_000))


def _type_boost(doc: Dict) -> float:
    """
    Pequeño boost por tipo de contenido para diversificación.
    Favorece artículos de fondo sobre listas cortas.
    """
    content_type = (doc.get("type") or doc.get("content_type") or "").lower()
    boosts = {
        "tutorial":     1.05,
        "article":      1.02,
        "documentation":1.04,
        "news":         1.01,
        "video":        1.00,
        "snippet":      0.95,
    }
    return boosts.get(content_type, 1.0)


class RankingEngine:
    """
    Motor de ranking multi-señal para el SRI de Tecnología y Software.

    Combina:
        - Score semántico LSI
        - Score vectorial (embeddings)
        - Frescura temporal
        - Popularidad
        - Boost por tipo de contenido
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_WEIGHTS
        # Normalizar pesos
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def rank(
        self,
        lsi_results: List[Dict],
        vector_results: List[Dict],
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Fusiona y reranquea resultados de LSI y del vector store.

        Args:
            lsi_results:    Resultados de LSIModel.query()
            vector_results: Resultados de VectorStore.query()
            top_k:          Número de resultados finales a devolver.

        Returns:
            Lista fusionada y ordenada de documentos con score final.
        """
        # Construir mapa {doc_id → datos}
        combined: Dict[str, Dict] = {}

        for r in lsi_results:
            did = r["doc_id"]
            combined[did] = {
                **r,
                "lsi_score":    r.get("score", 0.0),
                "vector_score": 0.0,
            }

        for r in vector_results:
            did = r["id"]
            meta = r.get("metadata", {})
            doc = {**meta, **r}
            if did in combined:
                combined[did]["vector_score"] = r.get("score", 0.0)
            else:
                combined[did] = {
                    "doc_id":       did,
                    "title":        meta.get("title", "Sin título"),
                    "url":          meta.get("url", ""),
                    "tags":         meta.get("tags", []),
                    "snippet":      (r.get("document") or "")[:200],
                    "lsi_score":    0.0,
                    "vector_score": r.get("score", 0.0),
                    "type":         meta.get("type", ""),
                    "date":         meta.get("date", ""),
                    "popularity":   meta.get("popularity", 0),
                }

        # Calcular score final para cada documento
        ranked = []
        for doc in combined.values():
            final_score = self._compute_score(doc)
            ranked.append({**doc, "final_score": round(final_score, 4)})

        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        return ranked[:top_k]

    def rank_single_source(
        self,
        results: List[Dict],
        source: str = "lsi",
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Rankea resultados de una única fuente (LSI o vector store).
        Útil cuando solo uno de los módulos devuelve resultados.
        """
        enriched = []
        for r in results:
            if source == "lsi":
                doc = {
                    **r,
                    "lsi_score":    r.get("score", 0.0),
                    "vector_score": 0.0,
                }
            else:
                meta = r.get("metadata", {})
                doc = {
                    "doc_id":       r.get("id", ""),
                    "title":        meta.get("title", "Sin título"),
                    "url":          meta.get("url", ""),
                    "tags":         meta.get("tags", []),
                    "snippet":      (r.get("document") or "")[:200],
                    "lsi_score":    0.0,
                    "vector_score": r.get("score", 0.0),
                    "type":         meta.get("type", ""),
                    "date":         meta.get("date", ""),
                    "popularity":   meta.get("popularity", 0),
                }
            doc["final_score"] = round(self._compute_score(doc), 4)
            enriched.append(doc)

        enriched.sort(key=lambda x: x["final_score"], reverse=True)
        return enriched[:top_k]

    # ------------------------------------------------------------------
    # Cálculo del score final
    # ------------------------------------------------------------------

    def _compute_score(self, doc: Dict) -> float:
        w = self.weights

        semantic  = doc.get("lsi_score",    0.0)
        vector    = doc.get("vector_score", 0.0)
        fresh     = _freshness_score(doc.get("date") or doc.get("published_at"))
        popular   = _popularity_score(doc)
        type_mult = _type_boost(doc)

        base = (
            w["semantic_score"]  * semantic +
            w["vector_score"]    * vector +
            w["freshness"]       * fresh +
            w["popularity"]      * popular
        )
        return base * type_mult

    # ------------------------------------------------------------------
    # Posicionamiento visual
    # ------------------------------------------------------------------

    def assign_positions(self, ranked_docs: List[Dict]) -> List[Dict]:
        """
        Asigna metadatos de posicionamiento visual:
        - position: número de posición (1-based)
        - display_type: featured / standard / compact
        - score_bucket: high / medium / low

        Útil para que la interfaz visual tome decisiones de presentación.
        """
        positioned = []
        for i, doc in enumerate(ranked_docs):
            score = doc.get("final_score", 0.0)
            if i == 0 and score > 0.6:
                display_type = "featured"    # resultado destacado
            elif i < 3 and score > 0.3:
                display_type = "standard"
            else:
                display_type = "compact"

            if score > 0.6:
                bucket = "high"
            elif score > 0.3:
                bucket = "medium"
            else:
                bucket = "low"

            positioned.append({
                **doc,
                "position":     i + 1,
                "display_type": display_type,
                "score_bucket": bucket,
            })
        return positioned
