"""
Módulo de Evaluación — SRI Tecnología y Software
==================================================
Implementa las métricas estándar de evaluación de sistemas de recuperación
de información:

  • Precision@k
  • Recall@k
  • F1@k
  • Average Precision (AP) y Mean Average Precision (MAP)
  • NDCG@k  (Normalized Discounted Cumulative Gain)
  • MRR     (Mean Reciprocal Rank)

Referencia:
  Manning, C. D., Raghavan, P., & Schütze, H. (2008).
  Introduction to Information Retrieval. Cambridge University Press.
  Cap. 8: Evaluation in information retrieval.
  https://nlp.stanford.edu/IR-book/
"""

import math
import json
import os
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Tipos usados en evaluación
# ---------------------------------------------------------------------------
# relevant_docs: conjunto de IDs de documentos relevantes para una consulta
# retrieved_docs: lista ordenada de IDs recuperados por el sistema


def precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """
    Precision@k = |Relevantes ∩ Recuperados[:k]| / k
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """
    Recall@k = |Relevantes ∩ Recuperados[:k]| / |Relevantes|
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def f1_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """
    F1@k = 2 * P@k * R@k / (P@k + R@k)
    """
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def average_precision(retrieved: List[str], relevant: set) -> float:
    """
    Average Precision = Σ (P@k * rel(k)) / |Relevantes|
    """
    if not relevant:
        return 0.0
    hits = 0
    ap_sum = 0.0
    for k, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            ap_sum += hits / k
    return ap_sum / len(relevant)


def ndcg_at_k(
    retrieved: List[str],
    relevance_grades: Dict[str, int],   # {doc_id: grade}  grade ∈ {0,1,2,3}
    k: int,
) -> float:
    """
    NDCG@k = DCG@k / IDCG@k

    DCG@k  = Σ_{i=1}^{k}  (2^rel_i - 1) / log2(i + 1)
    IDCG@k = DCG del ranking perfecto
    """
    def dcg(grades: List[int], k: int) -> float:
        return sum(
            (2 ** g - 1) / math.log2(i + 2)
            for i, g in enumerate(grades[:k])
        )

    # Grades del sistema
    sys_grades = [relevance_grades.get(doc_id, 0) for doc_id in retrieved[:k]]

    # Grades del ranking ideal (ordenados descendente)
    ideal_grades = sorted(relevance_grades.values(), reverse=True)

    idcg = dcg(ideal_grades, k)
    if idcg == 0:
        return 0.0
    return dcg(sys_grades, k) / idcg


def reciprocal_rank(retrieved: List[str], relevant: set) -> float:
    """
    Reciprocal Rank = 1 / posición del primer documento relevante.
    Devuelve 0 si ningún documento relevante aparece.
    """
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Evaluador completo
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Evalúa un SRI dado un conjunto de consultas de prueba (test queries)
    con juicios de relevancia manuales.

    Formato del archivo de queries:
    [
      {
        "query_id": "q1",
        "query":    "machine learning frameworks python",
        "relevant": ["doc_42", "doc_17", ...],          # relevancia binaria
        "grades":   {"doc_42": 3, "doc_17": 2, ...}    # opcional, para NDCG
      },
      ...
    ]
    """

    def __init__(self, k_values: Optional[List[int]] = None):
        self.k_values = k_values or [1, 3, 5, 10]
        self.results: List[Dict] = []

    # ------------------------------------------------------------------
    # Evaluación de una consulta
    # ------------------------------------------------------------------

    def evaluate_query(
        self,
        query_id: str,
        retrieved: List[str],
        relevant: List[str],
        grades: Optional[Dict[str, int]] = None,
    ) -> Dict:
        """
        Evalúa una consulta individual.

        Args:
            query_id:  Identificador de la consulta.
            retrieved: Lista ordenada de IDs recuperados (mayor relevancia primero).
            relevant:  Lista de IDs relevantes (ground truth binario).
            grades:    Dict {doc_id: grade} para NDCG graded. Si es None,
                       se generan automáticamente desde relevant con grade=1.

        Returns:
            Dict con todas las métricas para esta consulta.
        """
        rel_set = set(relevant)

        if grades is None:
            grades = {doc_id: 1 for doc_id in relevant}

        metrics = {
            "query_id": query_id,
            "num_relevant": len(rel_set),
            "num_retrieved": len(retrieved),
            "rr": round(reciprocal_rank(retrieved, rel_set), 4),
            "ap": round(average_precision(retrieved, rel_set), 4),
        }

        for k in self.k_values:
            metrics[f"p@{k}"]    = round(precision_at_k(retrieved, rel_set, k), 4)
            metrics[f"r@{k}"]    = round(recall_at_k(retrieved, rel_set, k), 4)
            metrics[f"f1@{k}"]   = round(f1_at_k(retrieved, rel_set, k), 4)
            metrics[f"ndcg@{k}"] = round(ndcg_at_k(retrieved, grades, k), 4)

        self.results.append(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Evaluación de todo el conjunto de test
    # ------------------------------------------------------------------

    def evaluate_all(
        self,
        test_queries: List[Dict],
        retrieval_fn,   # callable(query_text) -> List[str]  (IDs ordenados)
    ) -> Dict:
        """
        Evalúa el sistema sobre todas las consultas de prueba.

        Args:
            test_queries:  Lista de dicts con "query_id", "query", "relevant", "grades".
            retrieval_fn:  Función que recibe texto de consulta y devuelve lista de IDs.

        Returns:
            Dict con métricas agregadas (MAP, MRR, NDCG medio, etc.)
        """
        self.results = []

        for tq in test_queries:
            retrieved_ids = retrieval_fn(tq["query"])
            self.evaluate_query(
                query_id=tq["query_id"],
                retrieved=retrieved_ids,
                relevant=tq.get("relevant", []),
                grades=tq.get("grades"),
            )

        return self.aggregate()

    # ------------------------------------------------------------------
    # Agregación
    # ------------------------------------------------------------------

    def aggregate(self) -> Dict:
        """Calcula métricas medias sobre todos los resultados almacenados."""
        if not self.results:
            return {}

        n = len(self.results)
        agg = {
            "num_queries": n,
            "MAP":  round(sum(r["ap"] for r in self.results) / n, 4),
            "MRR":  round(sum(r["rr"] for r in self.results) / n, 4),
        }

        for k in self.k_values:
            agg[f"mean_P@{k}"]    = round(
                sum(r[f"p@{k}"] for r in self.results) / n, 4)
            agg[f"mean_R@{k}"]    = round(
                sum(r[f"r@{k}"] for r in self.results) / n, 4)
            agg[f"mean_F1@{k}"]   = round(
                sum(r[f"f1@{k}"] for r in self.results) / n, 4)
            agg[f"mean_NDCG@{k}"] = round(
                sum(r[f"ndcg@{k}"] for r in self.results) / n, 4)

        return agg

    # ------------------------------------------------------------------
    # Reporte
    # ------------------------------------------------------------------

    def report(self, output_path: Optional[str] = None) -> str:
        """
        Genera un reporte legible con todas las métricas.
        Si output_path es provisto, guarda el JSON completo.
        """
        agg = self.aggregate()

        lines = [
            "=" * 60,
            "   REPORTE DE EVALUACIÓN — SRI Tecnología y Software",
            "=" * 60,
            f"  Consultas evaluadas : {agg.get('num_queries', 0)}",
            f"  MAP  (Mean Avg Prec): {agg.get('MAP', 0):.4f}",
            f"  MRR  (Mean Recip Rk): {agg.get('MRR', 0):.4f}",
            "",
            "  Por nivel de corte:",
        ]

        for k in self.k_values:
            lines.append(
                f"    k={k:2d}  P={agg.get(f'mean_P@{k}', 0):.4f}  "
                f"R={agg.get(f'mean_R@{k}', 0):.4f}  "
                f"F1={agg.get(f'mean_F1@{k}', 0):.4f}  "
                f"NDCG={agg.get(f'mean_NDCG@{k}', 0):.4f}"
            )

        lines.append("=" * 60)
        report_str = "\n".join(lines)

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "aggregate": agg,
                    "per_query": self.results,
                }, f, ensure_ascii=False, indent=2)
            print(f"[Evaluador] Reporte JSON guardado: {output_path}")

        return report_str

    # ------------------------------------------------------------------
    # Persistencia de queries de prueba
    # ------------------------------------------------------------------

    @staticmethod
    def load_test_queries(path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_test_queries(queries: List[Dict], path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)
        print(f"[Evaluador] Test queries guardadas: {path}")
