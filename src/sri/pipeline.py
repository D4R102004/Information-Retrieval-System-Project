"""
Pipeline Principal — SRI Tecnología y Software
================================================
Integra todos los módulos:
  1. InvertedIndex  — indexación
  2. LSIModel       — recuperación semántica latente
  3. VectorStore    — base de datos vectorial
  4. RankingEngine  — posicionamiento multi-señal
  5. Evaluator      — evaluación con P, R, NDCG, MRR

Uso rápido:
    pipeline = SRIPipeline()
    pipeline.index(documents)
    results = pipeline.search("machine learning frameworks 2024")
"""

import os
import json
from typing import List, Dict, Optional

from ..indexing.indexer import InvertedIndex
from ..retrieval.lsi_model import LSIModel
from ..retrieval.vector_store import VectorStore
from ..ranking.ranking import RankingEngine
from ..evaluation.evaluation import Evaluator


DATA_DIR  = "data"
INDEX_DIR = os.path.join(DATA_DIR, "index")
MODEL_DIR = os.path.join(DATA_DIR, "models")


class SRIPipeline:
    """
    Orquestador del sistema de recuperación de información.
    """

    def __init__(
        self,
        lsi_components: int = 100,
        top_k: int = 10,
        load_existing: bool = True,
    ):
        self.top_k = top_k

        # Módulos
        self.indexer  = InvertedIndex(use_stemming=True)
        self.lsi      = LSIModel(n_components=lsi_components)
        self.vstore   = VectorStore(
            collection_name="tech_software",
            persist_dir=INDEX_DIR,
        )
        self.ranker   = RankingEngine()
        self.evaluator = Evaluator(k_values=[1, 3, 5, 10])

        if load_existing:
            self._load_all()

    # ------------------------------------------------------------------
    # Indexación
    # ------------------------------------------------------------------

    def index(self, documents: List[Dict], save: bool = True) -> None:
        """
        Indexa una lista de documentos en todos los módulos.

        Args:
            documents: Lista de dicts {"id","title","content","url","tags",...}
            save:      Si True, persiste todo a disco.
        """
        print(f"\n[Pipeline] Indexando {len(documents)} documentos...")

        # 1. Índice invertido
        self.indexer.build(documents)

        # 2. LSI
        self.lsi.fit(documents)

        # 3. Vector store
        self.vstore.add(documents)

        if save:
            self._save_all()

        # Estadísticas
        stats = self.indexer.stats()
        print(f"[Pipeline] Vocabulario: {stats['vocab_size']} términos | "
              f"Longitud media doc: {stats['avg_doc_len']:.0f} tokens")

    def add_document(self, doc: Dict) -> None:
        """Agrega un documento al sistema sin reentrenar LSI."""
        self.indexer.add_document(doc)
        self.vstore.add([doc])
        # LSI requiere refitting completo si cambia el corpus
        print(f"[Pipeline] Doc '{doc.get('id')}' agregado al índice y vector store.")

    # ------------------------------------------------------------------
    # Búsqueda
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_lsi: bool = True,
        use_vector: bool = True,
    ) -> List[Dict]:
        """
        Recupera y rankea documentos para una consulta.

        Returns:
            Lista de documentos posicionados con score final.
        """
        k = top_k or self.top_k
        lsi_results, vector_results = [], []

        if use_lsi and self.lsi.is_fitted:
            lsi_results = self.lsi.query(query, top_k=k * 2)

        if use_vector and self.vstore.count() > 0:
            vector_results = self.vstore.query(query, n_results=k * 2)

        if not lsi_results and not vector_results:
            # Fallback: búsqueda TF-IDF sobre índice invertido
            scores = self.indexer.get_tfidf_scores(query)
            sorted_ids = sorted(scores, key=scores.get, reverse=True)[: k * 2]
            lsi_results = [
                {
                    "doc_id":  did,
                    "score":   scores[did] / 10,  # normalizar
                    "title":   self.indexer.doc_metadata.get(did, {}).get("title", ""),
                    "snippet": "",
                    "url":     self.indexer.doc_metadata.get(did, {}).get("url", ""),
                    "tags":    [],
                }
                for did in sorted_ids
            ]

        # Rankeo multi-señal
        if lsi_results and vector_results:
            ranked = self.ranker.rank(lsi_results, vector_results, top_k=k)
        elif lsi_results:
            ranked = self.ranker.rank_single_source(lsi_results, "lsi", k)
        elif vector_results:
            ranked = self.ranker.rank_single_source(vector_results, "vector", k)
        else:
            return []

        return self.ranker.assign_positions(ranked)

    def search_ids(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Versión simplificada que devuelve solo IDs (para evaluación)."""
        results = self.search(query, top_k=top_k)
        return [r["doc_id"] for r in results]

    # ------------------------------------------------------------------
    # Evaluación
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_queries_path: str,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Evalúa el sistema con un archivo de consultas de prueba.
        """
        test_queries = Evaluator.load_test_queries(test_queries_path)
        results = self.evaluator.evaluate_all(
            test_queries=test_queries,
            retrieval_fn=lambda q: self.search_ids(q, top_k=10),
        )
        report = self.evaluator.report(output_path=output_path)
        print(report)
        return results

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def _save_all(self) -> None:
        os.makedirs(INDEX_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.indexer.save(INDEX_DIR)
        self.lsi.save(MODEL_DIR)
        self.vstore.save()
        print("[Pipeline] Todos los módulos guardados.")

    def _load_all(self) -> None:
        self.indexer.load(INDEX_DIR)
        if os.path.exists(os.path.join(MODEL_DIR, "lsi_model.pkl")):
            self.lsi.load(MODEL_DIR)
        self.vstore.load()
