"""
Vector Database — ChromaDB-compatible
======================================
Base de datos vectorial para almacenar y recuperar representaciones
densas de documentos del dominio Tecnología y Software.

Se implementa una capa local persistente compatible con la API de ChromaDB.
Cuando ChromaDB esté disponible en el entorno, se puede activar con
USE_CHROMADB=True para usar el backend real.

Referencia:
  Chroma Documentation. https://docs.trychroma.com/
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Intentamos importar ChromaDB; si no está, usamos backend propio
# ---------------------------------------------------------------------------
try:
    import chromadb
    from chromadb.utils import embedding_functions
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


class LocalEmbedder:
    """
    Embeddigs ligeros basados en TF-IDF cuando no hay modelo de lenguaje
    disponible. Sirve como fallback reproducible.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.vectorizer = TfidfVectorizer(
            max_features=dim,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]{2,}\b",
        )
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        mat = self.vectorizer.transform(texts).toarray()
        # L2-normalizar
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return mat / norms

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self._fitted = True


class VectorStore:
    """
    Base de datos vectorial con API compatible con ChromaDB.

    Operaciones principales:
        add(documents)        — agrega documentos y genera embeddings
        query(query_text, n)  — recupera los n más similares
        get(doc_id)           — recupera un documento por ID
        delete(doc_id)        — elimina un documento
        count()               — número de documentos almacenados
        save() / load()       — persistencia en disco
    """

    def __init__(
        self,
        collection_name: str = "tech_software",
        persist_dir: str = "data/index",
        embedding_dim: int = 256,
        use_chromadb: bool = False,
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.embedding_dim = embedding_dim
        self._use_chroma = use_chromadb and _CHROMA_AVAILABLE

        # Embedder local (siempre disponible)
        self.embedder = LocalEmbedder(dim=embedding_dim)

        # Almacenamiento interno
        self._ids: List[str] = []
        self._embeddings: List[np.ndarray] = []
        self._metadatas: List[Dict] = []
        self._documents: List[str] = []

        # ChromaDB client (opcional)
        self._chroma_collection = None
        if self._use_chroma:
            self._init_chromadb()

    # ------------------------------------------------------------------
    # Inicialización ChromaDB
    # ------------------------------------------------------------------

    def _init_chromadb(self) -> None:
        """Inicializa colección ChromaDB persistente."""
        client = chromadb.PersistentClient(path=self.persist_dir)
        ef = embedding_functions.DefaultEmbeddingFunction()
        self._chroma_collection = client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[VectorDB] ChromaDB inicializado — colección: {self.collection_name}")

    # ------------------------------------------------------------------
    # Operaciones CRUD
    # ------------------------------------------------------------------

    def add(self, documents: List[Dict]) -> None:
        """
        Agrega documentos al vector store.

        Args:
            documents: Lista de dicts {"id", "title", "content", **metadata}
        """
        if not documents:
            return

        texts = [self._doc_text(d) for d in documents]

        if not self.embedder._fitted:
            self.embedder.fit(texts)

        embeddings = self.embedder.embed(texts)

        for i, doc in enumerate(documents):
            doc_id = str(doc.get("id", f"doc_{len(self._ids) + i}"))
            # Evitar duplicados
            if doc_id in self._ids:
                idx = self._ids.index(doc_id)
                self._embeddings[idx] = embeddings[i]
                self._metadatas[idx] = {k: v for k, v in doc.items()
                                        if k != "content"}
                self._documents[idx] = texts[i]
                continue
            self._ids.append(doc_id)
            self._embeddings.append(embeddings[i])
            self._metadatas.append({k: v for k, v in doc.items()
                                    if k not in ("content",)})
            self._documents.append(texts[i])

        # Espejo en ChromaDB si está disponible
        if self._use_chroma and self._chroma_collection:
            self._chroma_collection.upsert(
                ids=[str(d.get("id", i)) for i, d in enumerate(documents)],
                documents=texts,
                metadatas=[{k: str(v) for k, v in d.items()
                            if k not in ("content",)} for d in documents],
            )

        print(f"[VectorDB] {len(documents)} docs agregados. "
              f"Total: {self.count()}")

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Recupera los documentos más similares a la consulta.

        Returns:
            Lista de {"id", "score", "metadata", "document"}
        """
        if self.count() == 0:
            return []

        # Usar ChromaDB si está disponible
        if self._use_chroma and self._chroma_collection:
            results = self._chroma_collection.query(
                query_texts=[query_text],
                n_results=min(n_results, self.count()),
            )
            out = []
            for i, doc_id in enumerate(results["ids"][0]):
                out.append({
                    "id":       doc_id,
                    "score":    1 - results["distances"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "document": results["documents"][0][i],
                })
            return out

        # Backend local
        q_emb = self.embedder.embed([query_text])             # (1, dim)
        matrix = np.vstack(self._embeddings)                  # (n, dim)
        sims = cosine_similarity(q_emb, matrix)[0]           # (n,)

        top_indices = np.argsort(sims)[::-1][:n_results]
        results = []
        for idx in top_indices:
            results.append({
                "id":       self._ids[idx],
                "score":    round(float(sims[idx]), 4),
                "metadata": self._metadatas[idx],
                "document": self._documents[idx],
            })
        return results

    def get(self, doc_id: str) -> Optional[Dict]:
        """Recupera un documento por su ID."""
        if doc_id in self._ids:
            idx = self._ids.index(doc_id)
            return {
                "id":       doc_id,
                "metadata": self._metadatas[idx],
                "document": self._documents[idx],
            }
        return None

    def delete(self, doc_id: str) -> bool:
        """Elimina un documento del store."""
        if doc_id not in self._ids:
            return False
        idx = self._ids.index(doc_id)
        self._ids.pop(idx)
        self._embeddings.pop(idx)
        self._metadatas.pop(idx)
        self._documents.pop(idx)
        return True

    def count(self) -> int:
        """Número de documentos almacenados."""
        return len(self._ids)

    def list_ids(self) -> List[str]:
        return list(self._ids)

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persiste el vector store en disco."""
        os.makedirs(self.persist_dir, exist_ok=True)
        data = {
            "ids":        self._ids,
            "embeddings": [e.tolist() for e in self._embeddings],
            "metadatas":  self._metadatas,
            "documents":  self._documents,
        }
        store_path = os.path.join(self.persist_dir, f"{self.collection_name}.json")
        with open(store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Guardar embedder
        emb_path = os.path.join(self.persist_dir,
                                f"{self.collection_name}_embedder.pkl")
        self.embedder.save(emb_path)
        print(f"[VectorDB] Store guardado: {store_path}")

    def load(self) -> None:
        """Carga el vector store desde disco."""
        store_path = os.path.join(self.persist_dir, f"{self.collection_name}.json")
        if not os.path.exists(store_path):
            print("[VectorDB] No existe store previo. Comenzando vacío.")
            return

        with open(store_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._ids        = data["ids"]
        self._embeddings = [np.array(e) for e in data["embeddings"]]
        self._metadatas  = data["metadatas"]
        self._documents  = data["documents"]

        emb_path = os.path.join(self.persist_dir,
                                f"{self.collection_name}_embedder.pkl")
        if os.path.exists(emb_path):
            self.embedder.load(emb_path)

        print(f"[VectorDB] Store cargado: {self.count()} documentos.")

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_text(doc: Dict) -> str:
        title   = doc.get("title", "") or ""
        content = doc.get("content", "") or ""
        tags    = " ".join(doc.get("tags", [])) if doc.get("tags") else ""
        return f"{title} {content} {tags}".strip()
