"""
Módulo de Indexación
=====================
Construye y mantiene índices invertidos para recuperación eficiente
en el dominio Tecnología y Software.

Integra:
- Preprocesamiento NLP (tokenización, stemming, normalización)
- Índice invertido con TF, DF, y TF-IDF pesos
- Serialización eficiente (JSON)
"""

import os
import re
import json
import math
import string
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional


# ---------------------------------------------------------------------------
# Stop words en español e inglés para el dominio tech/software
# ---------------------------------------------------------------------------
STOP_WORDS_ES = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con",
    "contra", "cual", "cuando", "de", "del", "desde", "donde", "durante",
    "e", "el", "ella", "ellas", "ellos", "en", "entre", "era", "erais",
    "eran", "eras", "eres", "es", "esa", "esas", "ese", "eso", "esos",
    "esta", "estaba", "estado", "estamos", "estan", "estar", "estas",
    "este", "esto", "estos", "estuvo", "fue", "fueron", "ha", "habia",
    "han", "hasta", "hay", "he", "la", "las", "le", "les", "lo", "los",
    "mas", "me", "mi", "mientras", "mis", "mucho", "muy", "no", "nos",
    "nuestro", "o", "para", "pero", "por", "que", "quien", "se", "ser",
    "si", "sin", "sobre", "son", "su", "sus", "también", "tan", "te",
    "tengo", "toda", "todo", "todos", "tu", "tus", "un", "una", "uno",
    "y", "ya", "yo",
}

STOP_WORDS_EN = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "this", "that", "these",
    "those", "it", "its", "we", "our", "you", "your", "they", "their",
    "he", "she", "his", "her", "not", "no", "as", "if", "so", "all",
    "any", "each", "more", "also", "about", "into", "than", "then",
    "when", "how", "what", "which", "who",
}

STOP_WORDS = STOP_WORDS_ES | STOP_WORDS_EN


def simple_stem(word: str) -> str:
    """Stemming básico para español/inglés (sufijos comunes)."""
    suffixes = [
        "aciones", "ación", "mente", "idades", "idad", "ando", "endo",
        "ados", "idos", "ado", "ido", "ando", "ar", "er", "ir",
        "ing", "tion", "tions", "ness", "ment", "ments", "ers", "es", "s",
    ]
    for suf in suffixes:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


class TextPreprocessor:
    """Normalización y tokenización de texto."""

    def __init__(self, use_stemming: bool = True):
        self.use_stemming = use_stemming

    def process(self, text: str) -> List[str]:
        """
        Devuelve lista de tokens normalizados.
        """
        if not text:
            return []
        # Minúsculas + eliminar acentos simples para comparación
        text = text.lower()
        text = (text
                .replace("á", "a").replace("é", "e").replace("í", "i")
                .replace("ó", "o").replace("ú", "u").replace("ü", "u")
                .replace("ñ", "n"))
        # Solo letras y números
        tokens = re.findall(r"\b[a-z0-9][a-z0-9_\-\.]*[a-z0-9]\b|[a-z0-9]", text)
        # Filtrar stop words y tokens muy cortos
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) >= 2]
        if self.use_stemming:
            tokens = [simple_stem(t) for t in tokens]
        return tokens


class InvertedIndex:
    """
    Índice invertido con pesos TF-IDF.

    Estructura:
        index[term] = {
            "df": int,
            "postings": {
                doc_id: {"tf": int, "tfidf": float, "positions": [int]}
            }
        }
    """

    def __init__(self, use_stemming: bool = True):
        self.preprocessor = TextPreprocessor(use_stemming)
        self.index: Dict[str, Dict] = {}
        self.doc_lengths: Dict[str, int] = {}   # número de tokens por doc
        self.doc_metadata: Dict[str, Dict] = {} # metadata de cada doc
        self.num_docs: int = 0

    # ------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------

    def build(self, documents: List[Dict]) -> None:
        """
        Construye el índice invertido desde una lista de documentos.

        Args:
            documents: Lista de {"id", "title", "content", **}
        """
        self.index = {}
        self.doc_lengths = {}
        self.doc_metadata = {}
        self.num_docs = len(documents)

        # Primera pasada: TF y positions
        for doc in documents:
            doc_id = str(doc.get("id", ""))
            text = f"{doc.get('title', '')} {doc.get('content', '')}"
            tokens = self.preprocessor.process(text)
            self.doc_lengths[doc_id] = len(tokens)
            self.doc_metadata[doc_id] = {
                k: v for k, v in doc.items() if k != "content"
            }
            for pos, token in enumerate(tokens):
                if token not in self.index:
                    self.index[token] = {"df": 0, "postings": {}}
                postings = self.index[token]["postings"]
                if doc_id not in postings:
                    postings[doc_id] = {"tf": 0, "tfidf": 0.0, "positions": []}
                postings[doc_id]["tf"] += 1
                postings[doc_id]["positions"].append(pos)

        # Segunda pasada: DF y TF-IDF
        N = max(self.num_docs, 1)
        for term, entry in self.index.items():
            df = len(entry["postings"])
            entry["df"] = df
            idf = math.log((N + 1) / (df + 1)) + 1   # suavizado

            for doc_id, posting in entry["postings"].items():
                tf = posting["tf"]
                # TF logarítmico
                tf_log = 1 + math.log(tf) if tf > 0 else 0
                posting["tfidf"] = round(tf_log * idf, 4)

        print(f"[Índice] Construido: {self.num_docs} docs, "
              f"{len(self.index)} términos únicos.")

    def add_document(self, doc: Dict) -> None:
        """Agrega un documento al índice existente (actualización incremental)."""
        doc_id = str(doc.get("id", ""))
        if doc_id in self.doc_metadata:
            return  # ya existe

        self.num_docs += 1
        text = f"{doc.get('title', '')} {doc.get('content', '')}"
        tokens = self.preprocessor.process(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_metadata[doc_id] = {k: v for k, v in doc.items() if k != "content"}

        for pos, token in enumerate(tokens):
            if token not in self.index:
                self.index[token] = {"df": 0, "postings": {}}
            postings = self.index[token]["postings"]
            if doc_id not in postings:
                postings[doc_id] = {"tf": 0, "tfidf": 0.0, "positions": []}
            postings[doc_id]["tf"] += 1
            postings[doc_id]["positions"].append(pos)

        # Recalcular IDF solo para tokens nuevos (aproximación)
        N = self.num_docs
        for token in set(tokens):
            if token in self.index:
                entry = self.index[token]
                df = len(entry["postings"])
                entry["df"] = df
                idf = math.log((N + 1) / (df + 1)) + 1
                for did, posting in entry["postings"].items():
                    tf = posting["tf"]
                    tf_log = 1 + math.log(tf) if tf > 0 else 0
                    posting["tfidf"] = round(tf_log * idf, 4)

    # ------------------------------------------------------------------
    # Búsqueda booleana básica (complementa LSI)
    # ------------------------------------------------------------------

    def boolean_search(self, query: str, mode: str = "AND") -> Set[str]:
        """
        Búsqueda booleana simple (AND / OR) sobre el índice.
        """
        tokens = self.preprocessor.process(query)
        if not tokens:
            return set()

        result_sets = []
        for token in tokens:
            if token in self.index:
                result_sets.append(set(self.index[token]["postings"].keys()))
            else:
                result_sets.append(set())

        if not result_sets:
            return set()

        if mode == "AND":
            result = result_sets[0]
            for s in result_sets[1:]:
                result = result & s
        else:  # OR
            result = set()
            for s in result_sets:
                result = result | s

        return result

    def get_tfidf_scores(self, query: str) -> Dict[str, float]:
        """
        Devuelve scores TF-IDF acumulados por documento para la consulta.
        """
        tokens = self.preprocessor.process(query)
        scores: Dict[str, float] = defaultdict(float)
        for token in tokens:
            if token in self.index:
                for doc_id, posting in self.index[token]["postings"].items():
                    scores[doc_id] += posting["tfidf"]
        return dict(scores)

    # ------------------------------------------------------------------
    # Estadísticas
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        return {
            "num_docs":   self.num_docs,
            "vocab_size": len(self.index),
            "avg_doc_len": (
                sum(self.doc_lengths.values()) / max(len(self.doc_lengths), 1)
            ),
            "top_10_terms": sorted(
                self.index.items(), key=lambda x: x[1]["df"], reverse=True
            )[:10],
        }

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        payload = {
            "num_docs":      self.num_docs,
            "doc_lengths":   self.doc_lengths,
            "doc_metadata":  self.doc_metadata,
            "index":         self.index,
        }
        out_path = os.path.join(path, "inverted_index.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"[Índice] Guardado: {out_path} "
              f"({os.path.getsize(out_path) / 1024:.1f} KB)")

    def load(self, path: str) -> None:
        idx_path = os.path.join(path, "inverted_index.json")
        if not os.path.exists(idx_path):
            print("[Índice] No existe índice previo.")
            return
        with open(idx_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.num_docs     = data["num_docs"]
        self.doc_lengths  = data["doc_lengths"]
        self.doc_metadata = data["doc_metadata"]
        self.index        = data["index"]
        print(f"[Índice] Cargado: {self.num_docs} docs, "
              f"{len(self.index)} términos.")
