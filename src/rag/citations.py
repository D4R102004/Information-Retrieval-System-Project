"""
Citation Management Module

Handles extraction and enrichment of citations from LLM-generated text.
Ensures sources are correctly attributed and validated against retrieved documents.
"""

import re
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict
import logging
from .config import rag_config

logger = logging.getLogger(__name__)

class Citation(BaseModel):
    """Citation object linking to source document."""

    doc_id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    url: Optional[str] = Field(None, description="Document URL")
    snippet: Optional[str] = Field(None, description="Relevant excerpt")
    source: Optional[str] = Field(None, description="Data source")
    score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score")
    date: Optional[str] = Field(None, description="Publication date")


class CitationExtractor:
    """Extract and validate citations from LLM output."""

    CITATION_PATTERN = r"\[([^\]]+)\]"
    """Regex pattern for [doc_id] citation format."""

    CODE_BLOCK_PATTERN = r"```[\s\S]*?```"
    """Regex pattern to identify fenced code blocks for exclusion."""

    @staticmethod
    def _normalize_citation_ids(
        answer: str, citation_ids: List[str], documents: Optional[List[Dict]] = None
    ) -> Tuple[str, List[str]]:
        """
        Normalize citation IDs by deduplicating and optionally validating.

        Args:
            answer: Generated answer text (for potential in-place citation ID replacement)
            citation_ids: Candidate citation IDs
            documents: Optional documents used to validate citations

        Returns:
            Answer text (potentially with normalized citation IDs) and
            ordered, unique citation IDs. If documents are provided,
            only IDs present in the document set are returned and
            answer is not modified.
        """
        # Remove duplicates while preserving order
        unique_ids = list(dict.fromkeys(citation_ids))

        if not documents:
            return answer, unique_ids

        # Build set of known document ids
        valid_doc_ids = {doc.get("id") or doc.get("doc_id") for doc in documents if doc.get("id") or doc.get("doc_id")}

        valid_citations: List[str] = []
        replacements: Dict[str, str] = {}

        for cid in unique_ids:
            # If citation directly matches a document id, accept it
            if cid in valid_doc_ids:
                valid_citations.append(cid)
                continue

            # Support positional citation formats like doc_3 or id_5 -> map to documents[2]/documents[4]
            match = re.match(r'^(?:doc|id)_(\d+)$', cid)
            if match:
                try:
                    idx = int(match.group(1)) - 1 # Convert to 0-based index
                    if 0 <= idx < len(documents):
                        mapped = documents[idx].get('id') or documents[idx].get('doc_id')
                        if mapped:
                            valid_citations.append(mapped)
                            replacements[cid] = mapped
                            logger.debug(f"Mapped positional citation {cid} -> {mapped}")
                            continue
                except Exception:
                    pass

        if replacements:
            def _replace_match(match: re.Match[str]) -> str:
                citation_id = match.group(1)
                return f"[{replacements.get(citation_id, "")}]"

            answer = re.sub(CitationExtractor.CITATION_PATTERN, _replace_match, answer)

        logger.debug(
            "Normalized %s citation ids into %s valid ids",
            len(citation_ids),
            len(valid_citations),
        )
        return answer, valid_citations

    @staticmethod
    def extract_citations(text: str, documents: List[Dict]) -> Tuple[str, List[Citation]]:
        """
        Extract citations from text using [doc_id] format.

        Validates citations against available documents to ensure
        accuracy and prevent hallucinations.

        Args:
            text: Generated text containing citations
            documents: Available documents for validation

        Returns:
            Answer text and list of enriched Citation objects.

        Example:
            >>> text = "Python is great [doc_001]. It has libraries [doc_002]."
            >>> docs = [{"id": "doc_001"}, {"id": "doc_002"}]
            >>> CitationExtractor.extract_citations(text, docs)
            ('Python is great [doc_001]. It has libraries [doc_002].', [Citation(...)])
        """
        # Remove fenced code blocks (triple backticks) to ignore bracketed expressions inside code samples.
        try:
            text_no_code = re.sub(CitationExtractor.CODE_BLOCK_PATTERN, "", text)
        except Exception:
            text_no_code = text

        # Find all citation patterns outside code blocks
        matches = re.findall(CitationExtractor.CITATION_PATTERN, text_no_code)

        text, unique_citation_ids = CitationExtractor._normalize_citation_ids(text, matches, documents)

        if not documents:
            citations = [Citation(doc_id=cid, title=f"Document {cid}", source="extracted") for cid in unique_citation_ids]
            logger.debug(
                f"Extracted {len(citations)} citations from {len(matches)} matches (no documents provided)"
            )
            return text, citations[:rag_config.max_cites]

        enriched_dicts = CitationExtractor._enrich_citations_dicts(unique_citation_ids, documents)
        citations = [Citation(**citation_dict) for citation_dict in enriched_dicts]

        logger.debug(
            f"Extracted {len(citations)} citations from {len(matches)} matches"
        )

        return text, citations[:rag_config.max_cites]

    @staticmethod
    def citations_from_ids(
        text: str, citation_ids: List[str], documents: Optional[List[Dict]] = None
    ) -> Tuple[str, List[Citation]]:
        """
        Convert citation ID strings to enriched Citation objects.

        Handles both cases:
        - With documents: Validates and enriches with metadata
        - Without documents: Creates minimal Citation objects

        Args:
            text: Generated text containing citations
            citation_ids: List of document IDs from LLM
            documents: Optional documents for validation and enrichment

        Returns:
            Tuple of (modified text, list of enriched Citation objects)
        """
        text, normalized_ids = CitationExtractor._normalize_citation_ids(text, citation_ids, documents)

        if not documents:
            # No documents - create minimal Citations from IDs
            return text, [
                Citation(doc_id=cid, title=f"Document {cid}", source="extracted")
                for cid in normalized_ids
            ][:rag_config.max_cites]

        if len(normalized_ids) < len(citation_ids):
            filtered_out = len(citation_ids) - len(normalized_ids)
            logger.info(
                f"Filtered out {filtered_out} hallucinated citations. "
                f"Valid: {len(normalized_ids)}/{len(citation_ids)}"
            )

        enriched_dicts = CitationExtractor._enrich_citations_dicts(normalized_ids, documents)
        return text, [Citation(**citation_dict) for citation_dict in enriched_dicts][:rag_config.max_cites]

    @staticmethod
    def _enrich_citations_dicts(
        citations: List[str],
        documents: List[Dict],
    ) -> List[Dict]:
        """
        Internal helper: Create citation dictionaries with document metadata.

        Args:
            citations: List of document IDs (assumed normalized)
            documents: Available documents with metadata

        Returns:
            List of enriched citation dictionaries with:
                - doc_id: Document identifier
                - title: Document title
                - url: Document URL (if available)
                - source: Data source (if available)
                - snippet: Relevant content excerpt
                - score: Relevance score (if available)
        """
        doc_map = {doc.get("id"): doc for doc in documents if doc.get("id")}
        enriched = []

        for citation_id in citations:
            if citation_id not in doc_map:
                logger.warning(f"Citation {citation_id} not found in documents")
                continue

            doc = doc_map[citation_id]
            snippet = doc.get("content", "")
            if len(snippet) > rag_config.max_snippet_length:
                snippet = snippet[:rag_config.max_snippet_length] + "..."
            enriched_citation = {
                "doc_id": citation_id,
                "title": doc.get("title", "Unknown"),
                "url": doc.get("url", ""),
                "source": doc.get("source", "unknown"),
                "snippet": snippet,
                "score": doc.get("score"),
                "date": doc.get("date"),
            }

            if "popularity" in doc and "score" not in doc:
                enriched_citation["score"] = doc["popularity"]

            enriched.append(enriched_citation)

        logger.info(f"Enriched {len(enriched)} citations")
        return enriched

    @staticmethod
    def enrich_citations(
        text: str,
        citations: List[str],
        documents: List[Dict],
    ) -> Tuple[str, List[Citation]]:
        """
        Create rich citation objects with document metadata.

        Args:
            text: Generated text containing citations (for potential in-place ID normalization)
            citations: List of document IDs
            documents: Available documents with metadata

        Returns:
            Normalized text and
            List of enriched Citation objects with:
                - doc_id: Document identifier
                - title: Document title
                - url: Document URL (if available)
                - source: Data source (if available)
                - snippet: Relevant content excerpt

        Example:
            >>> citations = ["doc_001"]
            >>> docs = [{"id": "doc_001", "title": "Python Basics", "url": "..."}]
            >>> CitationExtractor.enrich_citations(citations, docs)
            [Citation(doc_id="doc_001", title="Python Basics", ...)]
        """
        text, normalized_citations = CitationExtractor._normalize_citation_ids(text, citations, documents)
        enriched_dicts = CitationExtractor._enrich_citations_dicts(normalized_citations, documents)
        return text, [Citation(**citation_dict) for citation_dict in enriched_dicts]

    @staticmethod
    def validate_citations(
        citations: List[str], documents: List[Dict]
    ) -> tuple[List[str], List[str]]:
        """
        Validate citations against available documents.

        Args:
            citations: Document IDs to validate
            documents: Available documents

        Returns:
            Tuple of (valid_citations, invalid_citations)
        """
        valid_doc_ids = {doc.get("id") for doc in documents}
        valid = [c for c in citations if c in valid_doc_ids]
        invalid = [c for c in citations if c not in valid_doc_ids]

        if invalid:
            logger.warning(f"Found {len(invalid)} invalid citations: {invalid}")

        return valid, invalid

    
