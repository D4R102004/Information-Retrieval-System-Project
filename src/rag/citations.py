"""
Citation Management Module

Handles extraction and enrichment of citations from LLM-generated text.
Ensures sources are correctly attributed and validated against retrieved documents.
"""

import re
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict
import logging

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

    @staticmethod
    def normalize_citation_ids(
        citation_ids: List[str], documents: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Normalize citation IDs by deduplicating and optionally validating.

        Args:
            citation_ids: Candidate citation IDs
            documents: Optional documents used to validate citations

        Returns:
            Ordered, unique citation IDs. If documents are provided,
            only IDs present in the document set are returned.
        """
        # Remove duplicates while preserving order
        unique_ids = list(dict.fromkeys(citation_ids))

        if not documents:
            return unique_ids

        valid_doc_ids = {doc.get("id") for doc in documents if doc.get("id")}
        valid_citations = [cid for cid in unique_ids if cid in valid_doc_ids]

        logger.debug(
            "Normalized %s citation ids into %s valid ids",
            len(citation_ids),
            len(valid_citations),
        )
        return valid_citations

    @staticmethod
    def extract_citations(text: str, documents: List[Dict]) -> List[str]:
        """
        Extract citation IDs from text using [doc_id] format.

        Validates citations against available documents to ensure
        accuracy and prevent hallucinations.

        Args:
            text: Generated text containing citations
            documents: Available documents for validation

        Returns:
            List of cited document IDs (deduplicated, preserving order)

        Example:
            >>> text = "Python is great [doc_001]. It has libraries [doc_002]."
            >>> docs = [{"id": "doc_001"}, {"id": "doc_002"}]
            >>> CitationExtractor.extract_citations(text, docs)
            ['doc_001', 'doc_002']
        """
        # Find all citation patterns
        matches = re.findall(CitationExtractor.CITATION_PATTERN, text)

        unique_citations = CitationExtractor.normalize_citation_ids(matches, documents)

        logger.debug(
            f"Extracted {len(unique_citations)} citations from {len(matches)} matches"
        )

        return unique_citations

    @staticmethod
    def citations_from_ids(
        citation_ids: List[str], documents: Optional[List[Dict]] = None
    ) -> List[Citation]:
        """
        Convert citation ID strings to enriched Citation objects.

        Handles both cases:
        - With documents: Validates and enriches with metadata
        - Without documents: Creates minimal Citation objects

        Args:
            citation_ids: List of document IDs from LLM
            documents: Optional documents for validation and enrichment

        Returns:
            List of enriched Citation objects
        """
        normalized_ids = CitationExtractor.normalize_citation_ids(citation_ids)

        if not documents:
            # No documents - create minimal Citations from IDs
            return [
                Citation(doc_id=cid, title=f"Document {cid}", source="extracted")
                for cid in normalized_ids
            ]

        if len(normalized_ids) < len(citation_ids):
            filtered_out = len(citation_ids) - len(normalized_ids)
            logger.info(
                f"Filtered out {filtered_out} hallucinated citations. "
                f"Valid: {len(normalized_ids)}/{len(citation_ids)}"
            )

        enriched_dicts = CitationExtractor._enrich_citations_dicts(normalized_ids, documents)
        return [Citation(**citation_dict) for citation_dict in enriched_dicts]

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
            snippet = doc.get("snippet") or doc.get("content", "")[:200]
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
        citations: List[str],
        documents: List[Dict],
    ) -> List[Citation]:
        """
        Create rich citation objects with document metadata.

        Args:
            citations: List of document IDs
            documents: Available documents with metadata

        Returns:
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
        normalized_citations = CitationExtractor.normalize_citation_ids(citations, documents)
        enriched_dicts = CitationExtractor._enrich_citations_dicts(normalized_citations, documents)
        return [Citation(**citation_dict) for citation_dict in enriched_dicts]

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

    
