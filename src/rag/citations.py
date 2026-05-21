"""
Citation Management Module

Handles extraction and enrichment of citations from LLM-generated text.
Ensures sources are correctly attributed and validated against retrieved documents.
"""

import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class CitationExtractor:
    """Extract and validate citations from LLM output."""

    CITATION_PATTERN = r"\[([^\]]+)\]"
    """Regex pattern for [doc_id] citation format."""

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

        # Validate against available documents
        valid_doc_ids = {doc.get("id") for doc in documents}
        valid_citations = [m for m in matches if m in valid_doc_ids]

        # Remove duplicates while preserving order
        unique_citations = list(dict.fromkeys(valid_citations))

        logger.debug(
            f"Extracted {len(unique_citations)} citations from {len(matches)} matches"
        )

        return unique_citations

    @staticmethod
    def enrich_citations(
        citations: List[str],
        documents: List[Dict],
    ) -> List[Dict]:
        """
        Create rich citation objects with document metadata.

        Args:
            citations: List of document IDs
            documents: Available documents with metadata

        Returns:
            List of enriched citation dictionaries with:
                - doc_id: Document identifier
                - title: Document title
                - url: Document URL (if available)
                - source: Data source (if available)
                - snippet: Relevant content excerpt

        Example:
            >>> citations = ["doc_001"]
            >>> docs = [{"id": "doc_001", "title": "Python Basics", "url": "..."}]
            >>> CitationExtractor.enrich_citations(citations, docs)
            [{'doc_id': 'doc_001', 'title': 'Python Basics', ...}]
        """
        # Create document map for fast lookup
        doc_map = {doc.get("id"): doc for doc in documents}
        enriched = []

        for citation_id in citations:
            if citation_id not in doc_map:
                logger.warning(f"Citation {citation_id} not found in documents")
                continue

            doc = doc_map[citation_id]
            enriched_citation = {
                "doc_id": citation_id,
                "title": doc.get("title", "Unknown"),
                "url": doc.get("url", ""),
                "source": doc.get("source", "unknown"),
                "snippet": doc.get("content", "")[:200],  # First 200 chars
            }

            # Add optional metadata if available
            if "popularity" in doc:
                enriched_citation["popularity"] = doc["popularity"]
            if "date" in doc:
                enriched_citation["date"] = doc["date"]

            enriched.append(enriched_citation)

        logger.info(f"Enriched {len(enriched)} citations")

        return enriched

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
