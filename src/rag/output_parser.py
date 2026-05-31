"""
Output Parsing and Validation Module

Parses LLM-generated text into structured RAGResponse format.
Includes JSON repair mechanisms to handle imperfect LLM output.
"""

import json
import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import logging

from .citations import CitationExtractor, Citation
from .config import config as rag_config

logger = logging.getLogger(__name__)

class RAGResponse(BaseModel):
    """Structured RAG response with metadata."""

    answer: str = Field(..., description="Generated answer text")
    citations: List[Citation] = Field(
        default_factory=list, description="Source citations"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata (documents used, sources, timing, etc.)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "Python is widely used for machine learning...",
                "citations": [
                    {
                        "doc_id": "doc_001",
                        "title": "Python for ML",
                        "url": "https://example.com",
                        "snippet": "Python has excellent libraries...",
                        "source": "documentation",
                        "score": 0.95,
                    }
                ],
                "metadata": {
                    "auto_crawled": False,
                    "total_documents_used": 3,
                    "local_documents": 3,
                    "web_documents": 0,
                    "insufficiency_detected": False,
                    "insufficiency_reasons": [],
                    "generation_time_seconds": 2.5,
                },
            }
        }
    )


class OutputParser:
    """Parse and repair LLM output with multiple fallback strategies."""

    @staticmethod
    def repair_json(text: str) -> str:
        """
        Repair common JSON issues in LLM output.

        Handles:
        - Markdown code blocks (```json ... ```)
        - Single quotes → double quotes
        - Trailing commas
        - Common syntax errors

        Args:
            text: Raw LLM output

        Returns:
            Repaired JSON string

        Example:
            >>> broken = "{'answer': 'Hello', 'citations': []}"
            >>> OutputParser.repair_json(broken)
            '{"answer": "Hello", "citations": []}'
        """
        logger.debug("Attempting to repair JSON...")

        # Remove markdown code blocks
        text = re.sub(r"```json\s*|\s*```", "", text)
        text = re.sub(r"^```", "", text)

        # Single to double quotes (careful with contractions like 's)
        text = re.sub(r"(?<!\w)'([^']*)'(?!\w)", r'"\1"', text)

        # Remove trailing commas before ] or }
        text = re.sub(r",\s*]", "]", text)
        text = re.sub(r",\s*}", "}", text)

        # Fix contractions (restore 's if it was affected)
        text = text.replace('"\'s', "'s")

        logger.debug("JSON repair completed")

        return text

    @classmethod
    def parse(cls, text: str, documents: Optional[List[dict]] = None) -> RAGResponse:
        """
        Parse LLM output to structured response.

        Implements fallback strategy:
        1. Try direct JSON parsing
        2. Try repair and retry
        3. Fall back to text extraction with document enrichment

        Args:
            text: Raw LLM output
            documents: Optional context documents for citation enrichment.
                Used to link citations to real document metadata.
                Each document should have keys: id, title, url (optional)

        Returns:
            RAGResponse with answer and citations

        Note:
            Never fails - always returns a valid RAGResponse.
            Degraded output is returned if parsing fails.
        """
        logger.debug(f"Parsing output ({len(text)} chars)...")

        # Attempt 1: Direct parsing
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError(f"Expected JSON object, got {type(data).__name__}")
            answer = data.get("answer", "")
            citation_ids = data.get("citations", [])
            
            # Handle both Citation objects and string IDs
            if citation_ids and isinstance(citation_ids[0], dict):
                # Already enriched objects (shouldn't happen with LLM)
                response = RAGResponse(**data)
            else:
                # String IDs from LLM - convert to Citation objects
                answer, citations = CitationExtractor.citations_from_ids(answer, citation_ids, documents)
                response = RAGResponse(answer=answer, citations=citations)
            
            logger.info(f"Successfully parsed output as JSON with {len(response.citations)} citations")
            return response
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            logger.debug(f"Direct parsing failed: {e}")

        # Attempt 2: Repair and retry
        try:
            repaired = cls.repair_json(text)
            data = json.loads(repaired)
            if not isinstance(data, dict):
                raise ValueError(f"Expected JSON object, got {type(data).__name__}")
            answer = data.get("answer", "")
            citation_ids = data.get("citations", [])
            
            # Handle both Citation objects and string IDs
            if citation_ids and isinstance(citation_ids[0], dict):
                # Already enriched objects
                response = RAGResponse(**data)
            else:
                # String IDs from LLM - convert to Citation objects
                answer, citations = CitationExtractor.citations_from_ids(answer, citation_ids, documents)
                response = RAGResponse(answer=answer, citations=citations)
            
            logger.info(f"Successfully parsed output after repair with {len(response.citations)} citations")
            return response
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            logger.debug(f"Repair parsing failed: {e}")

        # Attempt 3: Fallback extraction with document enrichment
        logger.warning("Using fallback text extraction - output quality may be degraded")
        return cls._fallback_parse(text, documents)

    @staticmethod
    def _fallback_parse(text: str, documents: Optional[List[dict]] = None) -> RAGResponse:
        """Fallback parser using regex to extract answer and citations.

        When JSON parsing fails, extracts:
        - Answer: First RESPONSE_CHAR_LIMIT chars of text
        - Citations: Any [text] patterns found, enriched with document metadata

        Args:
            text: Raw text from LLM
            documents: Optional list of documents to enrich citations with metadata

        Returns:
            RAGResponse with extracted information and enriched citations
        """
        logger.debug("Entering fallback parsing mode...")

        # Try to find JSON answer block (in case it's partially in JSON)
        answer_match = re.search(
            r'"answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL
        )

        if answer_match:
            answer = answer_match.group(1)[:rag_config.response_char_limit]
        else:
            # Use first rag_config.response_char_limit chars as answer
            answer = text[:rag_config.response_char_limit]

        # Extract Citations
        answer, citations = CitationExtractor.extract_citations(answer, documents or [])
        
        # Fallback in case of exract_citations failiure.
        if not citations and documents:
            citation_ids = re.findall(r"\[([^\]]+)\]", text)
            citation_ids = list(dict.fromkeys(citation_ids))[:rag_config.max_cites]
            answer, citations = CitationExtractor.citations_from_ids(answer, citation_ids, documents)

        logger.info(
            f"Fallback extraction: answer ({len(answer)} chars), "
            f"{len(citations)} citations {f'enriched from {len(documents)} docs' if documents else '(unenriched)'}"
        )

        return RAGResponse(answer=answer, citations=citations)

    @staticmethod
    def validate(response: RAGResponse) -> bool:
        """
        Validate RAGResponse quality.

        Args:
            response: RAGResponse to validate

        Returns:
            True if response meets basic quality criteria
        """
        # Check minimum answer length
        if not response.answer or len(response.answer.strip()) < 10:
            logger.warning("Answer is too short")
            return False

        # Check for reasonable length
        if len(response.answer) > 10000:
            logger.warning("Answer is excessively long")
            return False

        # All validations passed
        return True
