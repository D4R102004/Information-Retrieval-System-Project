"""
Output Parsing and Validation Module

Parses LLM-generated text into structured RAGResponse format.
Includes JSON repair mechanisms to handle imperfect LLM output.
"""

import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
import logging

logger = logging.getLogger(__name__)
CITATION_LIMIT = 10
RESPONSE_CHAR_LIMIT = 1000

class Citation(BaseModel):
    """Citation object linking to source document."""

    doc_id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    url: Optional[str] = Field(None, description="Document URL")
    snippet: Optional[str] = Field(None, description="Relevant excerpt")
    source: Optional[str] = Field(None, description="Data source")
    score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score")


class RAGResponse(BaseModel):
    """Structured RAG response."""

    answer: str = Field(..., description="Generated answer text")
    citations: List[Citation] = Field(
        default_factory=list, description="Source citations"
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
            }
        }
    )


class OutputParser:
    """Parse and repair LLM output with multiple fallback strategies."""

    @staticmethod
    def _citations_from_ids(
        citation_ids: List[str], documents: Optional[List[dict]] = None
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
        if not documents:
            # No documents - create minimal Citations from IDs
            return [
                Citation(doc_id=cid, title=f"Document {cid}", source="extracted")
                for cid in citation_ids
            ]

        # Build document lookup
        doc_map = {doc.get("id"): doc for doc in documents if doc.get("id")}

        citations = []

        for cid in citation_ids:
            if cid not in doc_map:
                # Citation hallucinated
                logger.warning(
                    f"Hallucinated citation detected: [{cid}] not in document set. "
                    f"Available: {list(doc_map.keys())}"
                )
                continue

            # Document exists - create enriched Citation
            doc = doc_map[cid]
            citations.append(
                Citation(
                    doc_id=cid,
                    title=doc.get("title", "Untitled"),
                    url=doc.get("url"),
                    snippet=doc.get("snippet"),
                    source=doc.get("source", "retrieved"),
                    score=doc.get("score"),
                )
            )

        if len(citations) < len(citation_ids):
            filtered_out = len(citation_ids) - len(citations)
            logger.info(
                f"Filtered out {filtered_out} hallucinated citations. "
                f"Valid: {len(citations)}/{len(citation_ids)}"
            )

        return citations

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
                citations = cls._citations_from_ids(citation_ids, documents)
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
                citations = cls._citations_from_ids(citation_ids, documents)
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
            answer = answer_match.group(1)
        else:
            # Use first RESPONSE_CHAR_LIMIT chars as answer
            answer = text[:RESPONSE_CHAR_LIMIT]

        # Extract citation IDs from [doc_id] patterns
        citation_ids = re.findall(r"\[([^\]]+)\]", text)
        
        # Remove duplicates while preserving order
        citation_ids = list(dict.fromkeys(citation_ids))[:CITATION_LIMIT]

        # Convert IDs to enriched Citation objects
        citations = OutputParser._citations_from_ids(citation_ids, documents)

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
