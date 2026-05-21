"""
Output Parsing and Validation Module

Parses LLM-generated text into structured RAGResponse format.
Includes JSON repair mechanisms to handle imperfect LLM output.
"""

import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field
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


class RAGResponse(BaseModel):
    """Structured RAG response."""

    answer: str = Field(..., description="Generated answer text")
    citations: List[Citation] = Field(
        default_factory=list, description="Source citations"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
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
    def parse(cls, text: str) -> RAGResponse:
        """
        Parse LLM output to structured response.

        Implements fallback strategy:
        1. Try direct JSON parsing
        2. Try repair and retry
        3. Fall back to text extraction

        Args:
            text: Raw LLM output

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
            response = RAGResponse(**data)
            logger.info("Successfully parsed output as JSON")
            return response
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Direct parsing failed: {e}")

        # Attempt 2: Repair and retry
        try:
            repaired = cls.repair_json(text)
            data = json.loads(repaired)
            response = RAGResponse(**data)
            logger.info("Successfully parsed output after repair")
            return response
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Repair parsing failed: {e}")

        # Attempt 3: Fallback extraction
        logger.warning("Using fallback text extraction - output quality may be degraded")
        return cls._fallback_parse(text)

    @staticmethod
    def _fallback_parse(text: str) -> RAGResponse:
        """Fallback parser using regex to extract answer and citations.

        When JSON parsing fails, extracts:
        - Answer: First 500-1000 chars of text
        - Citations: Any [text] patterns found

        Args:
            text: Raw text

        Returns:
            RAGResponse with extracted information
        """
        logger.debug("Entering fallback parsing mode...")

        # Try to find JSON answer block
        answer_match = re.search(
            r'"answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL
        )

        if answer_match:
            answer = answer_match.group(1)
        else:
            # Use first 1000 chars as answer
            answer = text[:1000]

        # Extract citations from [doc_id] patterns
        citations = re.findall(r"\[([^\]]+)\]", text)

        # Remove duplicates
        citations = list(dict.fromkeys(citations))[:10]  # Limit to 10

        logger.info(f"Fallback extraction: answer ({len(answer)} chars), {len(citations)} citations")

        return RAGResponse(
            answer=answer,
            citations=[Citation(doc_id=c, title=c, source="extracted") for c in citations],
        )

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
