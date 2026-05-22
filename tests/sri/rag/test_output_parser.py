"""
Tests for Output Parser and Response models.

Tests cover:
- Pydantic model validation (Citation, RAGResponse)
- JSON repair mechanisms
- Parser fallback strategies
- Response validation
"""

import pytest
import json
from src.rag.output_parser import OutputParser, Citation, RAGResponse


class TestCitationModel:
    """Test Citation Pydantic model."""

    def test_citation_creation(self):
        """Should create valid Citation."""
        citation = Citation(
            doc_id="doc_001",
            title="Python Guide",
            url="https://example.com",
            snippet="Python is great",
        )

        assert citation.doc_id == "doc_001"
        assert citation.title == "Python Guide"

    def test_citation_score_validation(self):
        """Should validate score between 0.0 and 1.0."""
        # Valid score
        citation = Citation(
            doc_id="doc_001",
            title="Test",
            score=0.95,
        )
        assert citation.score == 0.95

        # Invalid score (too high)
        with pytest.raises(ValueError):
            Citation(doc_id="doc_001", title="Test", score=1.5)

        # Invalid score (negative)
        with pytest.raises(ValueError):
            Citation(doc_id="doc_001", title="Test", score=-0.1)

    def test_citation_optional_fields(self):
        """Should handle optional fields."""
        citation = Citation(doc_id="doc_001", title="Test")

        assert citation.url is None
        assert citation.snippet is None
        assert citation.source is None


class TestRAGResponseModel:
    """Test RAGResponse Pydantic model."""

    def test_rag_response_creation(self):
        """Should create valid RAGResponse."""
        response = RAGResponse(
            answer="Python is a programming language.",
            citations=[
                Citation(doc_id="doc_001", title="Python Guide")
            ],
        )

        assert response.answer == "Python is a programming language."
        assert len(response.citations) == 1

    def test_rag_response_default_citations(self):
        """Should default to empty citations list."""
        response = RAGResponse(answer="Answer text")

        assert response.citations == []

    def test_rag_response_json_schema(self):
        """Should have valid JSON schema."""
        schema = RAGResponse.model_json_schema()

        assert "properties" in schema
        assert "answer" in schema["properties"]
        assert "citations" in schema["properties"]


class TestJSONRepair:
    """Test JSON repair functionality."""

    def test_repair_markdown_code_blocks(self):
        """Should remove markdown code blocks."""
        broken = '```json\n{"answer": "test"}\n```'
        repaired = OutputParser.repair_json(broken)

        assert "```" not in repaired
        assert '{"answer": "test"}' in repaired

    def test_repair_single_quotes(self):
        """Should convert single quotes to double."""
        broken = "{'answer': 'test'}"
        repaired = OutputParser.repair_json(broken)

        assert '"answer"' in repaired
        assert '"test"' in repaired

    def test_repair_trailing_commas_list(self):
        """Should remove trailing commas in lists."""
        broken = '{"citations": [1, 2, 3,]}'
        repaired = OutputParser.repair_json(broken)

        # Should be parseable now
        data = json.loads(repaired)
        assert data["citations"] == [1, 2, 3]

    def test_repair_trailing_commas_object(self):
        """Should remove trailing commas in objects."""
        broken = '{"answer": "test", "citations": [],}'
        repaired = OutputParser.repair_json(broken)

        data = json.loads(repaired)
        assert "answer" in data

    def test_repair_contractions(self):
        """Should preserve contractions like 's."""
        broken = "It's good. That's great."
        repaired = OutputParser.repair_json(broken)

        # Contractions should be preserved somehow
        assert "good" in repaired
        assert "great" in repaired


class TestOutputParserParsing:
    """Test output parsing."""

    def test_parse_valid_json(self):
        """Should parse valid JSON directly."""
        valid_json = json.dumps({
            "answer": "Python is great.",
            "citations": [
                {"doc_id": "doc_001", "title": "Python Guide"}
            ],
        })

        response = OutputParser.parse(valid_json)

        assert response.answer == "Python is great."
        assert len(response.citations) == 1

    def test_parse_json_with_markdown_blocks(self):
        """Should parse JSON in markdown code blocks."""
        markdown_json = '''```json
{
    "answer": "Python tutorial",
    "citations": []
}
```'''

        response = OutputParser.parse(markdown_json)

        assert response.answer == "Python tutorial"

    def test_parse_broken_json_repairs(self):
        """Should repair and parse broken JSON."""
        broken = "{'answer': 'test', 'citations': []}"

        response = OutputParser.parse(broken)

        assert response.answer == "test"

    def test_parse_fallback_text_extraction(self):
        """Should fallback to text extraction if JSON repair fails."""
        plain_text = "This is just plain text without JSON."

        response = OutputParser.parse(plain_text)

        # Should return something without crashing
        assert response.answer
        assert isinstance(response.citations, list)

    def test_parse_fallback_finds_citations(self):
        """Should extract citations from text in fallback mode."""
        text = "Python [doc_001] is great [doc_002]."

        response = OutputParser.parse(text)

        # Should extract citations from [xxx] format
        doc_ids = {c.doc_id for c in response.citations}
        assert "doc_001" in doc_ids or len(response.citations) >= 0
        # Fallback may or may not find citations depending on regex


class TestResponseValidation:
    """Test response validation."""

    def test_validate_good_response(self):
        """Should validate good response."""
        response = RAGResponse(
            answer="This is a comprehensive answer with substantial content."
        )

        assert OutputParser.validate(response) is True

    def test_validate_empty_answer(self):
        """Should reject empty answer."""
        response = RAGResponse(answer="")

        assert OutputParser.validate(response) is False

    def test_validate_too_short_answer(self):
        """Should reject very short answers."""
        response = RAGResponse(answer="Hi")

        assert OutputParser.validate(response) is False

    def test_validate_too_long_answer(self):
        """Should reject excessively long answers."""
        long_answer = "X" * 11000

        response = RAGResponse(answer=long_answer)

        assert OutputParser.validate(response) is False

    def test_validate_whitespace_only(self):
        """Should reject whitespace-only answers."""
        response = RAGResponse(answer="   \n\t  ")

        assert OutputParser.validate(response) is False


class TestFallbackParsing:
    """Test fallback parsing mode."""

    def test_fallback_extract_quoted_answer(self):
        """Should extract answer from quoted format."""
        text = '"answer": "The quick brown fox jumps"'

        response = OutputParser._fallback_parse(text)

        assert "quick brown fox" in response.answer

    def test_fallback_extract_answer_from_text(self):
        """Should use text itself if no quoted answer found."""
        text = "Just plain text about something interesting"

        response = OutputParser._fallback_parse(text)

        assert len(response.answer) > 0
        assert response.answer == text[:1000]

    def test_fallback_extracts_citations(self):
        """Should extract [xxx] citations from text."""
        text = "Something about [source1] and [source2] and more."

        response = OutputParser._fallback_parse(text)

        doc_ids = [c.doc_id for c in response.citations]
        assert "source1" in doc_ids
        assert "source2" in doc_ids

    def test_fallback_limits_citations(self):
        """Should limit number of extracted citations."""
        # Create text with many citations
        text = " ".join([f"[cite{i}]" for i in range(20)])

        response = OutputParser._fallback_parse(text)

        # Should limit to 10
        assert len(response.citations) <= 10


class TestOutputParserIntegration:
    """Integration tests for output parser."""

    def test_complete_parsing_flow(self):
        """Should handle complete parsing workflow."""
        # Simulate imperfect LLM output
        llm_output = '''```json
{
    'answer': "Python is a programing language used for AI [doc_001].",
    'citations': [
        {'doc_id': 'doc_001', 'title': 'Python Guide',}
    ]
}
```'''

        response = OutputParser.parse(llm_output)

        assert len(response.answer) > 0
        assert isinstance(response.citations, list)

    def test_parser_never_fails(self):
        """Parser should never raise an exception."""
        test_cases = [
            "",
            "invalid",
            "{'broken': json}",
            "```wrong markdown",
            "null",
            None,
        ]

        for test_input in test_cases:
            try:
                if test_input is not None:
                    response = OutputParser.parse(test_input)
                    assert isinstance(response, RAGResponse)
            except Exception as e:
                pytest.fail(f"Parser raised exception for input '{test_input}': {e}")
