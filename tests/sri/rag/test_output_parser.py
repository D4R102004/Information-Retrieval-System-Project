"""
Tests for Output Parser and Response models.

Tests cover:
- Pydantic model validation (Citation, RAGResponse)
- JSON repair mechanisms
- Parser fallback strategies
- Response validation
- Citation enrichment and hallucination filtering
"""

from pathlib import Path
import json
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.output_parser import OutputParser, RAGResponse
from src.rag.citations import Citation, CitationExtractor


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
        """Should parse valid JSON directly and enrich string citations."""
        valid_json = json.dumps({
            "answer": "Python is great.",
            "citations": ["doc_001"],
        })

        documents = [
            {
                "id": "doc_001",
                "title": "Python Guide",
                "url": "https://example.com/python-guide",
                "snippet": "Python is a versatile language.",
                "source": "documentation",
                "score": 0.95,
            }
        ]

        response = OutputParser.parse(valid_json, documents)

        assert response.answer == "Python is great."
        assert len(response.citations) == 1
        assert response.citations[0].doc_id == "doc_001"
        assert response.citations[0].title == "Python Guide"
        assert response.citations[0].url == "https://example.com/python-guide"
        assert response.citations[0].source == "documentation"

    def test_extract_citations_returns_enriched_citations(self):
        """Should return enriched Citation objects directly from extract_citations."""
        text = "Python is great [doc_001] and ML is useful [doc_002]."
        documents = [
            {
                "id": "doc_001",
                "title": "Python Guide",
                "url": "https://example.com/python-guide",
                "content": "Python is a versatile language.",
                "source": "documentation",
                "score": 0.95,
            },
            {
                "id": "doc_002",
                "title": "ML Guide",
                "url": "https://example.com/ml-guide",
                "content": "Machine learning concepts.",
                "source": "documentation",
                "score": 0.90,
            },
        ]

        answer, citations = CitationExtractor.extract_citations(text, documents)

        assert answer == text
        assert len(citations) == 2
        assert [c.doc_id for c in citations] == ["doc_001", "doc_002"]
        assert citations[0].title == "Python Guide"
        assert citations[1].title == "ML Guide"

    def test_extract_citations_ignores_code_blocks(self):
        """Should ignore bracketed expressions inside fenced code blocks."""
        text = (
            "Explanation [doc_001].\n\n"
            "```python\n"
            "code_sample = [doc_999]\n"
            "more = [id_2]\n"
            "```\n\n"
            "Another reference [doc_002]."
        )

        documents = [
            {"id": "doc_001", "title": "Doc One", "content": "First document"},
            {"id": "doc_002", "title": "Doc Two", "content": "Second document"},
        ]

        answer, citations = CitationExtractor.extract_citations(text, documents)

        assert answer == text
        assert [c.doc_id for c in citations] == ["doc_001", "doc_002"]
        assert all(c.doc_id != "doc_999" for c in citations)

    def test_parse_valid_json_filters_hallucinated_citations(self):
        """Should drop citations that are not present in the documents."""
        valid_json = json.dumps({
            "answer": "Python is great.",
            "citations": ["doc_001", "doc_fake"],
        })

        documents = [
            {
                "id": "doc_001",
                "title": "Python Guide",
                "url": "https://example.com/python-guide",
            }
        ]

        response = OutputParser.parse(valid_json, documents)

        assert len(response.citations) == 1
        assert response.citations[0].doc_id == "doc_001"

    def test_parse_json_with_markdown_blocks(self):
        """Should parse JSON in markdown code blocks."""
        markdown_json = '''```json
{
    "answer": "Python tutorial",
    "citations": ["doc_001"]
}
```'''

        documents = [
            {
                "id": "doc_001",
                "title": "Python Guide",
                "url": "https://example.com/python-guide",
            }
        ]

        response = OutputParser.parse(markdown_json, documents)

        assert response.answer == "Python tutorial"
        assert len(response.citations) == 1
        assert response.citations[0].title == "Python Guide"

    def test_parse_broken_json_repairs(self):
        """Should repair and parse broken JSON."""
        broken = "{'answer': 'test', 'citations': ['doc_001']}"

        documents = [
            {
                "id": "doc_001",
                "title": "Test Document",
                "url": "https://example.com/test",
            }
        ]

        response = OutputParser.parse(broken, documents)

        assert response.answer == "test"
        assert len(response.citations) == 1
        assert response.citations[0].title == "Test Document"

    def test_parse_fallback_text_extraction(self):
        """Should fallback to text extraction if JSON repair fails."""
        plain_text = "This is just plain text without JSON."

        response = OutputParser.parse(plain_text)

        # Should return something without crashing
        assert response.answer
        assert isinstance(response.citations, list)

    def test_parse_fallback_finds_citations(self):
        """Should extract and enrich citations from text in fallback mode."""
        text = "Python [doc_001] is great [doc_002]."

        documents = [
            {"id": "doc_001", "title": "Python Guide", "url": "https://example.com/python"},
            {"id": "doc_002", "title": "ML Guide", "url": "https://example.com/ml"},
        ]

        response = OutputParser.parse(text, documents)

        doc_ids = {c.doc_id for c in response.citations}
        assert doc_ids == {"doc_001", "doc_002"}
        assert {c.title for c in response.citations} == {"Python Guide", "ML Guide"}

    def test_extract_citations_maps_positional_ids(self):
        """Should map [doc_i] and [id_i] to the i-th document in the list."""
        text = "First [doc_1], second [id_2], third [doc_3]."
        documents = [
            {"id": "alpha", "title": "Alpha", "content": "A"},
            {"id": "beta", "title": "Beta", "content": "B"},
            {"id": "gamma", "title": "Gamma", "content": "C"},
        ]

        answer, citations = CitationExtractor.extract_citations(text, documents)

        assert answer == "First [alpha], second [beta], third [gamma]."
        assert [c.doc_id for c in citations] == ["alpha", "beta", "gamma"]


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

    def test_fallback_ignores_code_block_citations(self):
        """Should ignore citations that appear only inside fenced code blocks."""
        text = (
            "Outside citation [source1].\n"
            "```text\n"
            "inside_code [source2]\n"
            "```"
        )

        response = OutputParser._fallback_parse(text)

        doc_ids = [c.doc_id for c in response.citations]
        assert "source1" in doc_ids
        assert "source2" not in doc_ids

    def test_fallback_limits_citations(self):
        """Should limit number of extracted citations."""
        # Create text with many citations
        text = " ".join([f"[cite{i}]" for i in range(20)])

        response = OutputParser._fallback_parse(text)

        # Should limit to 10
        assert len(response.citations) <= 10

    def test_fallback_enriches_citations_with_documents(self):
        """Should enrich fallback citations with document metadata."""
        text = "Relevant context [doc_001] and supporting evidence [doc_002]."
        documents = [
            {
                "id": "doc_001",
                "title": "Python Guide",
                "url": "https://example.com/python",
                "source": "docs",
            },
            {
                "id": "doc_002",
                "title": "ML Guide",
                "url": "https://example.com/ml",
                "source": "docs",
            },
        ]

        response = OutputParser._fallback_parse(text, documents)

        assert len(response.citations) == 2
        assert {c.doc_id for c in response.citations} == {"doc_001", "doc_002"}
        assert {c.source for c in response.citations} == {"docs"}

    def test_citations_from_ids_validates_against_documents(self):
        """Should convert citation IDs to enriched Citation objects."""
        citation_ids = ["doc_001", "doc_fake", "doc_002"]
        documents = [
            {"id": "doc_001", "title": "Python Guide", "url": "https://example.com/python"},
            {"id": "doc_002", "title": "ML Guide", "url": "https://example.com/ml"},
        ]

        answer, citations = CitationExtractor.citations_from_ids("Python [doc_001] and ML [doc_002]", citation_ids, documents)

        assert answer == "Python [doc_001] and ML [doc_002]"
        assert len(citations) == 2
        assert {c.doc_id for c in citations} == {"doc_001", "doc_002"}


class TestOutputParserIntegration:
    """Integration tests for output parser."""

    def test_complete_parsing_flow(self):
        """Should handle complete parsing workflow."""
        # Simulate imperfect LLM output
        llm_output = '''```json
{
    'answer': "Python is a programing language used for AI [doc_001].",
    'citations': ['doc_001']
}
```'''

        documents = [
            {
                "id": "doc_001",
                "title": "Python Guide",
                "url": "https://example.com/python",
                "source": "docs",
            }
        ]

        response = OutputParser.parse(llm_output, documents)

        assert len(response.answer) > 0
        assert isinstance(response.citations, list)
        assert response.citations[0].title == "Python Guide"

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
