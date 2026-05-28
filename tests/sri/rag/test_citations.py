"""
Tests for Citation extraction and enrichment.

Tests cover:
- Citation extraction from text
- Citation validation
- Citation enrichment with metadata
- Handling invalid citations
"""

from src.rag.citations import CitationExtractor


class TestCitationExtraction:
    """Test citation extraction from text."""

    def test_extract_single_citation(self):
        """Should extract single citation."""
        text = "Python is great [doc_001] for AI."
        documents = [{"id": "doc_001", "title": "Python"}]

        citations = CitationExtractor.extract_citations(text, documents)

        assert len(citations) == 1
        assert "doc_001" in citations

    def test_extract_multiple_citations(self):
        """Should extract multiple citations."""
        text = "Python [doc_001] is used for ML [doc_002] and AI [doc_003]."
        documents = [
            {"id": "doc_001"},
            {"id": "doc_002"},
            {"id": "doc_003"},
        ]

        citations = CitationExtractor.extract_citations(text, documents)

        assert len(citations) == 3
        assert "doc_001" in citations
        assert "doc_002" in citations
        assert "doc_003" in citations

    def test_extract_citations_deduplicates(self):
        """Should remove duplicate citations."""
        text = "Python [doc_001]. More Python [doc_001]. End [doc_001]."
        documents = [{"id": "doc_001"}]

        citations = CitationExtractor.extract_citations(text, documents)

        assert len(citations) == 1
        assert citations[0] == "doc_001"

    def test_extract_citations_invalid_ignored(self):
        """Should ignore citations not in documents."""
        text = "Python [doc_001] and [doc_invalid]."
        documents = [{"id": "doc_001"}]

        citations = CitationExtractor.extract_citations(text, documents)

        assert len(citations) == 1
        assert "doc_001" in citations
        assert "doc_invalid" not in citations

    def test_extract_no_citations(self):
        """Should return empty list when no citations."""
        text = "Python is a language. No citations here."
        documents = [{"id": "doc_001"}]

        citations = CitationExtractor.extract_citations(text, documents)

        assert len(citations) == 0

    def test_extract_citations_preserves_order(self):
        """Should preserve order of citations."""
        text = "[doc_c] then [doc_a] then [doc_b]."
        documents = [
            {"id": "doc_a"},
            {"id": "doc_b"},
            {"id": "doc_c"},
        ]

        citations = CitationExtractor.extract_citations(text, documents)

        assert citations == ["doc_c", "doc_a", "doc_b"]

    def test_extract_citations_various_ids(self):
        """Should handle various citation ID formats."""
        text = "Reference [001] and [retrieve_001] and [doc-mix_001]."
        documents = [
            {"id": "001"},
            {"id": "retrieve_001"},
            {"id": "doc-mix_001"},
        ]

        citations = CitationExtractor.extract_citations(text, documents)

        assert len(citations) == 3


class TestCitationEnrichment:
    """Test citation enrichment with metadata."""

    def test_enrich_single_citation(self):
        """Should enrich citation with document metadata."""
        citations = ["doc_001"]
        documents = [
            {
                "id": "doc_001",
                "title": "Python Guide",
                "url": "https://example.com/python",
                "source": "tutorial",
                "content": "Python is a programming language.",
            }
        ]

        enriched = CitationExtractor.enrich_citations(citations, documents)

        assert len(enriched) == 1
        assert enriched[0].doc_id == "doc_001"
        assert enriched[0].title == "Python Guide"
        assert enriched[0].url == "https://example.com/python"
        assert enriched[0].source == "tutorial"
        assert len(enriched[0].snippet) > 0

    def test_enrich_multiple_citations(self):
        """Should enrich multiple citations."""
        citations = ["doc_001", "doc_002"]
        documents = [
            {"id": "doc_001", "title": "Doc1", "content": "Content1"},
            {"id": "doc_002", "title": "Doc2", "content": "Content2"},
        ]

        enriched = CitationExtractor.enrich_citations(citations, documents)

        assert len(enriched) == 2
        assert enriched[0].title == "Doc1"
        assert enriched[1].title == "Doc2"

    def test_enrich_creates_snippet(self):
        """Should create snippet from content."""
        citations = ["doc_001"]
        long_content = "X" * 500
        documents = [
            {
                "id": "doc_001",
                "title": "Test",
                "content": long_content,
            }
        ]

        enriched = CitationExtractor.enrich_citations(citations, documents)

        snippet = enriched[0].snippet
        assert len(snippet) <= 200  # Snippet truncated to 200 chars

    def test_enrich_missing_fields(self):
        """Should handle documents with missing optional fields."""
        citations = ["doc_001"]
        documents = [
            {
                "id": "doc_001",
                "title": "Test",
                # Missing url, source, content
            }
        ]

        enriched = CitationExtractor.enrich_citations(citations, documents)

        assert enriched[0].title == "Test"
        assert enriched[0].url == ""
        assert enriched[0].source == "unknown"

    def test_enrich_optional_metadata(self):
        """Should include optional metadata if available."""
        citations = ["doc_001"]
        documents = [
            {
                "id": "doc_001",
                "title": "Test",
                "content": "Content",
                "score": 0.95,
                "date": "2024-01-15",
            }
        ]

        enriched = CitationExtractor.enrich_citations(citations, documents)

        assert enriched[0].score == 0.95
        assert enriched[0].date == "2024-01-15"

    def test_enrich_invalid_citation_skipped(self):
        """Should skip citations not found in documents."""
        citations = ["doc_001", "doc_invalid", "doc_002"]
        documents = [
            {"id": "doc_001", "title": "Doc1", "content": "Content1"},
            {"id": "doc_002", "title": "Doc2", "content": "Content2"},
        ]

        enriched = CitationExtractor.enrich_citations(citations, documents)

        assert len(enriched) == 2
        doc_ids = [c.doc_id for c in enriched]
        assert "doc_001" in doc_ids
        assert "doc_002" in doc_ids
        assert "doc_invalid" not in doc_ids


class TestCitationValidation:
    """Test citation validation."""

    def test_validate_all_valid(self):
        """Should validate all citations when they exist."""
        citations = ["doc_001", "doc_002"]
        documents = [
            {"id": "doc_001"},
            {"id": "doc_002"},
        ]

        valid, invalid = CitationExtractor.validate_citations(citations, documents)

        assert len(valid) == 2
        assert len(invalid) == 0

    def test_validate_some_invalid(self):
        """Should identify invalid citations."""
        citations = ["doc_001", "doc_invalid", "doc_002"]
        documents = [
            {"id": "doc_001"},
            {"id": "doc_002"},
        ]

        valid, invalid = CitationExtractor.validate_citations(citations, documents)

        assert len(valid) == 2
        assert len(invalid) == 1
        assert "doc_invalid" in invalid

    def test_validate_all_invalid(self):
        """Should handle all invalid citations."""
        citations = ["doc_bad1", "doc_bad2"]
        documents = [
            {"id": "doc_001"},
            {"id": "doc_002"},
        ]

        valid, invalid = CitationExtractor.validate_citations(citations, documents)

        assert len(valid) == 0
        assert len(invalid) == 2

    def test_validate_empty(self):
        """Should handle empty citation list."""
        citations = []
        documents = [{"id": "doc_001"}]

        valid, invalid = CitationExtractor.validate_citations(citations, documents)

        assert len(valid) == 0
        assert len(invalid) == 0


class TestCitationExtractorIntegration:
    """Integration tests for citation extraction workflow."""

    def test_extract_and_enrich_workflow(self):
        """Should extract and enrich citations in sequence."""
        text = "Python [doc_001] is great. ML [doc_002] uses it."
        documents = [
            {
                "id": "doc_001",
                "title": "Python Basics",
                "content": "Python is...",
                "url": "https://python.org",
            },
            {
                "id": "doc_002",
                "title": "ML Guide",
                "content": "ML uses Python...",
                "url": "https://ml.org",
            },
        ]

        # Extract
        citations = CitationExtractor.extract_citations(text, documents)
        assert len(citations) == 2

        # Validate
        valid, invalid = CitationExtractor.validate_citations(citations, documents)
        assert len(invalid) == 0

        # Enrich
        enriched = CitationExtractor.enrich_citations(valid, documents)
        assert len(enriched) == 2
        assert enriched[0].title == "Python Basics"
        assert enriched[1].title == "ML Guide"
