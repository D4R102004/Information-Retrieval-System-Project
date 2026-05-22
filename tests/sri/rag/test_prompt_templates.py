"""
Tests for Prompt Template module.

Tests cover:
- Basic template generation
- Domain-specific template with role-play
- Chain-of-Thought reasoning template
- Template factory and registration
"""

import pytest
from typing import List
from src.rag.prompt_templates import (
    PromptTemplate,
    BasicTemplate,
    DomainSpecificTemplate,
    ChainOfThoughtTemplate,
    PromptTemplateFactory,
)


class TestBasicTemplate:
    """Test BasicTemplate functionality."""

    def test_basic_template_applies(self):
        """BasicTemplate should generate simple prompt."""
        template = BasicTemplate()
        query = "How does Python work?"
        documents = [
            {"id": "doc_1", "title": "Python Basics", "content": "Python is great."}
        ]

        prompt = template.apply(query, documents)[0]

        assert query in prompt
        assert "doc_1" in prompt
        assert "Python Basics" in prompt

    def test_basic_template_empty_docs(self):
        """BasicTemplate should handle empty document list."""
        template = BasicTemplate()
        prompt = template.apply("What is AI?", [])[0]

        assert "What is AI?" in prompt
        assert "Answer:" in prompt

    def test_basic_template_context_truncation(self):
        """BasicTemplate should truncate context to max chars."""
        template = BasicTemplate()
        long_content = "x" * 5000  # Longer than max_chars

        documents = [
            {"id": "doc_1", "title": "Long Doc", "content": long_content}
        ]

        prompt = template.apply("Query", documents)[0]

        # Context should be truncated
        assert len(prompt) < len(long_content) + 200


class TestDomainSpecificTemplate:
    """Test DomainSpecificTemplate functionality."""

    def test_domain_specific_template_applies(self):
        """DomainSpecificTemplate should include role definition."""
        template = DomainSpecificTemplate()
        query = "Explain machine learning"
        documents = [
            {"id": "doc_2", "title": "ML Guide", "content": "ML uses algorithms..."}
        ]

        prompt = template.apply(query, documents)[0]

        assert "technical assistant" in prompt.lower()
        assert "cite" in prompt.lower()
        assert query in prompt

    def test_domain_specific_system_prompt(self):
        """Should include system prompt."""
        template = DomainSpecificTemplate()

        assert "technical assistant" in template.system_prompt.lower()

    def test_domain_specific_citation_instructions(self):
        """Should include citation format instructions."""
        template = DomainSpecificTemplate()
        documents = [{"id": "doc_1", "title": "Test", "content": "Content"}]

        prompt = template.apply("Query", documents)[0]

        assert "[doc_id]" in prompt.lower() or "[" in prompt
        assert "cite" in prompt.lower()


class TestChainOfThoughtTemplate:
    """Test ChainOfThoughtTemplate functionality."""

    def test_chain_of_thought_applies(self):
        """ChainOfThoughtTemplate should include reasoning steps."""
        template = ChainOfThoughtTemplate()
        query = "How does retrieval work?"
        documents = [
            {
                "id": "doc_3",
                "title": "Retrieval Methods",
                "content": "Methods include BM25...",
            }
        ]

        prompt = template.apply(query, documents)[0]

        assert query in prompt
        # Should include reasoning steps
        assert "step" in prompt.lower() or "reasoning" in prompt.lower()

    def test_chain_of_thought_document_analysis(self):
        """Should include document analysis instruction."""
        template = ChainOfThoughtTemplate()
        documents = [{"id": "doc_1", "title": "Test", "content": "Content"}]

        prompt = template.apply("Query", documents)[0]

        # Should mention analyzing documents
        lower_prompt = prompt.lower()
        assert (
            "relevance" in lower_prompt
            or "analysis" in lower_prompt
            or "document" in lower_prompt
        )


class TestPromptTemplateFactory:
    """Test template factory functionality."""

    def test_factory_create_basic(self):
        """Factory should create BasicTemplate."""
        template = PromptTemplateFactory.create("basic")

        assert isinstance(template, BasicTemplate)

    def test_factory_create_domain_specific(self):
        """Factory should create DomainSpecificTemplate."""
        template = PromptTemplateFactory.create("domain_specific")

        assert isinstance(template, DomainSpecificTemplate)

    def test_factory_create_chain_of_thought(self):
        """Factory should create ChainOfThoughtTemplate."""
        template = PromptTemplateFactory.create("chain_of_thought")

        assert isinstance(template, ChainOfThoughtTemplate)

    def test_factory_unknown_template(self):
        """Factory should raise ValueError for unknown template."""
        with pytest.raises(ValueError, match="Unknown template"):
            PromptTemplateFactory.create("unknown_template")

    def test_factory_available_templates(self):
        """Factory should list available templates."""
        templates = PromptTemplateFactory.available_templates()

        assert "basic" in templates
        assert "domain_specific" in templates
        assert "chain_of_thought" in templates

    def test_factory_register_custom(self):
        """Factory should support registering custom templates."""

        class CustomTemplate(PromptTemplate):
            def apply(self, query: str, documents, require_json: bool = False) -> List[str]:
                return [f"Custom: {query}{f"\n{self.json_response}" if require_json else ""}"]

        PromptTemplateFactory.register("custom", CustomTemplate)

        assert "custom" in PromptTemplateFactory.available_templates()
        custom_instance = PromptTemplateFactory.create("custom")
        assert isinstance(custom_instance, CustomTemplate)

    def test_factory_register_invalid_class(self):
        """Factory should reject non-PromptTemplate classes."""

        class NotATemplate:
            pass

        with pytest.raises(TypeError):
            PromptTemplateFactory.register("bad", NotATemplate)


class TestFormatContext:
    """Test context formatting."""

    def test_format_context_single_document(self):
        """Should format single document correctly."""
        template = BasicTemplate()
        documents = [
            {
                "id": "doc_1",
                "title": "Python Guide",
                "content": "Python is a language.",
            }
        ]

        context = template._format_context(documents)

        assert "doc_1" in context
        assert "Python Guide" in context
        assert "Python is a language" in context

    def test_format_context_multiple_documents(self):
        """Should format multiple documents."""
        template = BasicTemplate()
        documents = [
            {"id": "doc_1", "title": "Doc1", "content": "Content1"},
            {"id": "doc_2", "title": "Doc2", "content": "Content2"},
        ]

        context = template._format_context(documents)

        assert "doc_1" in context
        assert "doc_2" in context

    def test_format_context_respects_max_chars(self):
        """Should respect maximum character limit."""
        template = BasicTemplate()
        documents = [
            {"id": "doc_1", "title": "D1", "content": "X" * 3000},
            {"id": "doc_2", "title": "D2", "content": "Y" * 3000},
        ]

        context = template._format_context(documents, max_chars=1000)

        assert len(context) <= 1000

    def test_format_context_missing_fields(self):
        """Should handle documents with missing fields."""
        template = BasicTemplate()
        documents = [
            {"id": "doc_1"},  # Missing title and content
        ]

        context = template._format_context(documents)

        assert "doc_1" in context
        # Should not crash

    def test_format_context_truncates_content(self):
        """Should truncate content to 500 chars."""
        template = BasicTemplate()
        long_content = "X" * 1000
        documents = [
            {
                "id": "doc_1",
                "title": "Test",
                "content": long_content,
            }
        ]

        context = template._format_context(documents)

        # Content should be truncated in context
        assert len(context) < len(long_content)
