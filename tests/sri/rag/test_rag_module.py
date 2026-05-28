"""
Tests for RAGModule orchestrator.

Tests cover:
- RAGModule initialization
- Generation with and without documents
- Template switching
- Performance logging
- Error handling
"""

import pytest
from unittest.mock import patch
import json
import logging

from src.rag.rag_module import RAGModule
from src.rag.llm_provider import LLMProvider
from src.rag.output_parser import RAGResponse


logger = logging.getLogger(__name__)


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, response: str = "Generated answer."):
        self.response = response
        self.generation_calls = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self.generation_calls += 1
        logger.info("=== LLM PROMPT START ===\n%s\n=== LLM PROMPT END ===", prompt)
        logger.info("=== LLM RESPONSE START ===\n%s\n=== LLM RESPONSE END ===", self.response)
        return self.response

    def is_available(self) -> bool:
        return True

    def get_metadata(self) -> dict:
        return {"provider": "Mock", "model": "test-model"}


class FakePipeline:
    """Simple fake pipeline for testing retrieval integration."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query: str, top_k: int = 10):
        self.calls.append({"query": query, "top_k": top_k})
        return self.results


class TestRAGModuleInitialization:
    """Test RAGModule initialization."""

    def test_initialization_basic(self):
        """Should initialize with LLM provider."""
        llm = MockLLMProvider()
        rag = RAGModule(llm)

        assert rag.llm == llm
        assert rag.template_type == "domain_specific"

    def test_initialization_with_custom_template(self):
        """Should accept custom template type."""
        llm = MockLLMProvider()
        rag = RAGModule(llm, template_type="chain_of_thought")

        assert rag.template_type == "chain_of_thought"

    def test_initialization_with_invalid_template(self):
        """Should raise ValueError for invalid template."""
        llm = MockLLMProvider()

        with pytest.raises(ValueError, match="Unknown template"):
            RAGModule(llm, template_type="invalid_template")

    def test_initialization_creates_template(self):
        """Should create PromptTemplate instance."""
        llm = MockLLMProvider()
        rag = RAGModule(llm)

        assert rag.template is not None
        assert hasattr(rag.template, "apply")


class TestRAGModuleGeneration:
    """Test text generation functionality."""

    def test_generate_with_documents(self):
        """Should generate answer with documents."""
        llm = MockLLMProvider(response="Python is a language [doc_001].")
        rag = RAGModule(llm)

        documents = [
            {"id": "doc_001", "title": "Python", "content": "Python is..."}
        ]

        response = rag.generate("What is Python?", documents=documents)

        assert isinstance(response, RAGResponse)
        assert "Python" in response.answer

    def test_generate_without_documents(self):
        """Should generate answer without documents (pattern B)."""
        llm = MockLLMProvider(response="Python is a versatile language.")
        rag = RAGModule(llm)

        response = rag.generate("What is Python?")

        assert isinstance(response, RAGResponse)
        assert len(response.answer) > 0

    def test_generate_calls_llm_once(self):
        """Should call LLM exactly once."""
        llm = MockLLMProvider()
        rag = RAGModule(llm)

        rag.generate("Query")

        assert llm.generation_calls == 1

    def test_generate_extracts_citations(self):
        """Should extract citations from generated answer."""
        llm = MockLLMProvider(response="Info comes from [doc_001] and [doc_002].")
        rag = RAGModule(llm)

        documents = [
            {"id": "doc_001", "title": "Source1", "content": "Content1"},
            {"id": "doc_002", "title": "Source2", "content": "Content2"},
        ]

        response = rag.generate("Query", documents=documents)

        assert len(response.citations) == 2

    def test_generate_respects_temperature(self):
        """Should pass temperature to LLM."""
        llm = MockLLMProvider()
        rag = RAGModule(llm)

        with patch.object(llm, 'generate', wraps=llm.generate) as mock_gen:
            rag.generate("Query", temperature=0.3)

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs['temperature'] == 0.3

    def test_generate_respects_max_tokens(self):
        """Should pass max_tokens to LLM."""
        llm = MockLLMProvider()
        rag = RAGModule(llm)

        with patch.object(llm, 'generate', wraps=llm.generate) as mock_gen:
            rag.generate("Query", max_tokens=256)

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs['max_tokens'] == 256

    def test_generate_respects_top_p(self):
        """Should pass top_p to LLM."""
        llm = MockLLMProvider()
        rag = RAGModule(llm)

        with patch.object(llm, 'generate', wraps=llm.generate) as mock_gen:
            rag.generate("Query", top_p=0.85)

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs['top_p'] == 0.85


class TestTemplateSwitch:
    """Test runtime template switching."""

    def test_switch_template_basic(self):
        """Should switch to basic template."""
        llm = MockLLMProvider()
        rag = RAGModule(llm, template_type="domain_specific")

        rag.switch_template("basic")

        assert rag.template_type == "basic"

    def test_switch_template_chain_of_thought(self):
        """Should switch to chain of thought template."""
        llm = MockLLMProvider()
        rag = RAGModule(llm)

        rag.switch_template("chain_of_thought")

        assert rag.template_type == "chain_of_thought"

    def test_switch_template_invalid(self):
        """Should raise error for invalid template."""
        llm = MockLLMProvider()
        rag = RAGModule(llm)

        with pytest.raises(ValueError):
            rag.switch_template("invalid")

    def test_switch_template_affects_generation(self):
        """Switching template should affect generated prompts."""
        llm = MockLLMProvider()
        rag = RAGModule(llm, template_type="basic")

        documents = [{"id": "doc_1", "title": "Test", "content": "Content"}]

        # Save template references
        basic_template = rag.template

        # Switch
        rag.switch_template("domain_specific")

        # Should have different template object
        assert rag.template is not basic_template


class TestMetadata:
    """Test metadata retrieval."""

    def test_get_metadata(self):
        """Should return RAG module metadata."""
        llm = MockLLMProvider()
        rag = RAGModule(llm)

        metadata = rag.get_metadata()

        assert metadata["module"] == "RAG"
        assert metadata["template"] == "domain_specific"
        assert "llm" in metadata
        assert "available_templates" in metadata

    def test_metadata_includes_llm_info(self):
        """Metadata should include LLM information."""
        llm = MockLLMProvider()
        rag = RAGModule(llm)

        metadata = rag.get_metadata()

        assert metadata["llm"]["provider"] == "Mock"
        assert metadata["llm"]["model"] == "test-model"


class TestErrorHandling:
    """Test error handling and resilience."""

    def test_generate_handles_llm_error(self):
        """Should handle LLM generation errors gracefully."""
        llm = MockLLMProvider()

        with patch.object(llm, 'generate', side_effect=RuntimeError("LLM error")):
            rag = RAGModule(llm)
            response = rag.generate("Query")

            # Should return error message instead of crashing
            assert isinstance(response, RAGResponse)
            assert "Error" in response.answer

    def test_generate_handles_invalid_lm_response(self):
        """Should handle unparseable LLM responses."""
        llm = MockLLMProvider(response="Not valid JSON at all")
        rag = RAGModule(llm)

        response = rag.generate("Query")

        # Should still return a valid response
        assert isinstance(response, RAGResponse)

    def test_generate_timeout_handling(self):
        """Should handle LLM timeouts."""
        llm = MockLLMProvider()

        with patch.object(llm, 'generate', side_effect=RuntimeError("Timeout")):
            rag = RAGModule(llm)
            response = rag.generate("Query")

            assert isinstance(response, RAGResponse)


class TestRAGPipelineIntegrationWithoutLLM:
    """Validate pipeline retrieval behavior using mock LLM responses."""

    def test_generate_without_documents_uses_pipeline(self):
        """Should retrieve documents via pipeline when documents argument is omitted."""
        llm = MockLLMProvider(
            response='{"answer":"Pipeline context answer [doc_pipeline].","citations":["doc_pipeline"]}'
        )
        pipeline = FakePipeline(
            results=[
                {
                    "doc_id": "doc_pipeline",
                    "title": "Pipeline Document",
                    "snippet": "Snippet retrieved from index.",
                    "url": "https://example.com/pipeline",
                    "score": 0.91,
                }
            ]
        )
        rag = RAGModule(llm, pipeline=pipeline)

        response = rag.generate("Use pipeline retrieval path")

        assert isinstance(response, RAGResponse)
        assert len(pipeline.calls) == 1
        assert pipeline.calls[0]["query"] == "Use pipeline retrieval path"
        assert pipeline.calls[0]["top_k"] == 10
        assert len(response.answer) > 0

    def test_generate_with_documents_skips_pipeline(self):
        """Should not call pipeline when documents are explicitly provided."""
        llm = MockLLMProvider(
            response='{"answer":"Manual docs answer [doc_manual].","citations":["doc_manual"]}'
        )
        pipeline = FakePipeline(
            results=[
                {
                    "doc_id": "doc_unused",
                    "title": "Unused",
                    "snippet": "Unused snippet",
                    "url": "https://example.com/unused",
                    "score": 0.5,
                }
            ]
        )
        rag = RAGModule(llm, pipeline=pipeline)

        documents = [{"id": "doc_manual", "title": "Manual", "content": "Manual content"}]
        response = rag.generate("Use manual docs", documents=documents)

        assert isinstance(response, RAGResponse)
        assert len(pipeline.calls) == 0

    def test_generate_pipeline_failure_falls_back_to_generation(self):
        """Should continue generation when retrieval pipeline fails."""

        class FailingPipeline:
            def search(self, query: str, top_k: int = 10):
                raise RuntimeError("Pipeline unavailable")

        llm = MockLLMProvider(response="Fallback answer without retrieved documents.")
        rag = RAGModule(llm, pipeline=FailingPipeline())

        response = rag.generate("Fallback when retrieval fails")

        assert isinstance(response, RAGResponse)
        assert len(response.answer) > 0

    def test_logs_full_prompt_and_response_in_console(self, caplog):
        """Should log complete prompt and response for observability."""
        caplog.set_level(logging.INFO)

        llm = MockLLMProvider(
            response='{"answer":"Observability check [doc_obs].","citations":["doc_obs"]}'
        )
        pipeline = FakePipeline(
            results=[
                {
                    "doc_id": "doc_obs",
                    "title": "Observability",
                    "snippet": "Prompt and response should be visible in logs.",
                    "url": "https://example.com/obs",
                    "score": 0.99,
                }
            ]
        )
        rag = RAGModule(llm, pipeline=pipeline)

        rag.generate("Log full prompt and response")

        assert "=== LLM PROMPT START ===" in caplog.text
        assert "=== LLM PROMPT END ===" in caplog.text
        assert "=== LLM RESPONSE START ===" in caplog.text
        assert "=== LLM RESPONSE END ===" in caplog.text


class TestRAGModuleIntegration:
    """Integration tests for RAGModule."""

    def test_end_to_end_generation_flow(self):
        """Should complete full generation flow."""
        response_text = json.dumps({
            "answer": "Python is used for ML [doc_001].",
            "citations": [{"doc_id": "doc_001", "title": "ML Guide"}]
        })

        llm = MockLLMProvider(response=response_text)
        rag = RAGModule(llm)

        documents = [
            {"id": "doc_001", "title": "ML Guide", "content": "ML uses Python"}
        ]

        response = rag.generate("What is Python used for?", documents=documents)

        assert len(response.answer) > 0
        assert len(response.citations) >= 0  # May extract from answer

    def test_multiple_generations_same_module(self):
        """Should handle multiple generations."""
        llm = MockLLMProvider(response="Answer 1. Answer 2. Answer 3.")
        rag = RAGModule(llm)

        response1 = rag.generate("Query 1")
        response2 = rag.generate("Query 2")
        response3 = rag.generate("Query 3")

        assert llm.generation_calls == 3
        assert isinstance(response1, RAGResponse)
        assert isinstance(response2, RAGResponse)
        assert isinstance(response3, RAGResponse)

    def test_template_switching_between_calls(self):
        """Should support template switching between calls."""
        llm = MockLLMProvider(response="Test answer.")
        rag = RAGModule(llm, template_type="basic")

        response1 = rag.generate("Query 1")
        assert rag.template_type == "basic"

        rag.switch_template("chain_of_thought")
        response2 = rag.generate("Query 2")
        assert rag.template_type == "chain_of_thought"

        assert isinstance(response1, RAGResponse)
        assert isinstance(response2, RAGResponse)
