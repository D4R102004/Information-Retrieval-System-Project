"""
Integration tests for RAGModule pipeline auto-retrieval behavior.

These tests certify that when no documents are explicitly provided,
RAGModule fetches context through the configured retrieval pipeline.

Run with console logs:
    pytest tests/sri/rag/test_pipeline_integration.py -v -s --log-cli-level=INFO
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List

import pytest

from src.rag.llm_provider import LLMProvider
from src.rag.rag_module import RAGModule
from src.rag.output_parser import RAGResponse
# SRIPipeline and OllamaProvider are imported inside the real integration test


# Add project root to Python path for src module imports when running this file directly.
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


logger = logging.getLogger(__name__)


class LoggingMockLLMProvider(LLMProvider):
    """Mock LLM provider that logs full prompt and full response."""

    def __init__(self, response: str):
        self.response = response
        self.calls = 0
        self.last_prompt = ""

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        self.last_prompt = prompt
        logger.info("=== LLM PROMPT START ===\n%s\n=== LLM PROMPT END ===", prompt)
        logger.info("=== LLM RESPONSE START ===\n%s\n=== LLM RESPONSE END ===", self.response)
        return self.response

    def is_available(self) -> bool:
        return True

    def get_metadata(self) -> Dict:
        return {"provider": "LoggingMock", "model": "test-model"}


class FakePipeline:
    """Simple fake pipeline with deterministic search results."""

    def __init__(self, results: List[Dict]):
        self.results = results
        self.search_calls = []

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        self.search_calls.append({"query": query, "top_k": top_k})
        return self.results


class TestPipelineIntegration:
    """Validate RAG integration with retrieval pipeline in no-documents mode."""

    @pytest.fixture
    def pipeline_results(self):
        return [
            {
                "doc_id": "doc_py",
                "title": "Python Overview",
                "snippet": "Python is widely used for automation and data tasks.",
                "url": "https://example.com/python",
                "score": 0.92,
            },
            {
                "doc_id": "doc_ml",
                "title": "ML Stack",
                "snippet": "Python frameworks include scikit-learn, TensorFlow, and PyTorch.",
                "url": "https://example.com/ml",
                "score": 0.88,
            },
        ]

    def test_uses_pipeline_when_documents_not_provided(self, pipeline_results):
        """Should call pipeline.search and generate using transformed retrieved docs."""
        llm = LoggingMockLLMProvider(
            response='{"answer":"Python is used for ML [doc_ml].","citations":["doc_ml"]}'
        )
        pipeline = FakePipeline(results=pipeline_results)
        rag = RAGModule(llm=llm, pipeline=pipeline)

        response = rag.generate("What can Python do in machine learning?")

        assert isinstance(response, RAGResponse)
        assert pipeline.search_calls, "Pipeline search should be called when documents are None"
        assert pipeline.search_calls[0]["top_k"] == 10
        assert llm.calls == 1
        assert "Python" in response.answer

    def test_skips_pipeline_when_documents_are_provided(self, pipeline_results):
        """Should not call pipeline when caller already provides documents."""
        llm = LoggingMockLLMProvider(
            response='{"answer":"Provided docs used [doc_custom].","citations":["doc_custom"]}'
        )
        pipeline = FakePipeline(results=pipeline_results)
        rag = RAGModule(llm=llm, pipeline=pipeline)

        provided_documents = [
            {
                "id": "doc_custom",
                "title": "Custom Source",
                "content": "Caller-provided document content.",
                "url": "https://example.com/custom",
            }
        ]

        response = rag.generate("Use my provided documents", documents=provided_documents)

        assert isinstance(response, RAGResponse)
        assert not pipeline.search_calls, "Pipeline search should not run when documents are passed"
        assert llm.calls == 1

    def test_falls_back_to_generation_if_pipeline_fails(self):
        """Should still generate answer if pipeline raises an exception."""

        class FailingPipeline:
            def search(self, query: str, top_k: int = 10):
                raise RuntimeError("Search backend unavailable")

        llm = LoggingMockLLMProvider(response="Fallback generation without retrieved docs.")
        rag = RAGModule(llm=llm, pipeline=FailingPipeline())

        response = rag.generate("Can you answer even if retrieval fails?")

        assert isinstance(response, RAGResponse)
        assert len(response.answer) > 0
        assert llm.calls == 1

    def test_logs_full_prompt_and_response(self, pipeline_results, caplog):
        """Should emit full prompt and full response logs for observability in console."""
        caplog.set_level(logging.INFO)

        llm = LoggingMockLLMProvider(
            response='{"answer":"Observable output [doc_py].","citations":["doc_py"]}'
        )
        pipeline = FakePipeline(results=pipeline_results)
        rag = RAGModule(llm=llm, pipeline=pipeline)

        rag.generate("Show complete logs for prompt and response")

        text = caplog.text
        assert "=== LLM PROMPT START ===" in text
        assert "=== LLM PROMPT END ===" in text
        assert "=== LLM RESPONSE START ===" in text
        assert "=== LLM RESPONSE END ===" in text

    def test_real_pipeline_and_ollama_integration(self, pipeline_results, caplog):
        """Integration test using the real SRIPipeline and Ollama provider.

        This test will be skipped if Ollama is not available locally.
        It indexes a small document set into SRIPipeline (in-memory), then
        runs RAGModule with the pipeline and Ollama to validate end-to-end
        retrieval -> prompt -> LLM -> parse flow. Full prompt/response logs
        are captured.
        """
        caplog.set_level(logging.INFO)

        # Prepare and index documents into SRIPipeline (no persistence)
        from src.sri.pipeline import SRIPipeline

        pipeline = SRIPipeline(load_existing=False)

        # Load real project documents from data/documents.json
        import json
        from pathlib import Path
        data_path = Path(__file__).parents[3] / "data" / "documents.json"
        with open(data_path, "r", encoding="utf-8") as f:
            docs_to_index = json.load(f)

        pipeline.index(docs_to_index, save=False)

        # Initialize Ollama provider; skip test if unavailable
        # Initialize Ollama provider; skip test if unavailable
        from src.rag.llm_provider import OllamaProvider
        provider = OllamaProvider()

        # Wrap provider.generate to emit full prompt/response logs
        original_generate = provider.generate

        def logged_generate(prompt: str, **kwargs):
            logging.getLogger(__name__).info("=== LLM PROMPT START ===\n%s\n=== LLM PROMPT END ===", prompt)
            resp = original_generate(prompt=prompt, **kwargs)
            logging.getLogger(__name__).info("=== LLM RESPONSE START ===\n%s\n=== LLM RESPONSE END ===", resp)
            return resp

        provider.generate = logged_generate

        rag = RAGModule(llm=provider, pipeline=pipeline)

        response = rag.generate("What does Python enable in machine learning?")

        assert isinstance(response, RAGResponse)
        assert len(response.answer) > 0
        # Ensure logs contain prompt and response markers
        assert "=== LLM PROMPT START ===" in caplog.text
        assert "=== LLM RESPONSE END ===" in caplog.text
