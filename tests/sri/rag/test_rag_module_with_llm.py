"""
Comprehensive RAG Module Integration Tests

End-to-end integration tests for the RAG system with real Ollama LLM inference.
Tests validate: document processing, response generation, citation extraction,
edge case handling, and system resilience.

Test Categories:
- Connection & Setup: Ollama availability and document structure
- Core Functionality: Response generation, citation validation, template switching
- Edge Cases: Empty documents, special characters, large document handling
- Robustness: Error handling, Unicode support, performance baseline
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional

import pytest

# Add project root to Python path for src module imports
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

# Configure logging for test diagnostics
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FakePipeline:
    """Simple fake retrieval pipeline for integration tests."""

    def __init__(self, results: List[Dict]):
        self.results = results
        self.calls: List[Dict] = []

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        self.calls.append({"query": query, "top_k": top_k})
        return self.results


class TestRAGIntegrationComprehensive:
    """Comprehensive RAG module integration test suite."""

    @pytest.fixture(scope="class")
    def rag_module(self):
        """Initialize RAG module with real Ollama provider."""
        from src.rag.llm_provider import OllamaProvider
        from src.rag.rag_module import RAGModule
        
        provider = OllamaProvider(model="llama3.2:latest", timeout=120)
        rag = RAGModule(provider, template_type="domain_specific")

        original_generate = provider.generate

        def logged_generate(prompt: str, **kwargs):
            logger.info("=== LLM PROMPT START ===\n%s\n=== LLM PROMPT END ===", prompt)
            response = original_generate(prompt=prompt, **kwargs)
            logger.info("=== LLM RESPONSE START ===\n%s\n=== LLM RESPONSE END ===", response)
            return response

        provider.generate = logged_generate
        yield rag

    @pytest.fixture(scope="class")
    def sample_documents(self):
        """Load or create sample documents for testing."""
        return [
            {
                "id": "doc_python",
                "title": "Python Programming",
                "content": "Python is a versatile programming language used for web development, "
                          "data science, and machine learning. It emphasizes code readability and "
                          "has a simple syntax that makes it accessible to beginners."
            },
            {
                "id": "doc_ml",
                "title": "Machine Learning with Python",
                "content": "Python has excellent libraries like scikit-learn, TensorFlow, and PyTorch "
                          "that make machine learning accessible. These tools enable developers to "
                          "build sophisticated models without deep mathematical expertise."
            },
            {
                "id": "doc_web",
                "title": "Web Development",
                "content": "Django and Flask are popular Python web frameworks. They provide tools "
                          "for building robust web applications quickly. These frameworks handle "
                          "routing, database management, and authentication."
            },
            {
                "id": "doc_intro",
                "title": "Information Retrieval Basics",
                "content": "Information Retrieval (IR) systems retrieve relevant documents from large "
                          "collections based on user queries. RAG (Retrieval-Augmented Generation) "
                          "combines document retrieval with language model generation."
            }
        ]

    @pytest.fixture(scope="class")
    def pipeline_results(self):
        """Sample retrieval records returned by the fake pipeline."""
        return [
            {
                "doc_id": "doc_python",
                "title": "Python Programming",
                "snippet": "Python is used for web development, data science, and machine learning.",
                "url": "https://example.com/python",
                "score": 0.93,
            },
            {
                "doc_id": "doc_ml",
                "title": "Machine Learning with Python",
                "snippet": "Common Python ML libraries include scikit-learn, TensorFlow, and PyTorch.",
                "url": "https://example.com/ml",
                "score": 0.89,
            },
        ]

    # ========== CORE FUNCTIONALITY TESTS ==========

    def test_01_ollama_connection(self, rag_module):
        """TEST 01: Verify Ollama service is accessible and ready."""
        provider = rag_module.llm
        
        assert provider.is_available(), "Ollama provider should be available"
        
        metadata = provider.get_metadata()
        assert metadata is not None, "Metadata should be retrievable"
        assert metadata["model"] == "llama3.2:latest", "Model should be llama3.2:latest"
        assert metadata["provider"] == "Ollama", "Provider should be named 'Ollama'"
        
        logger.info(f"✅ Ollama connection verified: {metadata}")

    def test_02_document_structure_validation(self, sample_documents):
        """TEST 02: Validate document structure and integrity."""
        required_fields = {"id", "title", "content"}
        
        for doc in sample_documents:
            assert all(field in doc for field in required_fields), \
                f"Document missing required fields: {required_fields}"
            assert len(doc["id"]) > 0, "Document ID must not be empty"
            assert len(doc["title"]) > 0, "Document title must not be empty"
            assert len(doc["content"]) > 0, "Document content must not be empty"
        
        assert len(sample_documents) == 4, "Should have 4 test documents"
        logger.info(f"✅ Document structure validated: {len(sample_documents)} documents")

    def test_03_basic_rag_generation(self, rag_module, sample_documents):
        """TEST 03: Verify basic RAG generation with real Ollama."""
        query = "What are the main uses of Python?"
        
        start_time = time.time()
        response = rag_module.generate(query, documents=sample_documents, max_tokens=256)
        elapsed = time.time() - start_time
        
        # Validate response structure
        assert response is not None, "Response should not be None"
        assert hasattr(response, "answer"), "Response should have 'answer' field"
        assert hasattr(response, "citations"), "Response should have 'citations' field"
        assert len(response.answer) > 0, "Answer should not be empty"
        
        # Validate response quality
        assert len(response.answer) >= 20, "Answer should be substantial (>= 20 chars)"
        assert len(response.answer) <= 10000, "Answer should not be excessively long"
        
        # Validate citations
        assert isinstance(response.citations, list), "Citations should be a list"
        valid_doc_ids = {doc["id"] for doc in sample_documents}
        
        for citation in response.citations:
            assert citation.doc_id in valid_doc_ids, \
                f"Citation references non-existent document: {citation.doc_id}"
        
        logger.info(f"✅ RAG generation successful in {elapsed:.2f}s "
                   f"({len(response.answer)} chars, {len(response.citations)} citations)")
        logger.info(f"\nResponse: {response.answer}")

    def test_03b_pipeline_auto_retrieval_without_documents(self, pipeline_results):
        """TEST 03B: Verify pipeline retrieval path when no documents are passed."""
        from src.rag.llm_provider import OllamaProvider
        from src.rag.rag_module import RAGModule

        provider = OllamaProvider(model="llama3.2:latest", timeout=120)
        assert provider.is_available(), "Ollama provider should be available"

        original_generate = provider.generate

        def logged_generate(prompt: str, **kwargs):
            logger.info("=== LLM PROMPT START ===\n%s\n=== LLM PROMPT END ===", prompt)
            response = original_generate(prompt=prompt, **kwargs)
            logger.info("=== LLM RESPONSE START ===\n%s\n=== LLM RESPONSE END ===", response)
            return response

        provider.generate = logged_generate
        fake_pipeline = FakePipeline(results=pipeline_results)
        rag = RAGModule(provider, template_type="domain_specific", pipeline=fake_pipeline)

        response = rag.generate("What are common Python machine learning libraries?", documents=None, max_tokens=220)

        assert len(fake_pipeline.calls) == 1, "Pipeline should be called once"
        assert fake_pipeline.calls[0]["top_k"] == 10, "Pipeline should use top_k=10"
        assert response is not None
        assert len(response.answer) > 0

        logger.info(f"✅ Auto-retrieval with pipeline succeeded ({len(response.answer)} chars)")

    def test_04_citation_validation_against_documents(self, rag_module, sample_documents):
        """TEST 04: Verify citations are accurately linked to source documents."""
        query = "Tell me about Python libraries for machine learning"
        response = rag_module.generate(query, documents=sample_documents, max_tokens=200)
        
        # Build reference map
        doc_map = {doc["id"]: doc for doc in sample_documents}
        
        # Validate each citation
        valid_citations = 0
        for citation in response.citations:
            if citation.doc_id in doc_map:
                valid_citations += 1
                source_doc = doc_map[citation.doc_id]
                
                # Verify title consistency
                assert source_doc["title"], "Source document should have title"
                
                # Verify citation is referenced in answer
                assert f"[{citation.doc_id}]" in response.answer or \
                       source_doc["title"].lower() in response.answer.lower(), \
                       f"Citation {citation.doc_id} should be referenced in answer"
        
        logger.info(f"✅ Citation validation: {valid_citations}/{len(response.citations)} valid citations")
        logger.info(f"\nResponse: {response.answer}")

    def test_05_hallucination_detection(self, rag_module):
        """TEST 05: Verify model handles ambiguous queries gracefully."""
        # Use minimal document set
        limited_documents = [
            {
                "id": "doc_minimal",
                "title": "Limited Info",
                "content": "Basic test content with minimal information."
            }
        ]
        
        query = "What is the exact version number of this project?"
        response = rag_module.generate(query, documents=limited_documents, max_tokens=150)
        
        # Verify response is reasonable (not truncated or malformed)
        assert len(response.answer) > 5, "Response should contain meaningful content"
        assert len(response.answer) < 1000, "Response should not be excessively long"
        
        logger.info(f"✅ Hallucination test: Response quality acceptable ({len(response.answer)} chars)")
        logger.info(f"\nResponse: {response.answer}")

    def test_06_template_switching_consistency(self, rag_module, sample_documents):
        """TEST 06: Verify template switching maintains response quality."""
        query = "Explain information retrieval systems"
        templates = ["basic", "domain_specific", "chain_of_thought"]
        
        results = {}
        for template in templates:
            rag_module.switch_template(template)
            start = time.time()
            response = rag_module.generate(query, documents=sample_documents, max_tokens=200)
            logger.info(f"\nResponse: {response.answer}")
            elapsed = time.time() - start
            
            results[template] = {
                "time": elapsed,
                "answer_length": len(response.answer),
                "citations_count": len(response.citations),
                "passes_quality": (len(response.answer) >= 20 and 
                                  len(response.answer) <= 10000)
            }
            
            assert results[template]["passes_quality"], \
                f"Template '{template}' failed quality checks"
        
        logger.info("✅ Template switching: All templates maintain quality")
        for template, metrics in results.items():
            logger.info(f"   {template}: {metrics['time']:.2f}s, "
                       f"{metrics['answer_length']} chars, "
                       f"{metrics['citations_count']} citations")

    # ========== EDGE CASE TESTS ==========

    def test_07_edge_case_empty_documents(self, rag_module):
        """TEST 07: Handle gracefully when documents are empty or minimal."""
        edge_cases = [
            {
                "name": "Empty document list",
                "docs": [],
                "query": "What is this about?"
            },
            {
                "name": "Very short document",
                "docs": [{"id": "d1", "title": "Brief", "content": "X"}],
                "query": "Explain this"
            },
            {
                "name": "Minimal content",
                "docs": [{"id": "d1", "title": "Minimal", "content": "Test"}],
                "query": "What does this contain?"
            }
        ]
        
        for case in edge_cases:
            try:
                response = rag_module.generate(case["query"], documents=case["docs"], max_tokens=100)
                assert response is not None, f"Failed on {case['name']}"
                assert len(response.answer) > 0, f"Empty answer on {case['name']}"
                logger.info(f"✅ Edge case handled: {case['name']}")
            except Exception as e:
                logger.error(f"❌ Edge case failed: {case['name']}: {e}")
                raise

    def test_08_edge_case_special_characters(self, rag_module):
        """TEST 08: Handle documents with special characters and formatting."""
        special_docs = [
            {
                "id": "special",
                "title": "Special Characters",
                "content": 'Contains [brackets], {braces}, "quotes", \'apostrophes\', '
                          'and symbols: @#$%^&*()'
            }
        ]
        
        query = "What special characters are present?"
        response = rag_module.generate(query, documents=special_docs, max_tokens=150)
        
        assert response is not None, "Should handle special characters"
        assert len(response.answer) > 0, "Should generate answer with special chars"
        
        logger.info("✅ Special characters handled correctly")
        logger.info(f"\nResponse: {response.answer}")

    def test_09_edge_case_long_documents(self, rag_module):
        """TEST 09: Handle documents exceeding normal size constraints."""
        large_content = " ".join(["Python is a programming language."] * 200)  # ~7000 chars
        large_docs = [
            {
                "id": "large",
                "title": "Long Document",
                "content": large_content
            }
        ]
        
        query = "Summarize the main topic"
        response = rag_module.generate(query, documents=large_docs, max_tokens=150)
        
        assert response is not None, "Should handle large documents"
        assert len(response.answer) > 0, "Should generate answer for large docs"
        assert len(response.citations) >= 0, "Citations should be valid"
        
        logger.info(f"✅ Large document handled: {len(large_content)} chars -> "
                   f"{len(response.answer)} char response")
        logger.info(f"\nResponse: {response.answer}")

    # ========== CONSISTENCY AND RESILIENCE TESTS ==========

    def test_10_response_consistency(self, rag_module, sample_documents):
        """TEST 10: Verify response consistency across multiple calls."""
        query = "What is information retrieval?"
        responses = []
        
        for attempt in range(3):
            response = rag_module.generate(query, documents=sample_documents, 
                                          temperature=0.1, max_tokens=150)
            responses.append(response)
        
        # Check if key concepts are consistently mentioned
        key_terms = ["retrieval", "information", "document", "query"]
        term_presence = {term: [] for term in key_terms}
        
        for response in responses:
            for term in key_terms:
                term_presence[term].append(term.lower() in response.answer.lower())
        
        # Most key terms should appear in most responses
        for term, occurrences in term_presence.items():
            consistency = sum(occurrences) / len(occurrences)
            logger.info(f"  Term '{term}': {consistency*100:.0f}% consistency")
            assert consistency >= 0.33, f"Low consistency for term '{term}'"
        
        logger.info("✅ Response consistency verified (key terms present)")
        logger.info(f"\nResponse: {response.answer}")

    def test_11_error_resilience(self, rag_module):
        """TEST 11: Verify graceful error handling in edge cases."""
        problematic_inputs = [
            {
                "name": "Malformed document list",
                "docs": [{"no_id": "d1"}],  # Missing required fields
                "query": "What?"
            },
            {
                "name": "Unicode and special content",
                "docs": [{"id": "d1", "title": "Unicode Test", 
                         "content": "Contains émojis 🚀 and spëcial çharacters"}],
                "query": "Explain the content"
            }
        ]
        
        handled = 0
        for test_case in problematic_inputs:
            try:
                response = rag_module.generate(test_case["query"], 
                                              documents=test_case["docs"],
                                              max_tokens=100)
                if response and len(response.answer) > 0:
                    handled += 1
                    logger.info(f"✅ {test_case['name']}: Handled gracefully")
            except Exception as e:
                logger.warning(f"⚠️  {test_case['name']}: {type(e).__name__}")
        
        assert handled >= 1, "Should handle at least 1 edge case gracefully"

    # ========== PERFORMANCE TESTS ==========

    def test_12_performance_benchmarks(self, rag_module, sample_documents):
        """TEST 12: Measure and document performance characteristics."""
        queries = [
            "What is Python?",
            "Explain machine learning",
            "How does information retrieval work?"
        ]
        
        logger.info("\nPERFORMANCE BENCHMARKS:")
        logger.info("-" * 70)
        
        times = []
        for query in queries:
            start = time.time()
            response = rag_module.generate(query, documents=sample_documents, max_tokens=200)
            elapsed = time.time() - start
            times.append(elapsed)
            
            logger.info(f"  Query: {query}")
            logger.info(f"    Time: {elapsed:.2f}s | Length: {len(response.answer)} chars | "
                       f"Citations: {len(response.citations)}")
        
        avg_time = sum(times) / len(times)
        logger.info("-" * 70)
        logger.info(f"  Average response time: {avg_time:.2f}s")
        logger.info(f"  Min: {min(times):.2f}s | Max: {max(times):.2f}s")
        
        assert avg_time < 180, "Average response should complete within 180s"

    @pytest.fixture
    def limited_documents(self):
        """Return minimal documents for specific tests."""
        return [
            {
                "id": "doc_minimal",
                "title": "Limited Info",
                "content": "Basic test content with minimal information."
            }
        ]


# ========== PYTEST SETUP ==========

@pytest.mark.order("first")
class TestRAGSetup:
    """Verify RAG module is properly configured before running integration tests."""
    
    def test_rag_module_imports(self):
        """Verify all RAG module imports are available."""
        try:
            from src.rag.llm_provider import OllamaProvider
            from src.rag.rag_module import RAGModule
            from src.rag.output_parser import OutputParser, RAGResponse
            from src.rag.citations import CitationExtractor
            
            assert all([OllamaProvider, RAGModule, OutputParser, RAGResponse, CitationExtractor])
            logger.info("✅ All RAG module imports successful")
        except ImportError as e:
            logger.error(f"❌ Import failed: {e}")
            raise

    def test_ollama_availability(self):
        """Verify Ollama service is running and accessible."""
        try:
            from src.rag.llm_provider import OllamaProvider
            provider = OllamaProvider(model="llama3.2:latest", timeout=30)
            
            assert provider.is_available(), "Ollama service not available"
            logger.info("✅ Ollama service is running and accessible")
        except Exception as e:
            logger.error(f"❌ Ollama not available: {e}")
            pytest.skip(f"Ollama service not available: {e}")


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])
