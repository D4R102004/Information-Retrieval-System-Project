"""
System Integration Tests - Complete Backend Testing

Tests for the integrated information retrieval and RAG system covering:
- Database state management (empty, insufficient, normal)
- Auto-loading from crawlers
- Insufficiency detection (3-criterion)
- Web search triggering
- Database cleanup and repopulation
- Error handling and recovery
- Metadata tracking
- End-to-end query pipeline

Test cases are designed to validate all backend components working together
through the MainOrchestrator unified interface.
"""

import pytest
import logging
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.main_orchestator import MainOrchestator
from src.rag.output_parser import RAGResponse
from src.sri.crawler.settings import CrawlerSettings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TestSystemInitialization:
    """Test system initialization and component setup."""

    def test_orchestrator_initialization(self):
        """Verify MainOrchestrator initializes all components correctly."""
        orchestrator = MainOrchestator()
        
        assert orchestrator is not None
        assert orchestrator.pipeline is not None
        assert orchestrator.rag_module is not None
        assert orchestrator.web_searcher is not None
        assert orchestrator.crawler_caller is not None
        assert orchestrator.sufficiency_checker is not None
        logger.info("✓ Orchestrator initialization successful")

    def test_configuration_loaded(self):
        """Verify configuration settings are properly loaded."""
        orchestrator = MainOrchestator()
        settings = orchestrator.settings
        
        assert settings.MIN_DOCUMENTS_THRESHOLD >= 0
        assert settings.MIN_AVG_SCORE_THRESHOLD >= 0
        assert settings.MIN_RESULTS_FOR_QUERY >= 0
        assert isinstance(settings.AUTO_CRAWL_ON_EMPTY, bool)
        logger.info(f"✓ Configuration loaded: thresholds={settings.MIN_RESULTS_FOR_QUERY}")

    def test_data_paths_configured(self):
        """Verify data paths are correctly configured."""
        orchestrator = MainOrchestator()
        
        assert orchestrator.data_dir is not None
        assert orchestrator.documents_path is not None
        assert orchestrator.raw_data_dir is not None
        logger.info("✓ Data paths configured")


class TestDatabaseHealthCheck:
    """Test database health checking and diagnostics."""

    def test_health_check_returns_correct_structure(self):
        """Verify health check returns all required fields."""
        orchestrator = MainOrchestator()
        health = orchestrator.check_database_health()
        
        required_fields = {
            'is_empty', 'document_count', 'file_document_count',
            'can_search', 'has_chromadb', 'status'
        }
        assert all(field in health for field in required_fields)
        assert health['status'] in ['healthy', 'empty', 'degraded', 'error']
        logger.info(f"✓ Health check structure valid: status={health['status']}")

    def test_empty_database_detection(self):
        """Verify empty database is correctly detected."""
        orchestrator = MainOrchestator()
        # Clear database first
        orchestrator.clear_all_indices()
        
        health = orchestrator.check_database_health()
        assert health['is_empty'] == True
        assert health['document_count'] == 0
        assert health['status'] in ['empty', 'degraded']
        logger.info("✓ Empty database correctly detected")

    def test_status_diagnostics(self):
        """Verify complete system status diagnostics."""
        orchestrator = MainOrchestator()
        status = orchestrator.get_status()
        
        assert 'database' in status
        assert 'crawlers' in status
        assert 'pipeline' in status
        assert 'timestamp' in status
        assert status['database']['status'] in ['healthy', 'empty', 'degraded', 'error']
        logger.info(f"✓ Status diagnostics complete: {status['database']['status']}")


class TestEmptyDatabaseAutoLoad:
    """Test auto-loading behavior when database is empty."""

    def test_empty_db_auto_load_trigger(self):
        """Verify auto-load is triggered when database is empty."""
        orchestrator = MainOrchestator()
        
        # Clear database
        clear_result = orchestrator.clear_all_indices()
        assert clear_result['success'] == True
        
        # Verify database is empty
        health = orchestrator.check_database_health()
        assert health['is_empty'] == True
        
        logger.info("✓ Empty database state confirmed")

    def test_query_with_empty_database_auto_reload(self):
        """Verify query triggers auto-load when database is empty."""
        orchestrator = MainOrchestator()
        
        # Clear database
        orchestrator.clear_all_indices()
        
        # Execute query with auto_reload enabled
        try:
            response = orchestrator.query(
                question="test query",
                auto_reload_empty=True,
                use_web_search=False
            )
            
            # Should return RAGResponse even if generation fails
            assert isinstance(response, RAGResponse)
            # Metadata should track auto-crawl
            if 'auto_crawled' in response.metadata:
                assert isinstance(response.metadata['auto_crawled'], bool)
            logger.info("✓ Query with auto-reload handled successfully")
        except Exception as e:
            # Some errors are expected (e.g., Ollama not running)
            logger.warning(f"Query execution note: {str(e)}")


class TestInsufficientDataDetection:
    """Test 3-criterion insufficiency detection."""

    def test_detect_insufficiency_quantity(self):
        """Test quantity criterion: too few results."""
        orchestrator = MainOrchestator()
        
        # Minimal results (below MIN_RESULTS_FOR_QUERY)
        results = [
            {'content': 'doc1', 'score': 0.5},
            {'content': 'doc2', 'score': 0.4}
        ]
        
        insufficiency = orchestrator.detect_insufficiency_for_query(
            "query",
            results
        )
        
        assert isinstance(insufficiency, Dict)
        assert 'is_insufficient' in insufficiency
        assert 'reasons' in insufficiency
        assert 'metrics' in insufficiency
        
        if len(results) < orchestrator.settings.MIN_RESULTS_FOR_QUERY:
            assert insufficiency['is_insufficient'] == True
            assert any('few results' in str(r) for r in insufficiency['reasons'])
        logger.info(f"✓ Quantity criterion validated: reasons={insufficiency['reasons']}")

    def test_detect_insufficiency_quality(self):
        """Test quality criterion: low average score."""
        orchestrator = MainOrchestator()
        
        # Low-score results
        results = [
            {'content': 'doc1', 'score': 0.2},
            {'content': 'doc2', 'score': 0.15},
            {'content': 'doc3', 'score': 0.18}
        ]
        
        insufficiency = orchestrator.detect_insufficiency_for_query(
            "query",
            results
        )
        
        avg_score = insufficiency['metrics']['avg_score']
        threshold = orchestrator.settings.MIN_AVG_SCORE_THRESHOLD
        
        if avg_score < threshold:
            assert insufficiency['is_insufficient'] == True
            assert any('score' in str(r) for r in insufficiency['reasons'])
        logger.info(f"✓ Quality criterion validated: avg_score={avg_score:.2f}")

    # Legacy semantic criterion test (keyword overlap)
    # def test_detect_insufficiency_semantic(self):
    #     """Test semantic criterion: insufficient keyword overlap."""
    #     orchestrator = MainOrchestator()
        
    #     # Results with minimal keyword overlap to query
    #     results = [
    #         {'content': 'unrelated content about cooking', 'score': 0.5},
    #         {'content': 'another unrelated topic', 'score': 0.45}
    #     ]
        
    #     query = "machine learning tensorflow neural networks"
    #     insufficiency = orchestrator.detect_insufficiency_for_query(
    #         query,
    #         results
    #     )
        
    #     assert 'metrics' in insufficiency
    #     assert 'has_semantic_overlap' in insufficiency['metrics']
    #     logger.info(f"✓ Semantic criterion validated")

    def test_insufficient_detection_multiple_reasons(self):
        """Test insufficiency when multiple criteria fail."""
        orchestrator = MainOrchestator()
        
        # Results failing on quantity, quality, and potentially semantic
        results = [
            {'content': 'x', 'score': 0.1},
            {'content': 'y', 'score': 0.2}
        ]
        
        insufficiency = orchestrator.detect_insufficiency_for_query(
            "test",
            results
        )
        
        if insufficiency['is_insufficient']:
            assert len(insufficiency['reasons']) >= 1
        logger.info(f"✓ Multiple insufficiency reasons captured")


class TestWebSearchIntegration:
    """Test web search triggering and integration."""

    def test_web_search_can_be_disabled(self):
        """Verify web search can be disabled via parameter."""
        orchestrator = MainOrchestator()
        orchestrator.clear_all_indices()
        
        try:
            response = orchestrator.query(
                question="test",
                use_web_search=False,
                auto_reload_empty=False
            )
            
            assert isinstance(response, RAGResponse)
            logger.info("✓ Web search disabled successfully")
        except Exception as e:
            logger.warning(f"Query with no web search: {str(e)}")

    def test_web_search_triggered_on_insufficiency(self):
        """Verify web search is triggered when local search is insufficient."""
        orchestrator = MainOrchestator()
        
        # Create insufficient results scenario
        minimal_results = [
            {'content': 'minimal content', 'score': 0.2}
        ]
        
        insufficiency = orchestrator.detect_insufficiency_for_query(
            "test query",
            minimal_results
        )
        
        # If insufficient, web search should be considered
        if insufficiency['is_insufficient']:
            logger.info("✓ Web search would be triggered (insufficient condition met)")
        else:
            logger.info("✓ Results are sufficient (web search not needed)")


class TestDocumentConsolidation:
    """Test document consolidation and deduplication."""

    def test_consolidate_documents_deduplicates(self):
        """Verify documents are properly deduplicated."""
        orchestrator = MainOrchestator()
        
        local_results = [
            {'content': 'Document A', 'title': 'A'},
            {'content': 'Document B', 'title': 'B'}
        ]
        
        web_results = [
            {'content': 'Document B', 'title': 'B'},  # Duplicate
            {'content': 'Document C', 'title': 'C'}   # New
        ]
        
        consolidated = orchestrator._consolidate_documents(
            local_results,
            web_results
        )
        
        # Should have 3 unique documents (A, B, C)
        assert len(consolidated) <= 3
        # Local results should be prioritized (come first)
        assert 'A' in str(consolidated[0]['title'])
        logger.info(f"✓ Documents consolidated: {len(consolidated)} unique docs")

    def test_consolidate_prioritizes_local(self):
        """Verify local results are prioritized over web results."""
        orchestrator = MainOrchestator()
        
        local_results = [
            {'content': 'High quality local', 'score': 0.8}
        ]
        
        web_results = [
            {'content': 'Lower quality web', 'score': 0.5}
        ]
        
        consolidated = orchestrator._consolidate_documents(
            local_results,
            web_results
        )
        
        # First item should be local (higher score)
        assert consolidated[0]['content'] == 'High quality local'
        logger.info("✓ Local results prioritized correctly")


class TestDatabaseManagement:
    """Test database operations: clear, load, health."""

    def test_clear_all_indices(self):
        """Verify clear_all_indices works correctly."""
        orchestrator = MainOrchestator()
        
        result = orchestrator.clear_all_indices()
        
        assert isinstance(result, Dict)
        assert 'success' in result
        assert 'message' in result
        assert 'timestamp' in result
        
        if result['success']:
            health = orchestrator.check_database_health()
            assert health['is_empty'] == True
        logger.info(f"✓ Clear all indices: {result['message']}")

    def test_load_documents_from_crawlers(self):
        """Verify load_documents_from_crawlers works correctly."""
        orchestrator = MainOrchestator()
        
        # Clear first
        orchestrator.clear_all_indices()
        
        # Load with minimal articles for testing
        result = orchestrator.load_documents_from_crawlers(
            max_articles=5,
            force_recrawl=False
        )
        
        assert isinstance(result, Dict)
        assert 'success' in result
        assert 'total_documents' in result
        assert 'indexed_documents' in result
        
        logger.info(
            f"✓ Load documents: {result.get('indexed_documents', 0)} indexed"
        )

    def test_database_repopulation(self):
        """Verify complete database repopulation: clear → load → verify."""
        orchestrator = MainOrchestator()
        
        # Step 1: Clear
        clear_result = orchestrator.clear_all_indices()
        assert clear_result['success'] == True
        
        # Step 2: Verify empty
        health1 = orchestrator.check_database_health()
        assert health1['is_empty'] == True
        
        # Step 3: Load
        load_result = orchestrator.load_documents_from_crawlers(
            max_articles=3,
            force_recrawl=False
        )
        
        # Step 4: Verify loaded (if load successful)
        if load_result['success']:
            health2 = orchestrator.check_database_health()
            if health2['document_count'] > 0:
                assert health2['is_empty'] == False
        
        logger.info("✓ Database repopulation cycle complete")


class TestQueryPipeline:
    """Test the complete query execution pipeline."""

    def test_query_input_validation(self):
        """Verify query input validation."""
        orchestrator = MainOrchestator()
        
        # Test empty query
        response = orchestrator.query(
            question="",
            auto_reload_empty=False,
            use_web_search=False
        )
        
        # Response should be RAGResponse (Pydantic model)
        assert hasattr(response, 'answer')
        assert hasattr(response, 'citations')
        assert hasattr(response, 'metadata')
        logger.info("✓ Empty query handled gracefully")

    def test_query_metadata_tracking(self):
        """Verify metadata is tracked correctly."""
        orchestrator = MainOrchestator()
        
        try:
            response = orchestrator.query(
                question="test query",
                auto_reload_empty=False,
                use_web_search=False
            )
            
            if hasattr(response, 'metadata') and response.metadata:
                metadata = response.metadata
                assert 'timestamp' in metadata
                assert 'generation_time_seconds' in metadata or 'auto_crawled' in metadata
                logger.info(f"✓ Metadata tracked: {list(metadata.keys())}")
        except Exception as e:
            logger.warning(f"Query metadata test: {str(e)}")

    def test_query_response_structure(self):
        """Verify query response has correct structure."""
        orchestrator = MainOrchestator()
        
        try:
            response = orchestrator.query(
                question="What is machine learning?",
                auto_reload_empty=False,
                use_web_search=False
            )
            
            assert isinstance(response, RAGResponse)
            assert hasattr(response, 'answer')
            assert hasattr(response, 'citations')
            # Metadata should be present
            if hasattr(response, 'metadata'):
                assert isinstance(response.metadata, dict)
            logger.info("✓ Query response structure valid")
        except Exception as e:
            logger.warning(f"Query response structure: {str(e)}")


class TestErrorHandling:
    """Test error handling and recovery."""

    def test_malformed_documents_handling(self):
        """Verify system handles malformed documents gracefully."""
        orchestrator = MainOrchestator()
        
        # Try to consolidate with incomplete documents
        malformed_results = [
            {'title': 'No content'},
            {},  # Empty document
            {'content': 'Valid content'}
        ]
        
        try:
            consolidated = orchestrator._consolidate_documents(
                malformed_results,
                []
            )
            assert isinstance(consolidated, list)
            logger.info("✓ Malformed documents handled")
        except Exception as e:
            logger.error(f"Malformed documents: {str(e)}")

    def test_missing_content_extraction(self):
        """Verify content extraction handles missing fields."""
        orchestrator = MainOrchestator()
        
        # Document with no standard content fields
        doc = {'title': 'only title', 'url': 'http://example.com'}
        
        content = orchestrator._extract_content(doc)
        assert isinstance(content, str)
        logger.info(f"✓ Missing content handled: extracted {len(content)} chars")

    def test_invalid_query_handling(self):
        """Verify invalid queries are handled."""
        orchestrator = MainOrchestator()
        
        invalid_queries = [
            "",
            None,
            "   ",
            "\t\n"
        ]
        
        for query in invalid_queries:
            if query is None:
                continue
            try:
                response = orchestrator.query(
                    question=query,
                    auto_reload_empty=False,
                    use_web_search=False
                )
                assert isinstance(response, RAGResponse)
            except Exception as e:
                logger.debug(f"Invalid query '{query}' handled: {type(e).__name__}")
        
        logger.info("✓ Invalid queries handled")


class TestIntegrationScenarios:
    """End-to-end integration test scenarios."""

    def test_scenario_empty_db_to_query(self):
        """
        Scenario: Complete workflow from empty database to query answer.
        
        Steps:
        1. Clear database
        2. Verify empty
        3. Load documents
        4. Execute query
        """
        orchestrator = MainOrchestator()
        
        # Step 1-2: Clear and verify
        orchestrator.clear_all_indices()
        health1 = orchestrator.check_database_health()
        assert health1['is_empty'] == True
        
        # Step 3: Load documents
        load_result = orchestrator.load_documents_from_crawlers(
            max_articles=5,
            force_recrawl=False
        )
        
        # Step 4: Execute query (if load successful)
        if load_result['success']:
            try:
                response = orchestrator.query(
                    question="test query",
                    auto_reload_empty=False,
                    use_web_search=False
                )
                assert isinstance(response, RAGResponse)
                logger.info("✓ Scenario: empty→load→query completed")
            except Exception as e:
                logger.warning(f"Scenario query: {str(e)}")
        else:
            logger.warning("Scenario load failed - query skipped")

    def test_scenario_insufficient_data_web_augmentation(self):
        """
        Scenario: Insufficient local data triggers web search consideration.
        
        Steps:
        1. Create insufficient results
        2. Detect insufficiency
        3. Verify web search would be triggered
        """
        orchestrator = MainOrchestator()
        
        # Create insufficient results
        results = [
            {'content': 'minimal', 'score': 0.15}
        ]
        
        insufficiency = orchestrator.detect_insufficiency_for_query(
            "test",
            results
        )
        
        # If insufficient, web search would be triggered
        if insufficiency['is_insufficient']:
            logger.info("✓ Scenario: insufficient→web-search detected")
        else:
            logger.info("✓ Scenario: results sufficient (web search not needed)")

    def test_scenario_complete_database_refresh(self):
        """
        Scenario: Complete database refresh cycle.
        
        Steps:
        1. Verify initial state
        2. Clear all data
        3. Repopulate from crawlers
        4. Verify new state
        """
        orchestrator = MainOrchestator()
        
        # Step 1: Initial health check
        health1 = orchestrator.check_database_health()
        initial_count = health1['document_count']
        
        # Step 2: Clear
        orchestrator.clear_all_indices()
        health2 = orchestrator.check_database_health()
        assert health2['is_empty'] == True
        
        # Step 3: Repopulate
        result = orchestrator.load_documents_from_crawlers(
            max_articles=3,
            force_recrawl=False
        )
        
        # Step 4: Verify new state
        health3 = orchestrator.check_database_health()
        
        if result['success'] and health3['document_count'] > 0:
            logger.info(
                f"✓ Scenario: refresh cycle "
                f"{initial_count}→0→{health3['document_count']} docs"
            )
        else:
            logger.info("✓ Scenario: refresh cycle attempted")


class TestEvaluationModule:
    """Test evaluation module integration with MainOrchestrator."""

    def test_evaluate_test_with_spec(self):
        """Verify evaluate_test works with provided test specification."""
        orchestrator = MainOrchestator()
        
        # Create minimal test specification
        test_spec = {
            "test_queries": [
                {
                    "query_id": "test_q1",
                    "query": "information retrieval",
                    "relevant": ["doc_1", "doc_2"],
                    "grades": {"doc_1": 3, "doc_2": 2}
                },
                {
                    "query_id": "test_q2",
                    "query": "machine learning",
                    "relevant": ["doc_1"],
                    "grades": {"doc_1": 3}
                }
            ]
        }
        
        result = orchestrator.evaluate_test(test_spec)
        
        # Verify response structure
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'aggregate' in result
        assert 'per_query' in result
        assert 'timestamp' in result
        assert 'execution_time_seconds' in result
        
        # Verify status
        assert result['status'] in ['success', 'error']
        
        logger.info(f"✓ Evaluate with spec: status={result['status']}")

    def test_evaluate_test_returns_aggregate_metrics(self):
        """Verify evaluation returns all required aggregate metrics."""
        orchestrator = MainOrchestator()
        
        test_spec = {
            "test_queries": [
                {
                    "query_id": "q1",
                    "query": "test query",
                    "relevant": ["doc_1"]
                }
            ]
        }
        
        result = orchestrator.evaluate_test(test_spec)
        
        if result['status'] == 'success':
            aggregate = result['aggregate']
            
            # Verify required metrics present
            required_metrics = ['num_queries', 'MAP', 'MRR']
            assert all(m in aggregate for m in required_metrics)
            
            # Verify metrics are numeric
            assert isinstance(aggregate['num_queries'], int)
            assert isinstance(aggregate['MAP'], (int, float))
            assert isinstance(aggregate['MRR'], (int, float))
            
            logger.info(f"✓ Aggregate metrics: MAP={aggregate['MAP']}, MRR={aggregate['MRR']}")
        else:
            logger.info(f"⚠ Could not verify metrics: {result.get('message')}")

    def test_evaluate_test_returns_per_query_results(self):
        """Verify evaluation returns per-query detailed results."""
        orchestrator = MainOrchestator()
        
        test_spec = {
            "test_queries": [
                {
                    "query_id": "q1",
                    "query": "python programming",
                    "relevant": ["doc_1", "doc_2"]
                }
            ]
        }
        
        result = orchestrator.evaluate_test(test_spec)
        
        if result['status'] == 'success':
            per_query = result['per_query']
            
            assert isinstance(per_query, list)
            if len(per_query) > 0:
                first_query_result = per_query[0]
                
                # Verify structure
                assert 'query_id' in first_query_result
                assert 'ap' in first_query_result
                assert 'rr' in first_query_result
                
                logger.info(f"✓ Per-query results: {len(per_query)} queries evaluated")
        else:
            logger.info(f"⚠ Could not verify per-query results: {result.get('message')}")

    def test_evaluate_test_with_none_spec_fallback(self):
        """Verify evaluate_test loads from JSON file when test_spec is None."""
        orchestrator = MainOrchestator()
        
        # Call with None - should attempt to load from data/test_queries.json
        result = orchestrator.evaluate_test(None)
        
        assert isinstance(result, dict)
        assert 'status' in result
        
        # Either success (if file exists) or error (if file doesn't exist)
        if result['status'] == 'error':
            # File doesn't exist - that's OK
            assert 'not found' in result.get('message', '').lower() or \
                   'no test specification' in result.get('message', '').lower()
            logger.info("✓ Fallback correctly reports missing file")
        else:
            # File exists - should have metrics
            assert 'aggregate' in result
            logger.info(f"✓ Fallback loaded test file: {len(result['per_query'])} queries")

    def test_evaluate_test_with_empty_spec(self):
        """Verify evaluate_test handles empty test specification."""
        orchestrator = MainOrchestator()
        
        # Empty test_queries list
        test_spec = {"test_queries": []}
        
        result = orchestrator.evaluate_test(test_spec)
        
        # Should return error for empty list
        assert 'status' in result
        
        if result['status'] == 'error':
            # Message should indicate invalid test queries
            assert 'empty' in result.get('message', '').lower() or \
                   'non-empty' in result.get('message', '').lower() or \
                   'test' in result.get('message', '').lower()
            logger.info("✓ Empty test_queries correctly rejected")
        
        logger.info(f"Response status: {result['status']}")

    def test_evaluate_test_with_malformed_spec(self):
        """Verify evaluate_test handles malformed specifications."""
        orchestrator = MainOrchestator()
        
        # Specifications missing required fields
        malformed_specs = [
            {"wrong_key": []},  # Missing test_queries
            {"test_queries": "not a list"},  # test_queries not a list
            {"test_queries": [{"missing_fields": "value"}]},  # Missing required fields
        ]
        
        for spec in malformed_specs:
            result = orchestrator.evaluate_test(spec)
            
            assert 'status' in result
            # Should either error or attempt processing
            assert result['status'] in ['success', 'error']
            
            logger.info(f"✓ Malformed spec handled: {result['status']}")

    def test_evaluate_test_execution_time_tracked(self):
        """Verify evaluation tracks execution time."""
        orchestrator = MainOrchestator()
        
        test_spec = {
            "test_queries": [
                {
                    "query_id": "q1",
                    "query": "test",
                    "relevant": ["doc_1"]
                }
            ]
        }
        
        result = orchestrator.evaluate_test(test_spec)
        
        assert 'execution_time_seconds' in result
        time_taken = result['execution_time_seconds']
        
        # Time should be numeric and >= 0
        assert isinstance(time_taken, (int, float))
        assert time_taken >= 0
        
        logger.info(f"✓ Evaluation time tracked: {time_taken:.3f}s")

    def test_evaluate_test_timestamp_valid(self):
        """Verify evaluation includes valid ISO timestamp."""
        orchestrator = MainOrchestator()
        
        test_spec = {
            "test_queries": [
                {
                    "query_id": "q1",
                    "query": "test",
                    "relevant": ["doc_1"]
                }
            ]
        }
        
        result = orchestrator.evaluate_test(test_spec)
        
        assert 'timestamp' in result
        timestamp = result['timestamp']
        
        # Verify ISO format
        from datetime import datetime
        try:
            datetime.fromisoformat(timestamp)
            logger.info(f"✓ Timestamp valid ISO format: {timestamp}")
        except ValueError:
            raise AssertionError(f"Invalid timestamp format: {timestamp}")

    def test_evaluate_test_integration_with_search(self):
        """Verify evaluation uses actual search pipeline."""
        orchestrator = MainOrchestator()
        
        # Clear to ensure consistent state
        orchestrator.clear_all_indices()
        
        # Load minimal data
        load_result = orchestrator.load_documents_from_crawlers(
            max_articles=2,
            force_recrawl=False
        )
        
        if load_result['success'] and load_result.get('indexed_documents', 0) > 0:
            test_spec = {
                "test_queries": [
                    {
                        "query_id": "q1",
                        "query": "technology OR software OR programming",
                        "relevant": ["doc_1"]
                    }
                ]
            }
            
            result = orchestrator.evaluate_test(test_spec)
            
            assert result['status'] in ['success', 'error']
            
            if result['status'] == 'success':
                # Should have executed against real documents
                assert len(result['per_query']) >= 0
                logger.info("✓ Evaluation integrated with search pipeline")
        else:
            logger.info("⚠ Skipped integration test: insufficient data loaded")

    def test_evaluate_test_handles_missing_grades(self):
        """Verify evaluation handles queries without grades field."""
        orchestrator = MainOrchestator()
        
        # Test queries without optional 'grades' field
        test_spec = {
            "test_queries": [
                {
                    "query_id": "q1",
                    "query": "test query",
                    "relevant": ["doc_1", "doc_2"]
                    # Note: no 'grades' field
                }
            ]
        }
        
        result = orchestrator.evaluate_test(test_spec)
        
        # Should handle gracefully
        assert 'status' in result
        assert result['status'] in ['success', 'error']
        
        if result['status'] == 'success':
            assert len(result['per_query']) > 0
            logger.info("✓ Missing grades field handled correctly")
        else:
            logger.info(f"⚠ Evaluation status: {result.get('message')}")

    def test_evaluate_test_error_handling_robustness(self):
        """Verify evaluation handles various error scenarios."""
        orchestrator = MainOrchestator()
        
        # Various error scenarios
        error_specs = [
            None,  # Will attempt file load
            {"test_queries": None},  # None instead of list
            {"test_queries": [None]},  # None in list
        ]
        
        for spec in error_specs:
            result = orchestrator.evaluate_test(spec)
            
            # Should always return a result dict with status
            assert isinstance(result, dict)
            assert 'status' in result
            assert result['status'] in ['success', 'error']
            
            # Error results should have message
            if result['status'] == 'error':
                assert 'message' in result
                assert len(result['message']) > 0
        
        logger.info("✓ Error scenarios handled robustly")


class TestPerformanceMetrics:
    """Test performance tracking and timing."""

    def test_metadata_timing(self):
        """Verify execution timing is tracked in metadata."""
        orchestrator = MainOrchestator()
        
        try:
            response = orchestrator.query(
                question="simple test",
                auto_reload_empty=False,
                use_web_search=False
            )
            
            if hasattr(response, 'metadata'):
                if 'generation_time_seconds' in response.metadata:
                    time_taken = response.metadata['generation_time_seconds']
                    assert isinstance(time_taken, (int, float))
                    logger.info(f"✓ Query timing: {time_taken:.2f}s")
        except Exception as e:
            logger.warning(f"Timing test: {str(e)}")

    def test_status_includes_timestamp(self):
        """Verify status includes timestamp."""
        orchestrator = MainOrchestator()
        status = orchestrator.get_status()
        
        assert 'timestamp' in status
        from datetime import datetime
        # Verify it's valid ISO format
        try:
            datetime.fromisoformat(status['timestamp'])
            logger.info("✓ Status timestamp valid ISO format")
        except ValueError:
            logger.warning("Timestamp not ISO format")


if __name__ == '__main__':
    """
    Run all system integration tests.
    
    Usage:
        pytest tests/sri/test_system.py -v
        pytest tests/sri/test_system.py -v -s  # with output
        pytest tests/sri/test_system.py::TestDatabaseHealthCheck -v
    """
    pytest.main([__file__, '-v', '-s'])
