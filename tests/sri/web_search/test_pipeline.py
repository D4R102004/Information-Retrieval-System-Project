"""Tests for WebSearchPipeline.

This module verifies the orchestration logic of the pipeline,
ensuring correct branching between local results and web fallback,
as well as correct interaction with dependencies (checker, searcher, indexer).
"""

# Local
# Standard library
from unittest.mock import MagicMock

from sri.web_search.pipeline import WebSearchPipeline

# Local


def test_pipeline_returns_local_results_when_sufficient():
    """Pipeline should return local results and not call searcher/indexer."""
    # Arrange
    checker = MagicMock()
    checker.is_sufficient.return_value = True

    searcher = MagicMock()
    indexer = MagicMock()

    pipeline = WebSearchPipeline(checker, searcher, indexer)

    local_results = [{"title": "Local", "url": "x"}]

    # Act
    result = pipeline.search("python", local_results)

    # Assert
    assert result == local_results
    searcher.search.assert_not_called()
    indexer.save_article.assert_not_called()


def test_pipeline_returns_web_results_when_insufficient():
    """Pipeline should return web results when local results are insufficient."""
    # Arrange
    checker = MagicMock()
    checker.is_sufficient.return_value = False

    web_results = [{"title": "Web result", "url": "y"}]

    searcher = MagicMock()
    searcher.search.return_value = web_results

    indexer = MagicMock()

    pipeline = WebSearchPipeline(checker, searcher, indexer)

    local_results = [{"title": "Local", "url": "x"}]

    # Act
    result = pipeline.search("python", local_results)

    # Assert
    assert result == web_results
    searcher.search.assert_called_once_with("python")


def test_pipeline_indexes_results_when_falling_back_to_web():
    """Pipeline should call indexer for each web result."""
    # Arrange
    checker = MagicMock()
    checker.is_sufficient.return_value = False

    web_results = [
        {"id": "1", "title": "A"},
        {"id": "2", "title": "B"},
    ]

    searcher = MagicMock()
    searcher.search.return_value = web_results

    indexer = MagicMock()

    pipeline = WebSearchPipeline(checker, searcher, indexer)

    # Act
    pipeline.search("python", [])

    # Assert
    assert indexer.save_article.call_count == 2
    indexer.save_article.assert_any_call(web_results[0])
    indexer.save_article.assert_any_call(web_results[1])
