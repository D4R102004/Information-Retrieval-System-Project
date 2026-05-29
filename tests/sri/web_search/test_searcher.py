"""Tests for WebSearcher module.

This module validates that the WebSearcher correctly interacts with the
DuckDuckGo API (mocked via unittest.mock), and that it properly transforms
and filters raw search results into the standardized article format.
"""

# Standard library
from unittest.mock import patch

# Local
from sri.web_search.searcher import WebSearcher


def test_search_returns_articles():
    """Search should return normalized articles when valid results are returned."""
    # Arrange
    fake_results = [
        {
            "title": "Docker tutorial",
            "href": "https://example.com/docker",
            "body": "Docker is a container tool...",
        }
    ]
    with (
        patch("sri.web_search.searcher.DDGS") as mock_ddgs,
        patch(
            "sri.web_search.searcher.WebSearcher._fetch_full",
            return_value="Full Docker article content.",
        ),
    ):
        mock_ddgs.return_value.__enter__.return_value.text.return_value = fake_results
        searcher = WebSearcher()
        # Act
        results = searcher.search("docker")
    # Assert
    assert len(results) == 1
    assert results[0]["title"] == "Docker tutorial"
    assert results[0]["url"] == "https://example.com/docker"
    assert results[0]["content"] == "Full Docker article content."


def test_search_filters_results_without_title():
    """Search should filter out results missing a title."""
    fake_results = [
        {
            "title": None,
            "href": "https://example.com/docker",
            "body": "Docker is a container tool...",
        }
    ]

    with patch("sri.web_search.searcher.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = fake_results

        searcher = WebSearcher()
        results = searcher.search("docker")

    assert len(results) == 0


def test_search_filters_results_without_url():
    """Search should filter out results missing a URL."""
    fake_results = [
        {
            "title": "Docker tutorial",
            "href": None,
            "body": "Docker is a container tool...",
        }
    ]

    with patch("sri.web_search.searcher.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = fake_results

        searcher = WebSearcher()
        results = searcher.search("docker")

    assert len(results) == 0


def test_search_filters_results_without_content():
    """Search should filter out results when full content fetching fails."""
    # Arrange
    fake_results = [
        {
            "title": "Docker tutorial",
            "href": "https://example.com/docker",
            "body": "Docker is a container tool...",
        }
    ]

    with (
        patch("sri.web_search.searcher.DDGS") as mock_ddgs,
        patch(
            "sri.web_search.searcher.WebSearcher._fetch_full",
            return_value="",
        ),
    ):
        mock_ddgs.return_value.__enter__.return_value.text.return_value = fake_results

        searcher = WebSearcher()

        # Act
        results = searcher.search("docker")

    # Assert
    assert len(results) == 0
