"""Tests for the DevToSpider class.

Verifies that the Dev.to spider correctly fetches articles,
handles HTTP errors gracefully, and builds ArticleItems properly.
All HTTP calls are mocked to avoid real network requests.
"""

from unittest.mock import MagicMock

import httpx

from sri.crawler.spiders.devto import DevToSpider


def test_get_json_returns_parsed_json_on_success():
    """_get_json should return parsed JSON when HTTP request succeeds."""
    # Arrange
    spider = DevToSpider()
    mock_response = MagicMock()
    mock_response.json.return_value = {"title": "Test Article"}
    spider._client.get = MagicMock(return_value=mock_response)

    # Act
    result = spider._get_json("https://dev.to/api/articles/1")

    # Assert
    assert result == {"title": "Test Article"}


def test_get_json_returns_none():
    """_get_json should return None when HTTP request fails."""
    # Arrange
    spider = DevToSpider()
    spider._client.get = MagicMock(side_effect=httpx.HTTPError("HTTP Error"))

    # Act
    result = spider._get_json("https://dev.to/api/articles/1")

    # Assert
    assert result is None


def test_build_item_returns_none():
    """_build_item should return None when article data is invalid."""
    # Arrange
    spider = DevToSpider()
    raw_article = {
        "invalid": "data",
        "id": 123,
    }
    spider._fetch_full = MagicMock(return_value={})

    # Act
    result = spider._build_item(raw_article)

    # Assert
    assert result is None


def test_build_item_returns_article_item():
    """_build_item should return an ArticleItem when article data is valid."""
    # Arrange
    spider = DevToSpider()
    raw_article = {
        "title": "Test Article",
        "id": 123,
        "published_at": "2024-01-01T00:00:00Z",
        "url": "https://dev.to/test-article",
    }
    full_article = {
        "title": "Test Article",
        "url": "https://dev.to/test-article",
        "published_at": "2024-01-01T00:00:00Z",
        "body_markdown": "This is the body of the test article.",
        "tag_list": ["python", "software"],
    }
    spider._fetch_full = MagicMock(return_value=full_article)

    # Act
    result = spider._build_item(raw_article)

    # Assert
    assert result is not None
    assert result["title"] == "Test Article"
    assert result["url"] == "https://dev.to/test-article"
    assert result["source"] == "devto"
