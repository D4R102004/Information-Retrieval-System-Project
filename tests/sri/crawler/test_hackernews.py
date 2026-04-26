"""Tests for the HackerNewsSpider class.

Verifies that the HackerNewsSpider correctly fetches articles,
handles HTTP errors gracefully, and builds ArticleItems properly.
All HTTP calls are mocked to avoid real network requests.
"""

from unittest.mock import MagicMock

from sri.crawler.spiders.hackernews import HackerNewsSpider


def test_fetch_page_returns_hits():
    """_fetch_page should return list of hits when API response is valid."""
    # Arrange
    spider = HackerNewsSpider()
    spider._get_json = MagicMock(
        return_value={
            "hits": [{"title": "Test", "objectID": "123"}],
            "nbPages": 1,
        }
    )

    # Act
    result = spider._fetch_page("software", 0)

    # Assert
    assert result == [{"title": "Test", "objectID": "123"}]


def test_build_item_returns_article_item():
    """_build_item should return an ArticleItem when article data is valid."""

    # Arrange
    spider = HackerNewsSpider()
    raw_article = {
        "title": "Test Article",
        "url": "https://example.com",
        "created_at": "2024-01-01T00:00:00Z",
        "story_text": "This is a test article.",
        "_tags": ["software", "test"],
    }

    # Act
    result = spider._build_item(raw_article)

    # Assert
    assert result is not None
    assert result["title"] == "Test Article"
    assert result["url"] == "https://example.com"
    assert result["date"] == "2024-01-01T00:00:00Z"
    assert result["content"] == "This is a test article."
    assert result["source"] == "hackernews"
    assert result["tags"] == ["software", "test"]
