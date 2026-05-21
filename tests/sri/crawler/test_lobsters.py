"""Tests for the LobstersSpider class.

Verifies that the Lobste.rs spider correctly fetches articles,
handles HTTP errors gracefully, and builds ArticleItems properly.

All HTTP and external page requests are mocked to avoid real network calls.
"""

from unittest.mock import MagicMock

import httpx

from sri.crawler.spiders.lobsters import LobstersSpider


def test_fetch_page_returns_articles_on_success():
    """_fetch_page should return articles when API request succeeds."""
    # Arrange
    spider = LobstersSpider()
    mock_response = [
        {
            "short_id": "abc123",
            "title": "Test Article",
            "url": "https://example.com/article",
            "created_at": "2026-05-20T15:29:05.000-05:00",
            "tags": ["python", "web"],
        }
    ]
    spider._get_json = MagicMock(return_value=mock_response)
    # Act
    result = spider._fetch_page("", 1)
    # Assert
    assert result == mock_response


def test_fetch_page_returns_empty_list_on_error():
    """_fetch_page should return an empty list when API returns None."""

    # Arrange
    spider = LobstersSpider()

    spider._get_json = MagicMock(return_value=None)

    # Act
    result = spider._fetch_page("", 1)

    # Assert
    assert result == []


def test_build_item_returns_article_item_on_success():
    """_build_item should return an ArticleItem when article data and scraping succeed."""

    # Arrange
    spider = LobstersSpider()

    raw_article = {
        "short_id": "e7lsqn",
        "title": "Test Article",
        "url": "https://example.com/article",
        "created_at": "2026-05-20T15:29:05.000-05:00",
        "tags": ["browsers", "security"],
    }

    html = """
    <html>
        <body>
            <p>First paragraph.</p>
            <p>Second paragraph.</p>
        </body>
    </html>
    """

    spider._client.get = MagicMock(return_value=MagicMock(text=html))

    # Act
    result = spider._build_item(raw_article)

    # Assert
    assert result is not None
    assert result["title"] == "Test Article"
    assert result["url"] == "https://example.com/article"
    assert result["source"] == "lobsters"
    assert "First paragraph." in result["content"]
    assert "Second paragraph." in result["content"]


def test_build_item_returns_none_on_scraping_error():
    """_build_item should return None when external page request fails."""

    # Arrange
    spider = LobstersSpider()

    raw_article = {
        "short_id": "e7lsqn",
        "title": "Test Article",
        "url": "https://example.com/article",
        "created_at": "2026-05-20T15:29:05.000-05:00",
        "tags": ["browsers", "security"],
    }

    spider._client.get = MagicMock(side_effect=httpx.HTTPError("Network error"))

    # Act
    result = spider._build_item(raw_article)

    # Assert
    assert result is None
