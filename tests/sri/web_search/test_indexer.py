"""Tests for WebResultIndexer.

This module verifies that articles are correctly persisted to disk
as JSON files and that the saved content matches the expected format.
"""

# Standard library
import json

# Local
from sri.web_search.indexer import WebResultIndexer


def test_save_article_creates_file(tmp_path):
    """save_article should create a JSON file on disk."""
    # Arrange
    indexer = WebResultIndexer(output_directory=tmp_path)

    article = {
        "id": "123",
        "title": "Docker Guide",
        "content": "Learn Docker...",
        "url": "https://example.com/docker",
        "source": "realpython",
        "date": "2026-01-01",
        "tags": ["docker"],
    }

    # Act
    indexer.save_article(article)

    # Assert
    expected_file = tmp_path / "realpython" / "123.json"
    assert expected_file.exists()


def test_save_article_writes_correct_json(tmp_path):
    """Saved file should contain the correct JSON structure."""
    # Arrange
    indexer = WebResultIndexer(output_directory=tmp_path)

    article = {
        "id": "456",
        "title": "Python Async",
        "content": "Async explained...",
        "url": "https://example.com/async",
        "source": "realpython",
        "date": "2026-01-01",
        "tags": ["python", "async"],
    }

    # Act
    indexer.save_article(article)

    # Assert
    file_path = tmp_path / "realpython" / "456.json"

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data == article
