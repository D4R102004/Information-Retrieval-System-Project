"""Tests for the JsonPipeline class.

Verifies that the pipeline correctly saves ArticleItems as JSON files,
handles directory creation, and writes the correct file format.
"""

import json

from sri.crawler.items import ArticleItem
from sri.crawler.pipeline import JsonPipeline


def test_save_item_creates_json_file(tmp_path):
    """JsonPipeline should save ArticleItem as a JSON file in the correct location."""
    # Arrange
    pipeline = JsonPipeline(output_directory=tmp_path)
    item = ArticleItem()
    item["id"] = "123"
    item["source"] = "testsource"
    item["title"] = "Test Article"
    item["url"] = "https://example.com/test-article"
    item["date"] = "2024-01-01T00:00:00Z"
    item["content"] = "This is a test article."
    item["tags"] = []

    # Act
    pipeline.save_item(item)

    # Assert
    expected_file = tmp_path / "testsource" / "123.json"
    assert expected_file.exists(), "JSON file was not created at the expected location."

    with expected_file.open(encoding="utf-8") as f:
        data = json.load(f)
        assert data["id"] == "123"
        assert data["source"] == "testsource"
        assert data["title"] == "Test Article"
        assert data["url"] == "https://example.com/test-article"
        assert data["date"] == "2024-01-01T00:00:00Z"
        assert data["content"] == "This is a test article."


def test_save_item_creates_directory_if_not_exists(tmp_path):
    """JsonPipeline should save ArticleItem as a JSON file, while creating the directory if it does not exist."""
    # Arrange
    nested_dir = tmp_path / "new" / "nested" / "dir"
    pipeline = JsonPipeline(output_directory=nested_dir)
    item = ArticleItem()
    item["id"] = "123"
    item["source"] = "testsource"
    item["title"] = "Test Article"
    item["url"] = "https://example.com/test-article"
    item["date"] = "2024-01-01T00:00:00Z"
    item["content"] = "This is a test article."
    item["tags"] = []

    # Act
    pipeline.save_item(item)

    # Assert
    expected_file = nested_dir / "testsource" / "123.json"
    assert expected_file.exists(), "JSON file was not created at the expected location."
