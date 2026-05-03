"""Indexer module for storing raw web search results.

This module provides functionality to persist normalized article data
to disk in JSON format. Each article is stored as an individual file
inside the configured output directory.
"""

# Standard library
import json
from pathlib import Path


class WebResultIndexer:
    """Handles persistence of web search results to the filesystem."""

    def __init__(self, output_directory: Path | str = "data/raw") -> None:
        """Initializes the indexer with an output directory.

        Args:
            output_directory (Path | str, optional): Directory where articles
                will be stored. Defaults to "data/raw".
        """
        self.output_directory = Path(output_directory)

    def save_article(self, article: dict) -> None:
        """Saves a single article to disk as a JSON file.

        The file is named using the article's unique identifier.

        Args:
            article (dict): Article data following the contract format.

        Raises:
            KeyError: If the article does not contain an "id" field.
        """
        article_id = article["id"]
        file_path = self.output_directory / article["source"] / f"{article_id}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(article, f, ensure_ascii=False, indent=2)
