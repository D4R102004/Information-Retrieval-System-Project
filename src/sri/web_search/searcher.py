"""Web search module using DuckDuckGo.

This module provides a simple interface to perform web searches using
DuckDuckGo and transform the results into a standardized article format.
"""

# Standard library
import uuid
from datetime import datetime

# Third-party
from duckduckgo_search import DDGS


class WebSearcher:
    """Searches the web using DuckDuckGo and returns normalized article data."""

    def __init__(self, max_results: int = 10) -> None:
        """Initializes the searcher with a maximum number of results.

        Args:
            max_results (int, optional): Maximum number of search results
                to retrieve. Defaults to 10.
        """
        self.max_results = max_results

    def search(self, query: str) -> list[dict]:
        """Performs a web search and returns normalized article results.

        Args:
            query (str): Search query string.

        Returns:
            list[dict]: A list of articles in the standardized format.
        """
        results: list[dict] = []

        with DDGS() as ddgs:
            raw_results = ddgs.text(query, max_results=self.max_results)

            for raw in raw_results:
                article = self._build_article(raw)
                if article:
                    results.append(article)

        return results

    def _build_article(self, raw: dict) -> dict | None:
        """Converts a raw DuckDuckGo result into the article contract format.

        Args:
            raw (dict): Raw result from DuckDuckGo.

        Returns:
            dict: Normalized article dictionary with required fields.
        """
        title = raw.get("title")
        url = raw.get("href")
        content = raw.get("body")

        if not title or not url:
            return None

        return {
            "id": str(uuid.uuid4()),
            "title": title,
            "content": content or "",
            "url": url,
            "source": "web",
            "date": datetime.utcnow().isoformat(),
            "tags": [],
        }
