"""Web search module using DuckDuckGo.
This module provides a simple interface to perform web searches using
DuckDuckGo and transform the results into a standardized article format.
"""

# Standard library
import uuid
from datetime import datetime, timezone

# Third-party
import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


class WebSearcher:
    """Search the web and return normalized article data."""

    def __init__(self, max_results: int = 10) -> None:
        """Initialize the searcher.

        Args:
            max_results:
                Maximum number of search results to retrieve.
        """
        self.max_results = max_results

    def search(self, query: str) -> list[dict]:
        """Perform a web search and return normalized article results.

        Args:
            query:
                Search query string.

        Returns:
            A list of normalized article dictionaries.
        """
        results: list[dict] = []
        with DDGS() as ddgs:
            raw_results = ddgs.text(
                query,
                max_results=self.max_results,
            )
            for raw in raw_results:
                article = self._build_article(raw)
                if article:
                    results.append(article)
        return results

    def _fetch_full(self, url: str) -> str:
        """Fetch and extract full article content from a webpage.

        Args:
            url:
                URL of the article page.

        Returns:
            Extracted article text, or an empty string if fetching fails.
        """
        try:
            response = httpx.get(
                url,
                timeout=10.0,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join(paragraph.get_text(strip=True) for paragraph in paragraphs)
        return content.strip()

    def _build_article(self, raw: dict) -> dict | None:
        """Convert a DuckDuckGo result into the article contract format.

        Args:
            raw:
                Raw result returned by DuckDuckGo.

        Returns:
            A normalized article dictionary, or None if invalid.
        """
        title = raw.get("title")
        url = raw.get("href")

        if not title or not url:
            return None

        content = self._fetch_full(url)
        if not content:
            return None

        return {
            "id": str(uuid.uuid4()),
            "title": title,
            "content": content,
            "url": url,
            "source": "web",
            "date": datetime.now(timezone.utc).isoformat(),
            "tags": [],
        }
