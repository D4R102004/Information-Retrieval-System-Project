"""HackerNews spider — fetches technology articles from the Algolia public API.
This module contains the HackerNewsSpider class that retrieves articles
from Hacker News using their free public API, requiring no authentication.
API reference: http://hn.algolia.com/api/v1/search?tags=story&query=software
"""

import uuid

from sri.crawler.base import ApiSpider
from sri.crawler.items import ArticleItem


class HackerNewsSpider(ApiSpider):
    """Fetches technology articles from the Algolia public API.

    Retrieves articles page by page using the Hacker News public API,
    requiring no authentication or API key.

    Attributes:
        max_articles: Maximum number of articles to fetch in total.
        per_page: Number of articles to request per API call (max 100).
    """

    _start_page: int = 0  # Algolia uses 0-based pagination

    def __init__(self, max_articles: int = 500, per_page: int = 100) -> None:
        """Initialise the spider with fetch limits.

        Args:
            max_articles: Maximum number of articles to fetch in total.
            per_page: Number of articles to request per API call (max 100).
        """

        super().__init__(max_articles=max_articles)
        self.per_page = min(per_page, 100)

    def _search_terms(self) -> list[str]:
        """Return Algolia search queries for the technology domain.

        Returns:
            List of query strings to search for articles.
        """
        return ["software", "python", "programming", "webdev", "javascript"]

    def _fetch_page(self, term: str, page: int) -> list[dict]:
        """Fetch a single page of articles from the Algolia API.

        Args:
            term: The topic term to filter articles by.
            page: The page number to fetch (0-based).

        Returns:
            List of raw article dicts from the API, or empty list on error.
        """

        url = "https://hn.algolia.com/api/v1/search"

        params = {
            "tags": "story",  # only fetch stories, not comments
            "query": term,  # the search term
            "hitsPerPage": self.per_page,  # how many per page
            "page": page,  # which page
        }

        result = self._get_json(url, params=params)
        if not isinstance(result, dict) or "hits" not in result:
            return []

        return result.get("hits", [])

    def _build_item(self, raw_article: dict) -> ArticleItem | None:
        """Translate a raw Algolia article dict into an ArticleItem.

        Uses story_text as content when available, falls back to title
        for link posts that have no body text.

        Args:
            raw_article: Raw article dictionary from the Algolia hits list.

        Returns:
            Populated ArticleItem with all seven fields, or None if the
            article has no title.
        """

        if not raw_article.get("title"):
            return None

        article = ArticleItem()

        article["id"] = str(uuid.uuid4())
        article["title"] = raw_article.get("title", "")
        article["url"] = raw_article.get("url", "")
        article["date"] = raw_article.get("created_at", "")
        article["content"] = raw_article.get("story_text") or raw_article.get(
            "title", ""
        )

        article["source"] = "hackernews"
        article["tags"] = raw_article.get("_tags", [])

        return article
