"""Dev.to spider — fetches technology articles from the Dev.to public API.

This module contains the DevToSpider class that retrieves articles
from Dev.to using their free public API, requiring no authentication.

API reference: https://developers.forem.com/api/v1
"""

import uuid

from sri.crawler.base import ApiSpider
from sri.crawler.items import ArticleItem


class DevToSpider(ApiSpider):
    """Fetches technology articles from the Dev.to public API.

    Retrieves articles page by page using Dev.to's free REST API,
    requiring no authentication or API key.

    Attributes:
        max_articles: Maximum number of articles to fetch in total.
        per_page: Number of articles to request per API call (max 100).
    """

    def __init__(self, max_articles: int = 500, per_page: int = 100) -> None:
        """Initialise the spider with fetch limits.

        Args:
            max_articles: Maximum number of articles to fetch in total.
            per_page: Number of articles to request per API call (max 100).
        """
        super().__init__(
            max_articles=max_articles
        )  # Initialise the HTTP client from ApiSpider

        self.per_page = min(per_page, 100)  # Dev.to API max per

    def _fetch_page(self, tag: str, page: int) -> list[dict]:
        """Fetch a single page of articles from the Dev.to API.

        Args:
            tag: The topic tag to filter articles by.
            page: The page number to fetch (1-based).

        Returns:
            List of raw article dicts from the API, or empty list on error.
        """

        url = "https://dev.to/api/articles"
        params = {
            "tag": tag,
            "per_page": self.per_page,
            "page": page,
        }

        result = self._get_json(url, params=params)
        if not isinstance(result, list):
            return []
        return result

    def _fetch_full(self, article_id: int) -> dict:
        """Fetch the complete content of a single article from the Dev.to API.

        The articles list endpoint only returns summaries. This method fetches
        the full article including body_markdown, which is required for LSI
        to have enough text to analyze.

        Args:
            article_id: The numeric Dev.to article ID.

        Returns:
            Full article dict from the API, or empty dict on error.
        """

        url = f"https://dev.to/api/articles/{article_id}"

        result = self._get_json(url)
        if not isinstance(result, dict):
            return {}
        return result

    def _build_item(self, raw_article: dict) -> ArticleItem | None:
        """Translate a raw Dev.to article dict into an ArticleItem.

        Fetches the full article body separately since the list endpoint
        only returns summaries.

        Args:
            raw_article: Raw article dictionary from the Dev.to list endpoint.

        Returns:
            Populated ArticleItem with all seven fields, or None if the
            full article cannot be fetched.
        """

        body = self._fetch_full(raw_article["id"])
        if not body:
            return None

        article = ArticleItem()

        article["id"] = str(uuid.uuid4())
        article["title"] = body.get("title", "")
        article["url"] = body.get("url", "")
        article["date"] = body.get("published_at", "")
        article["content"] = body.get("body_markdown", "")
        article["source"] = "devto"
        article["tags"] = body.get("tag_list", [])

        return article

    def _search_terms(self) -> list[str]:
        """Return Dev.to topic tags for the technology domain.

        Returns:
            List of topic tags to search for articles.
        """

        # These tags cover the technology and software domain broadly
        tags = ["python", "software", "programming", "webdev", "javascript"]

        return tags
