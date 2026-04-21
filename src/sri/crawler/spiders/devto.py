"""Dev.to spider — fetches technology articles from the Dev.to public API.

This module contains the DevToSpider class that retrieves articles
from Dev.to using their free public API, requiring no authentication.

API reference: https://developers.forem.com/api/v1
"""

import time
import uuid

import httpx

from sri.crawler.base import BaseSpider
from sri.crawler.items import ArticleItem


class DevToSpider(BaseSpider):
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

        self.max_articles = max_articles
        self.per_page = min(per_page, 100)  # Dev.to API max per
        # Reuse one connection for all requests — faster and more polite
        self._client = httpx.Client(timeout=10.0)

    def fetch_articles(self) -> list[ArticleItem]:
        """Fetch technology articles from the Dev.to API page by page.

        Iterates through API pages, collecting articles until max_articles
        is reached. Waits one second between requests to respect Dev.to's
        rate limits.

        Returns:
            List of ArticleItem instances with all seven fields populated.
        """

        # These tags cover the technology and software domain broadly
        tags = ["python", "software", "programming", "webdev", "javascript"]

        collected: list[ArticleItem] = []

        for tag in tags:
            page = 1

            while len(collected) < self.max_articles:
                articles = self._fetch_page(tag=tag, page=page)

                # Empty response means no more pages for this tag
                if not articles:
                    break

                for raw_article in articles:
                    if len(collected) >= self.max_articles:
                        break

                    item = self._build_item(raw_article)
                    if item is not None:
                        collected.append(item)

                page += 1
                # Be polite — avoid hammering Dev.to's servers
                time.sleep(1)

        return collected

    def _get_json(
        self,
        url: str,
        params: dict | None = None,
    ) -> dict | list | None:
        """Make a GET request and return the parsed JSON response.

        Single point of HTTP communication for this spider. All requests
        go through here so error handling lives in one place (DRY).

        Args:
            url: The endpoint URL to request.
            params: Optional query parameters to append to the URL.

        Returns:
            Parsed JSON as dict or list, or None on any HTTP error.
        """

        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as error:
            print(f"[DevToSpider] HTTP error fetching {url}: {error}")
            return None

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
