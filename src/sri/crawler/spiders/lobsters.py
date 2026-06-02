"""Lobste.rs spider — fetches technology articles from the Lobste.rs community API.

This module contains the LobstersSpider class that retrieves articles
from Lobste.rs using their free public JSON API, requiring no authentication.

API reference: https://lobste.rs/hottest.json
"""

# Standard library
import uuid

# Third-party
from bs4 import BeautifulSoup

# Local
from sri.crawler.base import ApiSpider
from sri.crawler.items import ArticleItem
from sri.crawler.settings import crawler_settings


import logging
logger = logging.getLogger(__name__)


class LobstersSpider(ApiSpider):
    """Fetches technology articles from the Lobste.rs public API.

    Retrieves articles page by page using Lobste.rs's free JSON API,
    requiring no authentication or API key.

    Attributes:
        max_articles: Maximum number of articles to fetch in total.
    """

    def __init__(
        self,
        max_articles: int = crawler_settings["MAX_ARTICLES_PER_SPIDER"],
    ) -> None:
        """Initialise the spider with fetch limits.

        Args:
            max_articles: Maximum number of articles to fetch in total.
        """

        super().__init__(max_articles)

    def _search_terms(self) -> list[str]:
        """Return a placeholder search term to trigger a single crawl loop.

        Lobste.rs does not filter by tag — articles are fetched by popularity.
        A single empty string is returned to run the pagination loop once.

        Returns:
            List with a single empty string.
        """

        return [""]

    def _fetch_page(self, term: str, page: int) -> list[dict]:
        """Fetch a single page of articles from the Lobste.rs API.

        Lobste.rs does not support filtering by topic tag. The ``term``
        parameter is accepted only to match the shared spider interface
        used by ApiSpider.

        Args:
            term: Unused placeholder search term.
            page: The page number to fetch (1-based).

        Returns:
            List of raw article dicts from the API, or empty list on error.
        """

        if page == 1:
            url = f"{crawler_settings["LOBSTERS_BASE_URL"]}" "/hottest.json"
        else:
            url = f"{crawler_settings["LOBSTERS_BASE_URL"]}" f"/hottest/page/{page}.json"

        result = self._get_json(url)

        if not isinstance(result, list):
            return []

        return result

    def _build_item(self, raw_article: dict) -> ArticleItem | None:
        """Translate a raw Lobste.rs article dict into an ArticleItem.

        The Lobste.rs API only returns metadata and external URLs. This method
        visits the linked external article page to extract the full text content
        required for LSI analysis.

        Args:
            raw_article: Raw article dictionary from the Lobste.rs API.

        Returns:
            Populated ArticleItem with all seven fields, or None if the
            external article content cannot be retrieved.
        """

        article = ArticleItem()

        article["id"] = str(uuid.uuid4())
        article["title"] = raw_article.get("title", "")
        article["url"] = raw_article.get("url", "")
        article["date"] = raw_article.get("created_at", "")
        article["source"] = "lobsters"
        article["tags"] = raw_article.get("tags", [])

        try:
            response = self._client.get(article["url"])

            soup = BeautifulSoup(response.text, "html.parser")

            paragraphs = soup.find_all("p")

            article["content"] = " ".join(p.get_text(strip=True) for p in paragraphs)

        except Exception as error:
            logger.debug(f"[LobstersSpider] Failed to scrape " f"{article['url']}: {error}")
            return None

        return article
