"""Real Python spider — scrapes technology tutorials from realpython.com.

This module contains RealPythonSpider which retrieves article URLs
from the Real Python sitemap and extracts content via HTML scraping
using BeautifulSoup. Respects robots.txt before fetching any URL.
"""

# Standard library
import time
import uuid

# Third-party
import httpx
from bs4 import BeautifulSoup

# Local
from sri.crawler.base import BaseSpider
from sri.crawler.items import ArticleItem
from sri.crawler.settings import crawler_settings

import logging
logger = logging.getLogger(__name__)

class RealPythonSpider(BaseSpider):
    """Spider that scrapes articles from Real Python using its sitemap.

    This spider:
    - Loads URLs from the XML sitemap
    - Filters article URLs
    - Fetches each article HTML page
    - Extracts structured content via BeautifulSoup
    """

    def __init__(self, max_articles: int = 500) -> None:
        """Initializes HTTP client and crawl limits.

        Args:
            max_articles: Maximum number of articles to fetch.
        """
        super().__init__(max_articles)
        self._client = httpx.Client(timeout=crawler_settings["HTTP_SCRAPE_TIMEOUT"])

        self._base_url = crawler_settings["REALPYTHON_BASE_URL"]
        self._sitemap_url = crawler_settings["REALPYTHON_SITEMAP_URL"]

    def fetch_articles(self) -> list[ArticleItem]:
        """Fetch articles from Real Python sitemap and scrape content."""

        response = self._client.get(self._sitemap_url)
        soup = BeautifulSoup(response.text, "xml")

        urls = [
            loc.text
            for loc in soup.find_all("loc")
            if loc.text and loc.text.count("/") == 4 and "/tutorials/" not in loc.text
        ]

        collected: list[ArticleItem] = []

        for url in urls:
            if len(collected) >= self.max_articles:
                break

            if not self._can_fetch(url):
                continue

            try:
                article_response = self._client.get(url)
                article_soup = BeautifulSoup(article_response.text, "html.parser")

                item = self._build_item(url, article_soup)

                if item:
                    collected.append(item)

                time.sleep(1)

            except Exception as error:
                logger.debug(f"[RealPythonSpider] Failed to scrape {url}: {error}")

        return collected

    def _build_item(self, url: str, soup: BeautifulSoup) -> ArticleItem | None:
        """Builds an ArticleItem from parsed HTML."""

        title_tag = soup.find("h1")
        if not title_tag:
            return None

        date_tag = soup.find("time")
        content_tags = soup.find_all("p")

        tags = [
            a.get_text(strip=True)
            for a in soup.find_all("a")
            if a.get("href") and "/tag/" in a.get("href")
        ]

        item = ArticleItem()

        item["id"] = str(uuid.uuid4())
        item["title"] = title_tag.get_text(strip=True)
        item["url"] = url
        item["date"] = date_tag.get_text(strip=True) if date_tag else ""
        item["content"] = " ".join(p.get_text(strip=True) for p in content_tags)
        item["source"] = "realpython"
        item["tags"] = tags

        return item
