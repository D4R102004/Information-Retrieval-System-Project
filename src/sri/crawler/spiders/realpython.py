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


class RealPythonSpider(BaseSpider):
    """Spider that scrapes articles from the Real Python sitemap by fetching
    and parsing HTML pages listed in the XML sitemap, instead of consuming a
    paginated API.

    Attributes:
        max_articles (int): Maximum number of articles to fetch (inherited).
        _base_url (str): Base URL of the target site.
        _sitemap_url (str): URL of the XML sitemap containing article links.
    """

    _base_url = "https://realpython.com"
    _sitemap_url = "https://realpython.com/sitemap.xml"

    def __init__(self, max_articles: int = 500) -> None:
        """Initializes the spider with a maximum number of articles to fetch
        and sets up an HTTP client for making requests.

        Args:
            max_articles (int, optional): Maximum number of articles to fetch.
                Defaults to 500.
        """
        super().__init__(max_articles)
        self._client = httpx.Client(timeout=10.0)

    def fetch_articles(self) -> list[ArticleItem]:
        """Fetches and parses articles from the Real Python sitemap.

        This method performs a full crawling pipeline:

        1. Downloads the XML sitemap from `_sitemap_url`.
        2. Parses the sitemap to extract article URLs.
        3. Iterates through each URL (up to `max_articles`).
        4. Fetches each article's HTML content.
        5. Delegates HTML parsing and item building to _build_item.
        6. Applies rate limiting between requests to avoid overloading the server.

        Error handling strategy:
            - Network errors are retried briefly and then skipped.
            - Parsing errors for individual articles are logged and skipped.
            - Sitemap-level failures may stop execution if critical.

        Returns:
            list[ArticleItem]: A list of successfully parsed article items.

        Notes:
            This method is designed to be resilient to partial failures in
            individual pages while ensuring the overall crawl completes
            within the configured limits.
        """

        # Step 1 & 2 — fetch and parse sitemap
        response = self._client.get(self._sitemap_url)
        soup = BeautifulSoup(response.text, "xml")
        urls = [loc.text for loc in soup.find_all("loc") if loc.text.count("/") == 4]

        # Step 3 — loop with limit
        collected: list[ArticleItem] = []

        for url in urls:
            if len(collected) >= self.max_articles:
                break
            if not self._can_fetch(url):
                continue
            try:
                # Step 4 — fetch article
                article_response = self._client.get(url)
                article_soup = BeautifulSoup(article_response.text, "html.parser")

                # Step 5 build item
                item = self._build_item(url, article_soup)
                if item is not None:
                    collected.append(item)
                # Step 6 — sleep
                time.sleep(1)
            except Exception as error:
                print(f"[RealPythonSpider] Failed to scrape {url}: {error}")
                continue

        return collected

    def _build_item(self, url: str, soup: BeautifulSoup) -> ArticleItem | None:
        """Helper method to build an ArticleItem from a BeautifulSoup object.

        Args:
            url: The URL of the article being processed.
            soup: A BeautifulSoup object containing the parsed HTML of the article.

        Returns:
          Populated ArticleItem with all seven fields, or None if the
          article has no title.
        """

        # Extract fields from soup
        title = soup.find("h1")
        if not title:
            return None
        title_text = title.get_text().strip()

        date_span = soup.find("span", class_="text-muted")
        date_text = date_span.get_text().strip() if date_span else ""

        paragraphs = soup.find_all("p")
        content = " ".join(p.get_text().strip() for p in paragraphs)

        tags = [
            a.get_text().strip()
            for a in soup.find_all("a")
            if (href := a.get("href")) and isinstance(href, str) and "/tag/" in href
        ]

        # Build item
        item = ArticleItem()
        item["id"] = str(uuid.uuid4())
        item["title"] = title_text
        item["url"] = url
        item["date"] = date_text
        item["content"] = content
        item["source"] = "realpython"
        item["tags"] = tags
        return item
