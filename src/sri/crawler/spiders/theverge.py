"""The Verge spider — fetches technology articles from The Verge RSS/Atom feed.

This module contains the TheVergeSpider class that retrieves articles
from The Verge feed using Atom XML parsing, requiring no authentication
or API key.

The feed provides structured metadata (<entry> nodes) and full article
content, which is extracted and transformed into ArticleItem objects for
the SRI pipeline.
"""

# Standard library
import uuid

# Third-party
import httpx
from bs4 import BeautifulSoup

# Local
from sri.crawler.base import BaseSpider
from sri.crawler.items import ArticleItem
from sri.crawler.settings import CrawlerSettings


class TheVergeSpider(BaseSpider):
    """Fetches technology articles from The Verge Atom feed.

    Parses the Atom feed and extracts article metadata and full content
    directly from <entry> nodes. No additional crawling is required since
    the feed contains complete article bodies.

    Attributes:
        max_articles: Maximum number of articles to fetch in total.
    """

    def __init__(
        self,
        max_articles: int = CrawlerSettings.MAX_ARTICLES,
    ) -> None:
        """Initialise the spider with fetch limits.

        Args:
            max_articles: Maximum number of articles to fetch in total.
        """
        super().__init__(max_articles)

        self._client = httpx.Client(
            timeout=CrawlerSettings.HTTP_TIMEOUT,
        )

    def fetch_articles(self) -> list[ArticleItem]:
        """Fetch and parse articles from The Verge Atom feed.

        This method performs a single HTTP request to the Atom feed,
        parses the XML response using BeautifulSoup, iterates over all
        <entry> nodes, and extracts:

        - title
        - url (from link[@rel="alternate"])
        - publication date
        - tags/categories
        - full article content from <content>

        Each entry is converted into an ArticleItem until max_articles is
        reached. No additional page crawling is required.

        Returns:
            List of ArticleItem objects ready for indexing.
        """

        collected: list[ArticleItem] = []

        response = self._client.get(
            CrawlerSettings.THE_VERGE_FEED,
        )

        soup = BeautifulSoup(response.text, "xml")

        entries = soup.find_all("entry")

        for entry in entries:
            if len(collected) >= self.max_articles:
                break

            item = self._build_item(entry)

            if item is not None:
                collected.append(item)

        return collected

    def _build_item(self, entry: "BeautifulSoup") -> ArticleItem | None:
        """Convert an Atom <entry> from The Verge feed into an ArticleItem.

        Extracts and transforms structured data from the Atom XML entry:

        - title: from <title>
        - url: from <link rel="alternate"> (href attribute)
        - date: from <published>
        - tags: from all <category term="..."> attributes
        - content: from <content> (HTML body parsed into plain text)

        The <content> field contains HTML markup which is parsed using
        BeautifulSoup to extract only readable paragraph text.

        Args:
            entry: BeautifulSoup element representing an Atom <entry> node.

        Returns:
            ArticleItem populated with extracted fields, or None if
            required fields are missing.
        """

        title_tag = entry.find("title")
        date_tag = entry.find("published")
        content_tag = entry.find("content")

        link_tag = entry.find("link", rel="alternate")

        if not title_tag or not date_tag or not content_tag or not link_tag:
            return None

        categories = entry.find_all("category")

        tags = [cat.get("term", "") for cat in categories if cat.get("term")]

        html_content = content_tag.get_text()

        content_soup = BeautifulSoup(html_content, "html.parser")

        paragraphs = content_soup.find_all("p")

        content = "\n".join(p.get_text(strip=True) for p in paragraphs)

        article = ArticleItem()

        article["id"] = str(uuid.uuid4())
        article["title"] = title_tag.get_text(strip=True)
        article["url"] = link_tag.get("href", "")
        article["date"] = date_tag.get_text(strip=True)
        article["tags"] = tags
        article["content"] = content
        article["source"] = "theverge"

        return article
