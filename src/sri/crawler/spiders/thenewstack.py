"""The New Stack spider — fetches technology articles from The New Stack RSS feed.

This module contains the TheNewStackSpider class that retrieves articles
from https://thenewstack.io/feed/ using their public RSS feed (XML),
requiring no authentication or API key.

The feed provides structured metadata and article links, which are later
processed to extract full content for indexing in the SRI pipeline.
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


class TheNewStackSpider(BaseSpider):
    """Fetches technology articles from The New Stack RSS feed.

    Parses the RSS feed at https://thenewstack.io/feed/ and extracts
    article metadata and full content directly from the XML fields
    (including content:encoded). No additional HTTP requests to article
    pages are required.

    The feed already contains complete article bodies, so this spider
    operates as a single-pass XML parser.
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
        """Fetch and parse articles from The New Stack RSS feed.

        This method performs a single HTTP request to the RSS endpoint,
        parses the XML response using BeautifulSoup, iterates over all
        <item> entries, and extracts:

        - title
        - link
        - publication date
        - tags/categories
        - full article content from content:encoded

        Each item is converted into an ArticleItem until max_articles is
        reached. No additional page crawling is required since the RSS
        feed contains complete article content.

        Returns:
            List of ArticleItem objects ready for indexing.
        """

        collected: list[ArticleItem] = []

        response = self._client.get(
            CrawlerSettings.THE_NEW_STACK_FEED,
        )

        soup = BeautifulSoup(response.text, "xml")

        items = soup.find_all("item")

        for raw_item in items:
            if len(collected) >= self.max_articles:
                break

            item = self._build_item(raw_item)

            if item is not None:
                collected.append(item)

        return collected

    def _build_item(self, raw_item: "BeautifulSoup") -> ArticleItem | None:
        """Convert a <item> from The New Stack RSS feed into an ArticleItem.

        Extracts metadata and full article content from the XML structure:

        - title: from <title>
        - url: from <link>
        - date: from <pubDate>
        - tags: from multiple <category> fields
        - content: from <content:encoded> (parsed as HTML)

        The <content:encoded> field contains HTML wrapped in CDATA.
        This method parses that HTML using BeautifulSoup and extracts only
        the visible text from <p> tags.

        Args:
            raw_item: BeautifulSoup element representing an RSS <item>.

        Returns:
            ArticleItem populated with extracted fields, or None if
            required fields are missing.
        """

        title_tag = raw_item.find("title")
        link_tag = raw_item.find("link")
        date_tag = raw_item.find("pubDate")

        content_tag = raw_item.find("encoded")

        if not title_tag or not link_tag or not date_tag or not content_tag:
            return None

        categories = raw_item.find_all("category")

        tags = [c.get_text(strip=True) for c in categories]

        html_content = content_tag.get_text()

        content_soup = BeautifulSoup(
            html_content,
            "html.parser",
        )

        paragraphs = content_soup.find_all("p")

        content = "\n".join(p.get_text(strip=True) for p in paragraphs)

        article = ArticleItem()

        article["id"] = str(uuid.uuid4())
        article["title"] = title_tag.get_text(strip=True)
        article["url"] = link_tag.get_text(strip=True)
        article["date"] = date_tag.get_text(strip=True)
        article["tags"] = tags
        article["content"] = content
        article["source"] = "thenewstack"

        return article
