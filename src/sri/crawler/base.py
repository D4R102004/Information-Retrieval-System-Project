"""Base spider module — defines the contract for all crawlers.

Every spider in this system must inherit from BaseSpider and implement
fetch_articles(). This guarantees the pipeline can work with any spider
without knowing its internal details.
"""

import time
from abc import ABC, abstractmethod

import httpx

from sri.crawler.items import ArticleItem


class BaseSpider(ABC):
    """Abstract base class for all data acquisition spiders.

    Any class that inherits from BaseSpider MUST implement fetch_articles().
    This enforces a common interface across all spiders regardless of
    whether they use an API or HTML scraping internally.

    This follows SOLID principles:
        - L (Liskov): any spider can replace any other spider
        - D (Dependency Inversion): pipeline depends on this abstraction,
          not on concrete spider implementations
    """

    def __init__(self, max_articles: int = 500) -> None:
        """Initialise base spider with fetch limit.

        Args:
            max_articles: Maximum number of articles to fetch.
        """
        self.max_articles = max_articles

    @abstractmethod
    def fetch_articles(self) -> list[ArticleItem]:
        """Fetch articles from the data source.

        Returns:
            List of ArticleItem instances with all seven fields populated.
        """


class ApiSpider(BaseSpider):
    """Intermediate base class for API-based spiders.

    Provides shared HTTP infrastructure and pagination skeleton
    for spiders that fetch data from REST APIs. Subclasses must
    implement _search_terms(), _fetch_page(), and _build_item().
    """

    def __init__(self, max_articles: int = 500) -> None:
        """Initialise API spider with fetch limit and HTTP client.

        Args:
            max_articles: Maximum number of articles to fetch.
        """
        super().__init__(max_articles)

        self._client = httpx.Client(timeout=10.0)
        self._start_page = 1

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
            print(f"[{self.__class__.__name__}] HTTP error fetching {url}: {error}")
            return None

    def fetch_articles(self) -> list[ArticleItem]:
        """Fetch articles from the API page by page.

        Iterates through search terms and pages, collecting articles
        until max_articles is reached. Waits between requests to
        respect the API's rate limits.

        Returns:
            List of ArticleItem instances with all seven fields populated.
        """
        collected: list[ArticleItem] = []

        for term in self._search_terms():  # each spider provides its own terms
            page = self._start_page

            while len(collected) < self.max_articles:
                articles = self._fetch_page(term, page)

                if not articles:
                    break  # no more results for this term

                for raw_article in articles:
                    if len(collected) >= self.max_articles:
                        break

                    item = self._build_item(raw_article)
                    if item is not None:
                        collected.append(item)

                page += 1
                # Be polite — avoid hammering servers
                time.sleep(1)

        return collected

    @abstractmethod
    def _search_terms(self) -> list[str]:
        """Return the list of search terms to loop over.

        Returns:
            List of strings — tags or queries depending on the API.
        """

    @abstractmethod
    def _fetch_page(self, term: str, page: int) -> list[dict]:
        """Fetch a single page of articles for the given search term.

        Args:
            term: The search term to filter articles by.
            page: The page number to fetch.

        Returns:
            List of raw article dicts, or empty list on error.
        """

    @abstractmethod
    def _build_item(self, raw: dict) -> ArticleItem | None:
        """Translate a raw article dict into an ArticleItem.

        Args:
            raw: Raw article dictionary from the API response.

        Returns:
            Populated ArticleItem, or None if article cannot be built.
        """
