"""Base spider module — defines the contract for all crawlers.

Every spider in this system must inherit from BaseSpider and implement
fetch_articles(). This guarantees the pipeline can work with any spider
without knowing its internal details.
"""

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

    def __init__(self):
        """
        Initialise shared HTTP client for all spiders.
        """
        # Reuse one connection for all requests — faster and more polite
        self._client = httpx.Client(timeout=10.0)

    @abstractmethod
    def fetch_articles(self) -> list[ArticleItem]:
        """Fetch articles from the data source.

        Returns:
            List of ArticleItem instances with all seven fields populated.
        """

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
