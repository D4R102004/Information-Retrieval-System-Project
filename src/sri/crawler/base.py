"""Base spider module — defines the contract for all crawlers.

Every spider in this system must inherit from BaseSpider and implement
fetch_articles(). This guarantees the pipeline can work with any spider
without knowing its internal details.
"""

from abc import ABC, abstractmethod

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

    @abstractmethod
    def fetch_articles(self) -> list[ArticleItem]:
        """Fetch articles from the data source.

        Returns:
            List of ArticleItem instances with all seven fields populated.
        """
