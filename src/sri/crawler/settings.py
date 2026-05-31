"""Centralized configuration for crawler spiders and web sources."""


class CrawlerSettings:
    """Shared configuration values for crawler modules.

    This class centralizes HTTP settings, fetch limits, API endpoints,
    RSS feeds, and search terms used across all crawler spiders.
    """

    # HTTP
    HTTP_TIMEOUT: float = 10.0
    HTTP_SCRAPE_TIMEOUT: float = 30.0  # For heavy HTML pages

    # Fetch limits
    MAX_ARTICLES: int = 500
    PER_PAGE: int = 100

    # Dev.to
    DEVTO_API_URL: str = "https://dev.to/api/articles"
    DEVTO_TAGS: list[str] = [
        "python",
        "software",
        "programming",
        "webdev",
        "javascript",
    ]

    # Hacker News
    HN_API_URL: str = "https://hn.algolia.com/api/v1/search"
    HN_SEARCH_TERMS: list[str] = [
        "software",
        "python",
        "programming",
        "webdev",
        "javascript",
    ]

    # Lobsters
    LOBSTERS_BASE_URL: str = "https://lobste.rs"

    # RealPython
    REALPYTHON_BASE_URL: str = "https://realpython.com"
    REALPYTHON_SITEMAP_URL: str = f"{REALPYTHON_BASE_URL}/sitemap.xml"

    # RSS feeds
    THE_NEW_STACK_FEED: str = "https://thenewstack.io/feed/"
    THE_VERGE_FEED: str = "https://www.theverge.com/rss/index.xml"
