"""Centralized configuration for crawler spiders and web sources."""


class CrawlerSettings:
    """Shared configuration values for crawler modules.

    This class centralizes HTTP settings, fetch limits, API endpoints,
    RSS feeds, and search terms used across all crawler spiders.
    """

    # HTTP
    HTTP_TIMEOUT: float = 10.0

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

    # Database sufficiency thresholds
    MIN_DOCUMENTS_THRESHOLD: int = 1000  # Minimum docs for "sufficient" database
    MIN_AVG_SCORE_THRESHOLD: float = 0.5  # Minimum average score for results
    MIN_RESULTS_FOR_QUERY: int = 3  # Minimum results before web search
    #TODO: asses using fraction of max documents (specified in query) instead of fixed number for MIN_RESULTS_FOR_QUERY

    # Auto-reload behavior
    AUTO_CRAWL_ON_EMPTY: bool = True  # Execute crawlers if DB empty
    AUTO_CRAWL_ON_INSUFFICIENT: bool = True  # Execute if results insufficient
