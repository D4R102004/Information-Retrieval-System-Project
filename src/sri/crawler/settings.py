"""Centralized configuration for crawler spiders and web sources."""
from typing import Any

class CrawlerSettings:
    """Shared configuration values for crawler modules.

    This class centralizes HTTP settings, fetch limits, API endpoints,
    RSS feeds, and search terms used across all crawler spiders.
    """
    _instance = None  # singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CrawlerSettings, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_instanced"):
            # private dictionary with all settings
            self._default: dict[str, Any] = {
                # http
                "http_timeout": 10.0,
                "http_scrape_timeout": 30.0,  # for heavy HTML pages

                # fetch limits
                "max_articles_per_spider": 500,
                "per_page": 100,

                # dev.to
                "devto_api_url": "https://dev.to/api/articles",
                "devto_tags": ["python", "software", "programming", "webdev", "javascript"],

                # hacker news
                "hn_api_url": "https://hn.algolia.com/api/v1/search",
                "hn_search_terms": ["software", "python", "programming", "webdev", "javascript"],

                # lobsters
                "lobsters_base_url": "https://lobste.rs",

                # realpython
                "realpython_base_url": "https://realpython.com",
                "realpython_sitemap_url": "https://realpython.com/sitemap.xml",

                # rss feeds
                "the_new_stack_feed": "https://thenewstack.io/feed/",
                "the_verge_feed": "https://www.theverge.com/rss/index.xml",

                # database sufficiency thresholds
                "min_documents_threshold": 1000,  # minimum docs for "sufficient" database
                "min_avg_score_threshold": 0.5,   # minimum average score for results
                "min_results_for_query": 3,       # minimum results before web search
                # TODO: assess using fraction of max documents instead of fixed number

                # auto-reload behavior
                "auto_reload": True,        # execute crawlers if DB insufficient
            }

            # mutable copy of default
            self._settings: dict[str, Any] = dict(self._default)

            self._instanced = True

    def __getitem__(self, key: str) -> Any:
        """Return the value of a key if it exists."""
        try:
            return self._settings.get(key.lower())
        except Exception:
            raise KeyError(f"Key '{key}' not found in CrawlerSettings")

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Try to modify a configuration value.
        Returns True if changed, False if the key does not exist or is of a different value.
        """
        key = key.lower()

        if not self.has(key):
            raise KeyError(f"Key '{key}' not found in CrawlerConfig")
        
        current_value = self[key]
        if isinstance(value, type(current_value)):
            self._settings[key] = value
        else:
            raise TypeError(
            f"Type mismatch for '{key}': expected {type(current_value).__name__}, got {type(value).__name__}"
        )

    def has(self, key: str) -> bool:
        """Returns True if settings has key"""
        return key.lower() in self._settings.keys()

    def all(self) -> dict[str, Any]:
        """Return a copy of all settings."""
        return dict(self._settings)

    def default(self, key: str) -> Any:
        """Return the default value for a given key."""
        key = key.lower()
        if key in self._default:
            return self._default[key]
        raise KeyError(f"Default value for '{key}' not found")
    
# Global singleton instance
crawler_settings = CrawlerSettings()
