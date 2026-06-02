"""
    Main configuration file
"""
from typing import Any
from sri.crawler.settings import crawler_settings
from rag.config import rag_config

class MainConfig:
    """
    Main configuration holder (singleton).
    Wraps CrawlerSettings and RAGConfig, plus extra fields.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MainConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            # singleton sub-configs
            self._crawler_config = crawler_settings
            self._rag_config = rag_config

            # extra fields not present in crawler or rag
            self._default: dict[str, Any] = {
                "max_local_results": 5,
                "max_web_results": 10,
                "min_documents": 500, # Minimum documents in DB for local search to be considered sufficient
                "enable_web_search": True,
                "auto_reload": True,
                "query_min_length": 3,
                "query_max_length": 1000,
                "force_recrawl": False,
                "use_initial_corpus": True,
                "clear_raw": True
            }

            # mutable copy of default
            self._settings: dict[str, Any] = dict(self._default)

            self._initialized = True

    def __getitem__(self, key: str) -> Any:
        """Dict-like access to settings."""
        key = key.lower()
        if key in self._settings:
            return self._settings[key]
        
        # delegate to crawler config
        elif self._crawler_config.has(key):
            return self._crawler_config[key]
        
        # delegate to rag config
        elif self._rag_config.has(key):
            return self._rag_config[key]
        
        else:
            raise KeyError(f"Key '{key}' not found in MainConfig")

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-like assignment with type checking."""
        key = key.lower()
        if key in self._settings:
            current_value = self._settings[key]
            if isinstance(value, type(current_value)):
                self._settings[key] = value
            else:
                raise TypeError(
                    f"Type mismatch for '{key}': expected {type(current_value).__name__}, got {type(value).__name__}"
                )
        
        # delegate to crawler config
        elif self._crawler_config.has(key):
            self._crawler_config[key] = value

        # delegate to rag config
        elif self._rag_config.has(key):
            self._rag_config[key] = value

        else:
            raise KeyError(f"Key '{key}' not found in MainConfig")
        
    def has(self, key: str) -> bool:
        """Returns True if settings has key"""
        return (key in self._settings.keys() or 
               self._crawler_config.has(key) or 
               self._rag_config.has(key))

    def all(self) -> dict[str, Any]:
        """Return a merged dict of all settings."""
        merged = dict(self._settings)
        # merge crawler and rag configs
        merged.update(self._crawler_config.all())
        merged.update(self._rag_config.all())
        return merged
    
    def default(self, key: str) -> Any:
        """Return the default value for a given key."""
        key = key.lower()
        if key in self._default:
            return self._default[key]
        elif self._crawler_config.has(key):
            return self._crawler_config.default(key)
        elif self._rag_config.has(key):
            return self._rag_config.default(key)
        else:
            raise KeyError(f"Key '{key}' not found in MainConfig")

# Global singleton instance
main_config = MainConfig()