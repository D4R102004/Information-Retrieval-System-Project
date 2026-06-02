"""Web search package — public API exports.

The package keeps web-search dependencies optional so local search,
evaluation, and recommendation can run even when DuckDuckGo packages are not
installed. Web search itself still requires the project dependencies.
"""

from sri.web_search.checker import SufficiencyChecker
from sri.web_search.indexer import WebResultIndexer

try:  # Optional dependency path: ddgs / duckduckgo_search may be absent.
    from sri.web_search.searcher import WebSearcher
except ModuleNotFoundError:
    WebSearcher = None  # type: ignore[assignment]

try:
    from sri.web_search.pipeline import WebSearchPipeline
except ModuleNotFoundError:
    WebSearchPipeline = None  # type: ignore[assignment]

__all__ = [
    "SufficiencyChecker",
    "WebResultIndexer",
    "WebSearchPipeline",
    "WebSearcher",
]
