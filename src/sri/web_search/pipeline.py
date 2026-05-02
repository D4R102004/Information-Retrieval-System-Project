"""Hybrid search pipeline that combines local results with web fallback.

This module implements an orchestration layer that decides whether to
return precomputed local search results or fall back to a web search
engine when local results are insufficient.

The pipeline follows a simple decision flow:
    1. Receive a query and local search results.
    2. Ask a checker component if local results are sufficient.
    3. If sufficient, return local results directly.
    4. If not sufficient:
        - Perform a web search using a searcher component.
        - Persist each result using an indexer.
        - Return the web-derived results.

This design ensures low latency when possible (local-first strategy),
while maintaining completeness via web fallback when needed.
"""

from sri.web_search.checker import SufficiencyChecker
from sri.web_search.indexer import WebResultIndexer
from sri.web_search.searcher import WebSearcher


class WebSearchPipeline:
    """Orchestrates local search validation, web search fallback, and indexing.

    This class acts as the central coordinator of the search system. It does
    not implement search, validation, or persistence itself, but delegates
    these responsibilities to injected components:

        - checker: Determines whether local results are sufficient.
        - searcher: Executes web search when fallback is required.
        - indexer: Persists web search results to disk.

    The pipeline enforces a local-first strategy:
        - If local results are sufficient, they are returned immediately.
        - Otherwise, web search is triggered, results are stored, and returned.

    This separation of concerns allows for high modularity and testability.
    """

    def __init__(
        self,
        checker: SufficiencyChecker,
        searcher: WebSearcher,
        indexer: WebResultIndexer,
    ) -> None:
        """Initializes the pipeline with its component dependencies.

        Args:
            checker: An instance of a sufficiency checker.
            searcher: An instance of a web searcher.
            indexer: An instance of a result indexer.
        """
        self.checker = checker
        self.searcher = searcher
        self.indexer = indexer

    def search(self, query: str, local_results: list[dict]) -> list[dict]:
        """
        Runs the hybrid search strategy for a given query.

        Args:
            query: The user's search query.
            local_results: Results already retrieved from the local index.

        Returns:
            Local results if sufficient, otherwise web search results.
        """

        # Step 1: Validate local results

        if self.checker.is_sufficient(local_results):
            return local_results

        # Step 2: Fallback to web search
        web_results = self.searcher.search(query)

        # Step 3: Persist results
        for article in web_results:
            self.indexer.save_article(article)

        return web_results
