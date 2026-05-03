"""
Sufficiency Checker — decides whether local results are good enough.

If local results fail either the minimum-count or the minimum-score
threshold, the web-search fallback should be triggered.
"""

# Threshold recommended by production RAG literature (score 0.0–1.0)
DEFAULT_SCORE_THRESHOLD = 0.55
DEFAULT_MIN_RESULTS = 3


class SufficiencyChecker:
    """Decides whether a set of local retrieval results is sufficient.

    Two criteria must both pass for results to be considered sufficient:
    1. There are at least ``min_results`` results.
    2. The best result has a score >= ``score_threshold``.

    """

    def __init__(
        self,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        min_results: int = DEFAULT_MIN_RESULTS,
    ) -> None:
        """Initializes the result filtering configuration.

        This constructor sets the parameters that control the minimum
        relevance score and the minimum number of results to return.

        Args:
            score_threshold (float, optional): Minimum score a result must
                have to be considered valid. Defaults to DEFAULT_SCORE_THRESHOLD.
            min_results (int, optional): Minimum number of results to return,
                even if some do not meet the score threshold.
                Defaults to DEFAULT_MIN_RESULTS.
        """
        self.score_threshold = score_threshold
        self.min_results = min_results

    def is_sufficient(self, results: list[dict]) -> bool:
        """Return True if local results are good enough to skip web search.

        Args:
            results: List of result dicts, each must contain a ``score`` key.

        Returns:
            True if results pass both quantity and quality thresholds."""

        if not results:
            return False
        if len(results) < self.min_results:
            return False
        best_score = max(result["score"] for result in results)
        return best_score >= self.score_threshold
