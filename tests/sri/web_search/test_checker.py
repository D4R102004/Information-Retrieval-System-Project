"""
Tests for the SufficiencyChecker class.
Verifies that the checker correctly identifies sufficient and insufficient
local results based on result count and relevance score thresholds.
No mocks needed — SufficiencyChecker has no external dependencies.
"""

from sri.web_search.checker import SufficiencyChecker


def test_checker_checks_empty():
    """
    Checks wether SufficiencyChecker works with empty list
    """

    # Arrange

    checker = SufficiencyChecker()

    # Act

    result = []

    # Assert
    assert not checker.is_sufficient(result)


def test_checker_checks_low_count():
    """
    Checks wether SufficiencyChecker works with low count
    """

    checker = SufficiencyChecker()

    results = [
        {"score": 0.8},
        {"score": 0.7},
    ]

    assert not checker.is_sufficient(results)


def test_checker_checks_low_score():
    """
    Checks wether SufficiencyChecker works with low score
    """

    checker = SufficiencyChecker()

    results = [
        {"score": 0.4},
        {"score": 0.3},
        {"score": 0.2},
    ]

    assert not checker.is_sufficient(results)


def test_checker_checks_sufficient():
    """
    Checks wether SufficiencyChecker works with sufficient results
    """

    checker = SufficiencyChecker()

    results = [
        {"score": 0.6},
        {"score": 0.7},
        {"score": 0.4},
    ]

    assert checker.is_sufficient(results)
