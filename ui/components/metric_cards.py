"""Metric card helpers for evaluation output."""

from __future__ import annotations


def metric_card(label: str, value: float | int | str) -> dict[str, object]:
    """Return a simple metric card payload."""
    return {"label": label, "value": value}
