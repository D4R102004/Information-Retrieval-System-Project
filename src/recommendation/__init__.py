"""Recommendation module for the SRI project."""

from .recommender import ContentBasedRecommender
from .user_history import UserSearchHistory

__all__ = ["ContentBasedRecommender", "UserSearchHistory"]
