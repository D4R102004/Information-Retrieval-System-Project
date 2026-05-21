"""
RAG Module — Retrieval-Augmented Generation

Provides LLM-based answer generation augmented with retrieved documents.
Includes LLM provider abstraction.

Core components:
- LLMProvider: Abstract interface for LLM backends
"""

__version__ = "0.1.0"

from .llm_provider import LLMProvider, OllamaProvider

__all__ = [
    "LLMProvider",
    "OllamaProvider",
]
