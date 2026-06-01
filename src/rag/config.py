"""
RAG Module Configuration

Centralizes all RAG configuration parameters.
Supports environment variables for production deployment.
"""

from pydantic import ConfigDict
from typing import Any, Literal
import logging

logger = logging.getLogger(__name__)

class RAGConfig:
    """RAG module configuration from environment variables."""

    # singleton instance
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            # LLM Configuration
            self.ollama_model: str = "llama3.2:latest"
            self.ollama_base_url: str = "http://localhost:11434"
            self.ollama_timeout: int = 300

            # RAG Configuration
            self.rag_template: Literal["basic", "domain_specific", "chain_of_thought"] = "domain_specific"
            self.rag_temperature: float = 0.7
            self.rag_max_tokens: int = 1024
            self.rag_citation_threshold: float = 0.0
            self.max_snippet_length: int = 200 # Max chars from each document snippet in citations
            self.max_cites: int = 10 # Max number of citations to include in response
            self.response_char_limit: int = 2000 # Max chars in generated answer (truncated if no explicit [Answer] section)
            self.max_context_doc_length: int = 1000 # Max chars of document content to include in prompt (to prevent exceeding LLM context window)

            # Logging
            self.rag_log_level: str = "INFO"

            # Defaults snapshot
            self._default: dict[str, Any] = {
                k: v for k, v in self.__dict__.items()
                if not k.startswith("_")
            }

            self._initialized = True

    def validate_template(self) -> None:
        """Validate that template is valid."""
        valid_templates = ["basic", "domain_specific", "chain_of_thought"]
        if self.rag_template not in valid_templates:
            raise ValueError(
                f"Invalid template: {self.rag_template}. Must be one of: {valid_templates}"
            )

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to attributes."""
        key = key.lower()
        if not hasattr(self, key):
            raise KeyError(f"Key '{key}' not found in RAGConfig")
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-like assignment with type checking."""
        key = key.lower()
        if not hasattr(self, key):
            raise KeyError(f"Key '{key}' not found in RAGConfig")

        current_value = getattr(self, key)
        if isinstance(value, type(current_value)):
            setattr(self, key, value)
            if(key == "rag_template"):
                self.validate_template()
        else:
            raise TypeError(
                f"Type mismatch for '{key}': expected {type(current_value).__name__}, got {type(value).__name__}"
            )
        
    def default(self, key: str) -> Any:
        """Return the default value for a given key."""
        key = key.lower()
        if key in self._default:
            return self._default[key]
        raise KeyError(f"Default value for '{key}' not found")

# Global singleton instance
config = RAGConfig()
