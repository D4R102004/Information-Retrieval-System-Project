"""
RAG Module Configuration

Centralizes all RAG configuration parameters.
Supports environment variables for production deployment.
"""

from pydantic import BaseModel, ConfigDict
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)


class RAGConfig(BaseModel):
    """RAG module configuration from environment variables."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # LLM Configuration
    ollama_model: str = "llama3.2:latest"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 300

    # RAG Configuration
    rag_template: str = "domain_specific"  # basic, domain_specific, chain_of_thought
    rag_retrieval_top_k: int = 5
    rag_temperature: float = 0.7
    rag_max_tokens: int = 1024
    rag_citation_threshold: float = 0.0
    max_snippet_length: int = 200 # Max chars from each document snippet in citations
    max_cites: int = 10 # Max number of citations to include in response
    response_char_limit: int = 2000 # Max chars in generated answer (truncated if no explicit [Answer] section)
    max_doc_content_length: int = 1000 # Max chars of document content to include in prompt (to prevent exceeding LLM context window)

    # Logging
    rag_log_level: str = "INFO"

    def validate_template(self) -> None:
        """Validate that template is valid."""
        valid_templates = ["basic", "domain_specific", "chain_of_thought"]
        if self.rag_template not in valid_templates:
            raise ValueError(
                f"Invalid template: {self.rag_template}. "
                f"Must be one of: {valid_templates}"
            )

    def __init__(self, **data):
        """Initialize config and validate."""
        super().__init__(**data)
        self.validate_template()
        logger.info(f"RAG Config loaded: template={self.rag_template}, ollama_model={self.ollama_model}")


# Global config instance
config = RAGConfig()
