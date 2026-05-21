"""
LLM Provider Abstraction

Implements vendor-agnostic interface for Large Language Models,
supporting multiple backends (Ollama, OpenAI, HuggingFace, etc.).

Uses Strategy pattern to enable seamless migration between providers
without modifying RAG module logic.
"""

import requests
import os
from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for Language Model providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate text from the LLM.

        Args:
            prompt: Input prompt text
            temperature: Creativity parameter (0.0-1.0)
            max_tokens: Maximum output length
            top_p: Nucleus sampling parameter

        Returns:
            Generated text response

        Raises:
            RuntimeError: If generation fails
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is accessible."""
        pass

    def get_metadata(self) -> dict:
        """Return provider metadata (model name, version, etc.)"""
        return {}


class OllamaProvider(LLMProvider):
    """Interface to Ollama local LLM service.

    Ollama (https://ollama.ai) provides local execution of LLMs
    without requiring API keys or external services, ensuring
    reproducibility and privacy.

    Supported models: llama3.2, llama2, mistral, neural-chat, etc.
    """

    def __init__(
        self,
        model: str = "llama3.2:latest",
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        verify_ssl: bool = False,
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Model identifier (default: llama3.2:latest)
            base_url: Ollama service endpoint
            timeout: Request timeout in seconds (default: 300)
            verify_ssl: SSL verification for requests (default: False)

        Raises:
            RuntimeError: If Ollama service is not running
            ValueError: If specified model is not available
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl

        logger.info(f"Initializing OllamaProvider with model: {self.model}")
        self._validate_connection()
        self._validate_model()

    def _validate_connection(self) -> None:
        """Verify that Ollama service is accessible."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            logger.info(f"Successfully connected to Ollama at {self.base_url}")
        except requests.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start with: ollama serve"
            ) from e
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama connection failed: {e}") from e

    def _validate_model(self) -> None:
        """Verify that the specified model is available in Ollama."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            available_models = [m["name"] for m in response.json().get("models", [])]

            if self.model not in available_models:
                raise ValueError(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available models: {available_models}. "
                    f"Pull model with: ollama pull {self.model.split(':')[0]}"
                )
            logger.info(f"Model {self.model} is available")
        except (requests.RequestException, ValueError) as e:
            logger.error(f"Model validation failed: {e}")
            raise

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
    ) -> str:
        """Generate text using Ollama.

        Args:
            prompt: Input prompt
            temperature: Creativity (0.0=deterministic, 1.0=random)
            max_tokens: Maximum output tokens
            top_p: Nucleus sampling threshold

        Returns:
            Generated text

        Raises:
            RuntimeError: If generation fails
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }

        if max_tokens:
            payload["num_predict"] = max_tokens

        try:
            logger.debug(f"Sending request to Ollama... (prompt length: {len(prompt)})")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            logger.debug(f"Generated response ({len(result)} chars)")
            return result
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama generation timeout after {self.timeout}s. "
                "Consider increasing timeout or breaking prompt into smaller parts."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

    def is_available(self) -> bool:
        """Check if Ollama service is running and accessible.

        Returns:
            True if service is available, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=2,
                verify=self.verify_ssl,
            )
            return response.status_code == 200
        except requests.ConnectionError:
            logger.warning("Ollama service is not available")
            return False

    def get_metadata(self) -> dict:
        """Return Ollama provider metadata.

        Returns:
            Dictionary with provider information
        """
        return {
            "provider": "Ollama",
            "model": self.model,
            "base_url": self.base_url,
            "type": "local",
            "reproducible": True,
            "requires_internet": False,
            "requires_api_key": False,
        }
