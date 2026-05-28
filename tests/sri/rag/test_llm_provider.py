"""
Tests for LLM Provider abstraction.

Tests cover:
- OllamaProvider initialization and validation
- LLM availability checking
- Response generation
- Error handling and timeouts
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
import requests

from src.rag.llm_provider import LLMProvider, OllamaProvider


class TestLLMProviderInterface:
    """Test LLMProvider abstract interface."""

    def test_llm_provider_is_abstract(self):
        """LLMProvider should not be instantiable."""
        with pytest.raises(TypeError):
            LLMProvider()

    def test_llm_provider_requires_generate(self):
        """Subclass must implement generate method."""

        class IncompleteLLM(LLMProvider):
            def is_available(self):
                return True

        # Should raise TypeError for missing abstract method
        with pytest.raises(TypeError):
            IncompleteLLM()


class TestOllamaProviderInitialization:
    """Test OllamaProvider initialization and validation."""

    @patch("requests.get")
    def test_initialization_success(self, mock_get):
        """Should initialize successfully when Ollama is available."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama3.2:latest"}, {"name": "llama2:latest"}]
        }
        mock_get.return_value = mock_response

        provider = OllamaProvider(model="llama3.2:latest")

        assert provider.model == "llama3.2:latest"
        assert provider.base_url == "http://localhost:11434"
        assert mock_get.call_count >= 2  # Connection + model validation

    @patch("requests.get")
    def test_initialization_no_connection(self, mock_get):
        """Should raise RuntimeError if Ollama service unreachable."""
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            OllamaProvider()

    @patch("requests.get")
    def test_initialization_model_not_found(self, mock_get):
        """Should raise ValueError if model not available."""

        def side_effect(url, **kwargs):
            response = Mock()
            response.status_code = 200
            response.json.return_value = {
                "models": [{"name": "llama2:latest"}]  # llama3.2:latest not available
            }
            return response

        mock_get.side_effect = side_effect

        with pytest.raises(ValueError, match="Model.*not found"):
            OllamaProvider(model="llama3.2:latest")

    @patch("requests.get")
    def test_custom_base_url(self, mock_get):
        """Should support custom Ollama base URL."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "llama3.2:latest"}]}
        mock_get.return_value = mock_response

        provider = OllamaProvider(
            base_url="http://192.168.1.100:11434",
            model="llama3.2:latest",
        )

        assert provider.base_url == "http://192.168.1.100:11434"

    @patch("requests.get")
    def test_timeout_configuration(self, mock_get):
        """Should support custom timeout."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "llama3.2:latest"}]}
        mock_get.return_value = mock_response

        provider = OllamaProvider(timeout=600)

        assert provider.timeout == 600


class TestOllamaProviderGeneration:
    """Test LLM generation functionality."""

    @patch("requests.get")
    @patch("requests.post")
    def test_generate_success(self, mock_post, mock_get):
        """Should generate text successfully."""
        # Mock initialization
        init_response = Mock()
        init_response.status_code = 200
        init_response.json.return_value = {"models": [{"name": "llama3.2:latest"}]}
        mock_get.return_value = init_response

        # Mock generation
        gen_response = Mock()
        gen_response.json.return_value = {
            "response": "Python is a programming language."
        }
        mock_post.return_value = gen_response

        provider = OllamaProvider()
        result = provider.generate("What is Python?", temperature=0.7)

        assert "Python is a programming language" in result
        mock_post.assert_called_once()

    @patch("requests.get")
    @patch("requests.post")
    def test_generate_with_max_tokens(self, mock_post, mock_get):
        """Should pass max_tokens to API."""
        init_response = Mock()
        init_response.status_code = 200
        init_response.json.return_value = {"models": [{"name": "llama3.2:latest"}]}
        mock_get.return_value = init_response

        gen_response = Mock()
        gen_response.json.return_value = {"response": "Short answer."}
        mock_post.return_value = gen_response

        provider = OllamaProvider()
        provider.generate("Query", max_tokens=100)

        # Check that num_predict was set
        call_args = mock_post.call_args
        assert call_args[1]["json"]["num_predict"] == 100

    @patch("requests.get")
    @patch("requests.post")
    def test_generate_timeout(self, mock_post, mock_get):
        """Should raise RuntimeError on timeout."""
        init_response = Mock()
        init_response.status_code = 200
        init_response.json.return_value = {"models": [{"name": "llama3.2:latest"}]}
        mock_get.return_value = init_response

        mock_post.side_effect = requests.exceptions.Timeout("Request timeout")

        provider = OllamaProvider(timeout=5)

        with pytest.raises(RuntimeError, match="generation timeout"):
            provider.generate("Query")

    @patch("requests.get")
    @patch("requests.post")
    def test_generate_request_error(self, mock_post, mock_get):
        """Should raise RuntimeError on request error."""
        init_response = Mock()
        init_response.status_code = 200
        init_response.json.return_value = {"models": [{"name": "llama3.2:latest"}]}
        mock_get.return_value = init_response

        mock_post.side_effect = requests.exceptions.RequestException("Network error")

        provider = OllamaProvider()

        with pytest.raises(RuntimeError, match="generation failed"):
            provider.generate("Query")


class TestOllamaProviderAvailability:
    """Test availability checking."""

    @patch("requests.get")
    def test_is_available_true(self, mock_get):
        """Should return True when service is accessible."""
        response = Mock()
        response.status_code = 200
        mock_get.return_value = response

        provider_instance = MagicMock(spec=OllamaProvider)
        provider_instance.base_url = "http://localhost:11434"
        provider_instance.verify_ssl = False

        # Manually call is_available
        is_available = OllamaProvider.is_available(provider_instance)

        assert is_available is True

    @patch("requests.get")
    def test_is_available_false(self, mock_get):
        """Should return False when service is unreachable."""
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        provider_instance = MagicMock(spec=OllamaProvider)
        provider_instance.base_url = "http://localhost:11434"
        provider_instance.verify_ssl = False

        is_available = OllamaProvider.is_available(provider_instance)

        assert is_available is False


class TestMetadata:
    """Test metadata retrieval."""

    @patch("requests.get")
    def test_get_metadata(self, mock_get):
        """Should return provider metadata."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"models": [{"name": "llama3.2:latest"}]}
        mock_get.return_value = response

        provider = OllamaProvider(model="llama3.2:latest")
        metadata = provider.get_metadata()

        assert metadata["provider"] == "Ollama"
        assert metadata["model"] == "llama3.2:latest"
        assert metadata["type"] == "local"
        assert metadata["reproducible"] is True
        assert metadata["requires_api_key"] is False
