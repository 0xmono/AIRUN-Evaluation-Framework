"""Tests for the OllamaAdapter class."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

# Add the src directory to Python's module search path, to run this file as `python evaluate.py`
# import os
# import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
# src_dir = os.path.join(project_dir, "src")
# sys.path.append(src_dir)

from epam.auto_llm_eval.ollama_adapter import OllamaAdapter


# Setup fixtures
@pytest.fixture
def mock_ollama_response():
    return {
        "model": "llama3",
        "response": "This is a test response",
        "total_duration": 1000000000,  # 1 second in nanoseconds
        "load_duration": 200000000,  # 0.2 seconds in nanoseconds
        "sample_count": 100,
        "eval_count": 300,
        "eval_duration": 800000000,  # 0.8 seconds in nanoseconds
    }


class TestOllamaAdapter:
    """Test suite for OllamaAdapter."""

    def test_init_params(self):
        """Test that adapter initialization sets parameters correctly."""
        adapter = OllamaAdapter(
            model_name="llama3",
            temperature=0.5,
            top_p=0.8,
            top_logprobs=3,
            logprobs=True,
            api_base="http://localhost:12345"
        )

        assert adapter.model_name == "llama3"
        assert adapter.temperature == 0.5
        assert adapter.top_p == 0.8
        assert adapter.top_logprobs == 3
        assert adapter.logprobs is True
        assert adapter.api_base == "http://localhost:12345"

    def test_inheritance(self):
        """Test that OllamaAdapter inherits from the correct base classes."""
        adapter = OllamaAdapter(model_name="llama3")

        assert isinstance(adapter, BaseChatModel)
        assert isinstance(adapter, Runnable)

    def test_llm_type(self):
        """Test the _llm_type method."""
        adapter = OllamaAdapter(model_name="llama3")

        assert adapter._llm_type() == "ollama"

    def test_convert_messages_to_prompt(self):
        """Test the message to prompt conversion."""
        adapter = OllamaAdapter(model_name="llama3")

        messages = [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="Hello, how are you?"),
            AIMessage(content="I'm doing well, thank you for asking!")
        ]

        prompt = adapter._convert_messages_to_prompt(messages)

        assert "System: You are a helpful AI assistant." in prompt
        assert "Human: Hello, how are you?" in prompt
        assert "Assistant: I'm doing well, thank you for asking!" in prompt

    @patch("ollama.generate")
    def test_generate(self, mock_generate, mock_ollama_response):
        """Test the _generate method."""
        mock_generate.return_value = mock_ollama_response

        adapter = OllamaAdapter(model_name="llama3")

        messages = [
            HumanMessage(content="Tell me a joke")
        ]

        result = adapter._generate(messages)

        # Check that ollama.generate was called with correct parameters
        mock_generate.assert_called_once()
        call_args = mock_generate.call_args[1]
        assert call_args["model"] == "llama3"
        assert "Tell me a joke" in call_args["prompt"]

        # Check result structure
        assert result.generations[0].message.content == "This is a test response"
        assert result.generations[0].generation_info["total_duration"] == 1000000000

    @patch("ollama.generate")
    def test_generate_with_stop(self, mock_generate, mock_ollama_response):
        """Test _generate with stop tokens."""
        mock_generate.return_value = mock_ollama_response

        adapter = OllamaAdapter(model_name="llama3")

        messages = [
            HumanMessage(content="Count from 1 to 10")
        ]

        result = adapter._generate(messages, stop=["5"])

        # Check that stop tokens were passed
        call_args = mock_generate.call_args[1]
        assert call_args["options"]["stop"] == ["5"]

    def test_simulate_logprobs(self):
        """Test the logprobs simulation."""
        adapter = OllamaAdapter(model_name="llama3", top_logprobs=5, logprobs=True)

        logprobs = adapter._simulate_logprobs()

        assert len(logprobs) == 5
        assert logprobs[0]["token"] == "4"  # Highest probability

        # Check that probabilities descend
        probs = [np.exp(item["logprob"]) for item in logprobs]
        assert all(probs[i] >= probs[i+1] for i in range(len(probs)-1))

    def test_bind(self):
        """Test the bind method."""
        adapter = OllamaAdapter(
            model_name="llama3",
            temperature=0.0,
            top_p=1.0
        )

        # Create a new instance with different parameters
        new_adapter = adapter.bind(
            temperature=0.7,
            top_logprobs=3,
            logprobs=True
        )

        # Check that the original instance is unchanged
        assert adapter.temperature == 0.0
        assert adapter.top_logprobs is None
        assert adapter.logprobs is None

        # Check that the new instance has the updated parameters
        assert new_adapter.model_name == "llama3"  # Unchanged
        assert new_adapter.temperature == 0.7  # Changed
        assert new_adapter.top_logprobs == 3  # Changed
        assert new_adapter.logprobs is True  # Changed

    @patch("ollama.generate")
    def test_invoke(self, mock_generate, mock_ollama_response):
        """Test the invoke method."""
        mock_generate.return_value = mock_ollama_response

        adapter = OllamaAdapter(model_name="llama3")

        messages = [
            HumanMessage(content="What is the capital of France?")
        ]

        result = adapter.invoke(messages)

        # Check that the result is an AIMessage
        assert isinstance(result, AIMessage)
        assert result.content == "This is a test response"

    @patch("ollama.generate")
    def test_invoke_with_logprobs(self, mock_generate, mock_ollama_response):
        """Test invoke with logprobs enabled."""
        mock_generate.return_value = mock_ollama_response

        adapter = OllamaAdapter(
            model_name="llama3",
            logprobs=True,
            top_logprobs=5
        )

        messages = [
            HumanMessage(content="Give me one word: excellent, good, fair, poor, or bad")
        ]

        # This should include logprobs in the generation info
        result = adapter._generate(messages)

        # Check that logprobs are included
        assert "logprobs" in result.generations[0].generation_info
        content_logprobs = result.generations[0].generation_info["logprobs"]["content"]
        assert len(content_logprobs) == 1
        assert len(content_logprobs[0]["top_logprobs"]) == 5

    def test_identifying_params(self):
        """Test the _identifying_params property."""
        adapter = OllamaAdapter(
            model_name="llama3",
            temperature=0.7,
            top_p=0.9,
            api_base="http://example.com:11434"
        )

        params = adapter._identifying_params

        assert params["model_name"] == "llama3"
        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.9
        assert params["api_base"] == "http://example.com:11434"


class TestOllamaIntegration:
    """Integration tests that require an actual Ollama server."""

    @pytest.mark.integration
    #@pytest.mark.skipif(True, reason="Requires running Ollama server")
    def test_real_ollama_invoke(self):
        """Test with a real Ollama server - skip by default."""
        adapter = OllamaAdapter(model_name="qwen2.5", logprobs=True)#, temperature=0.0)

        messages = [
            HumanMessage(content="What is 2+2?")
        ]

        # This will actually call the Ollama API if a server is running
        result = adapter.invoke(messages)

        print (f"result.content: {result.content}")

        assert isinstance(result, AIMessage)
        assert len(result.content) > 0
        # The answer should contain 4 somewhere
        assert "4" in result.content