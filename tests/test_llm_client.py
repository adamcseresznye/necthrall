import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from google.api_core.exceptions import ResourceExhausted

from utils.llm_client import LLMClient


pytestmark = [pytest.mark.integration]


@patch("utils.llm_client.ChatGoogleGenerativeAI")
@patch("utils.llm_client.ChatGroq")
def test_primary_llm_success(mock_groq, mock_gemini):
    """Test successful generation using the primary LLM."""
    mock_gemini_instance = mock_gemini.return_value
    mock_gemini_instance.invoke.return_value = AIMessage(
        content="Hello from Gemini",
    )
    mock_gemini_instance.get_num_tokens.return_value = 5

    client = LLMClient()
    messages = [{"role": "user", "content": "Hello"}]
    response = client.generate(messages)

    assert response["content"] == "Hello from Gemini"
    assert response["model_used"].startswith("gemini")
    assert response["tokens_used"] == 5
    mock_gemini_instance.invoke.assert_called_once()
    mock_groq.return_value.invoke.assert_not_called()


@patch("utils.llm_client.ChatGoogleGenerativeAI")
@patch("utils.llm_client.ChatGroq")
def test_fallback_llm_success(mock_groq, mock_gemini):
    """Test fallback to Groq when Gemini fails."""
    mock_gemini_instance = mock_gemini.return_value
    mock_gemini_instance.invoke.side_effect = Exception("Gemini API Error")

    mock_groq_instance = mock_groq.return_value
    mock_groq_instance.invoke.return_value = AIMessage(
        content="Hello from Groq",
    )
    mock_groq_instance.get_num_tokens.return_value = 10

    client = LLMClient()
    messages = [{"role": "user", "content": "Hello"}]
    response = client.generate(messages)

    assert response["content"] == "Hello from Groq"
    assert response["model_used"].startswith("groq")
    assert response["tokens_used"] == 10
    mock_gemini_instance.invoke.assert_called_once()
    mock_groq_instance.invoke.assert_called_once()


@patch("utils.llm_client.ChatGoogleGenerativeAI")
@patch("utils.llm_client.ChatGroq")
def test_both_llms_fail(mock_groq, mock_gemini):
    """Test that an exception is raised when both LLMs fail."""
    mock_gemini_instance = mock_gemini.return_value
    mock_gemini_instance.invoke.side_effect = Exception("Gemini API Error")

    mock_groq_instance = mock_groq.return_value
    mock_groq_instance.invoke.side_effect = Exception("Groq API Error")

    client = LLMClient()
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(Exception) as context:
        client.generate(messages)

    assert "Groq API Error" in str(context.value)


@patch("utils.llm_client.ChatGoogleGenerativeAI")
@patch("utils.llm_client.ChatGroq")
def test_rate_limit_error(mock_groq, mock_gemini):
    """Test that a rate limit error on the primary LLM triggers the fallback."""
    mock_gemini_instance = mock_gemini.return_value
    mock_gemini_instance.invoke.side_effect = ResourceExhausted("Rate limit exceeded")

    mock_groq_instance = mock_groq.return_value
    mock_groq_instance.invoke.return_value = AIMessage(content="Groq response")
    mock_groq_instance.get_num_tokens.return_value = 3

    client = LLMClient()
    messages = [{"role": "user", "content": "Hello"}]

    response = client.generate(messages)

    # Check that the fallback was called
    mock_groq_instance.invoke.assert_called_once()
    # Check that the response is from the fallback
    assert response["content"] == "Groq response"


@patch("utils.llm_client.ChatGoogleGenerativeAI")
@patch("utils.llm_client.ChatGroq")
def test_fallback_rate_limit_error(mock_groq, mock_gemini):
    """Test that a rate limit error is handled correctly on the fallback LLM."""
    mock_gemini_instance = mock_gemini.return_value
    mock_gemini_instance.invoke.side_effect = Exception("Gemini API Error")

    mock_groq_instance = mock_groq.return_value
    mock_groq_instance.invoke.side_effect = ResourceExhausted("Rate limit exceeded")

    client = LLMClient()
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ResourceExhausted):
        client.generate(messages)
