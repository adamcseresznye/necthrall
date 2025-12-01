"""Unit tests for the SynthesisAgent.

Tests cover:
- Happy path: Valid nodes return synthesized answer with citations
- Empty nodes: Returns fallback message gracefully
- LLM errors: Exception handling and logging
- Context formatting: Proper passage indexing
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from llama_index.core.schema import NodeWithScore, TextNode

from agents.synthesis_agent import SynthesisAgent, CITATION_QA_TEMPLATE


# Helper to create mock NodeWithScore objects
def create_mock_node(text: str, score: float = 0.9) -> NodeWithScore:
    """Create a mock NodeWithScore for testing."""
    node = TextNode(text=text)
    return NodeWithScore(node=node, score=score)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesize_happy_path():
    """Synthesize should return LLM response with citations for valid nodes."""
    agent = SynthesisAgent()

    nodes = [
        create_mock_node("Intermittent fasting can affect heart rate variability."),
        create_mock_node("Studies show metabolic benefits of time-restricted eating."),
        create_mock_node("Some patients report fatigue during fasting periods."),
    ]

    expected_response = (
        "Intermittent fasting can affect heart rate variability [1]. "
        "Studies have also shown metabolic benefits [2], though some patients "
        "report fatigue during fasting periods [3]."
    )

    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = expected_response

        result = await agent.synthesize(
            query="What are the cardiovascular effects of intermittent fasting?",
            nodes=nodes,
        )

        assert result == expected_response
        mock_generate.assert_called_once()
        # Verify the prompt contains all passages
        call_args = mock_generate.call_args
        prompt = call_args[0][0]  # First positional arg is the prompt
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt
        assert "heart rate variability" in prompt


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesize_empty_nodes():
    """Synthesize should return fallback message for empty node list."""
    agent = SynthesisAgent()

    result = await agent.synthesize(
        query="What are the risks of fasting?",
        nodes=[],
    )

    assert result == SynthesisAgent.INSUFFICIENT_CONTEXT_MESSAGE


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesize_llm_error():
    """Synthesize should re-raise LLM errors after logging."""
    agent = SynthesisAgent()

    nodes = [
        create_mock_node("Some passage about fasting."),
    ]

    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.side_effect = Exception("LLM API timeout")

        with pytest.raises(Exception) as exc_info:
            await agent.synthesize(
                query="What is fasting?",
                nodes=nodes,
            )

        assert "LLM API timeout" in str(exc_info.value)
        mock_generate.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesize_single_node():
    """Synthesize should work correctly with a single node."""
    agent = SynthesisAgent()

    nodes = [
        create_mock_node("Fasting promotes autophagy in cells."),
    ]

    expected_response = "Fasting promotes autophagy in cells [1]."

    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = expected_response

        result = await agent.synthesize(
            query="What is autophagy?",
            nodes=nodes,
        )

        assert result == expected_response


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesize_calls_synthesis_model_type():
    """Synthesize should call LLM router with 'synthesis' model type."""
    agent = SynthesisAgent()

    nodes = [create_mock_node("Test passage.")]

    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = "Answer [1]."

        await agent.synthesize(query="Test query", nodes=nodes)

        # Verify generate was called with correct arguments
        mock_generate.assert_called_once()
        args, kwargs = mock_generate.call_args
        # Check if model_type is passed as positional or keyword arg
        if len(args) >= 2:
            assert args[1] == "synthesis"
        else:
            assert kwargs.get("model_type") == "synthesis"


@pytest.mark.unit
def test_format_context_multiple_passages():
    """_format_context should format passages with 1-based indices."""
    agent = SynthesisAgent()

    nodes = [
        create_mock_node("First passage."),
        create_mock_node("Second passage."),
        create_mock_node("Third passage."),
    ]

    context = agent._format_context(nodes)

    assert "[1] First passage." in context
    assert "[2] Second passage." in context
    assert "[3] Third passage." in context
    assert "---" in context  # Separator is present


@pytest.mark.unit
def test_format_context_empty_list():
    """_format_context should return empty string for empty list."""
    agent = SynthesisAgent()

    context = agent._format_context([])

    assert context == ""


@pytest.mark.unit
def test_extract_text_from_node():
    """_extract_text should get text from NodeWithScore."""
    agent = SynthesisAgent()

    node = create_mock_node("Extracted text content.")

    text = agent._extract_text(node)

    assert text == "Extracted text content."


@pytest.mark.unit
def test_citation_prompt_template_format():
    """CITATION_QA_TEMPLATE should be properly formatted with placeholders."""
    assert "{context_str}" in CITATION_QA_TEMPLATE
    assert "{query_str}" in CITATION_QA_TEMPLATE
    assert "ONLY" in CITATION_QA_TEMPLATE  # Emphasis on using only context
    assert "[N]" in CITATION_QA_TEMPLATE  # Citation format instruction


@pytest.mark.unit
def test_agent_initialization():
    """SynthesisAgent should initialize with default temperature."""
    agent = SynthesisAgent()

    assert agent.temperature == 0.3
    assert agent.router is not None


@pytest.mark.unit
def test_agent_custom_temperature():
    """SynthesisAgent should accept custom temperature."""
    agent = SynthesisAgent(temperature=0.7)

    assert agent.temperature == 0.7


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesize_strips_whitespace():
    """Synthesize should strip leading/trailing whitespace from response."""
    agent = SynthesisAgent()

    nodes = [create_mock_node("Test passage.")]

    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = "  Answer with whitespace [1].  \n"

        result = await agent.synthesize(query="Test query", nodes=nodes)

        assert result == "Answer with whitespace [1]."


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesize_preserves_multiline_response():
    """Synthesize should preserve internal newlines in response."""
    agent = SynthesisAgent()

    nodes = [create_mock_node("Test passage.")]

    multiline_response = "Point 1 [1].\n\nPoint 2 [1]."

    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = multiline_response

        result = await agent.synthesize(query="Test query", nodes=nodes)

        assert result == multiline_response
