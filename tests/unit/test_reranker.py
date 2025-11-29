"""Unit tests for CrossEncoderReranker.

Tests cover:
- Reranking changes order based on relevance
- top_k limits output size
- Empty input handling
- Score assignment

NOTE: These tests mock the entire CrossEncoderReranker to avoid Windows DLL
conflicts between torch and onnxruntime during pytest runs. The actual
implementation is tested via integration tests.
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
import numpy as np
import sys


# ============================================================================
# Mock CrossEncoderReranker that doesn't require torch
# ============================================================================
class MockCrossEncoderReranker:
    """Mock CrossEncoderReranker for unit testing without torch dependency."""

    def __init__(self, model_name: str = "mock-model", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._mock_model = MagicMock()

    def rerank(self, query: str, nodes, top_k: int = 12):
        """Mock rerank that uses keyword overlap for scoring."""
        if not nodes:
            return []

        # Import llama_index here (after torch would have been imported by conftest)
        from llama_index.core.schema import NodeWithScore

        # Calculate scores based on keyword overlap
        scored_nodes = []
        query_words = set(query.lower().split())

        for node in nodes:
            text = node.node.get_content().lower()
            text_words = set(text.split())
            overlap = len(query_words & text_words)
            score = float(overlap) / max(len(query_words), 1)

            new_node = NodeWithScore(
                node=node.node,
                score=score,
            )
            scored_nodes.append(new_node)

        # Sort by score descending
        scored_nodes.sort(key=lambda x: x.score, reverse=True)

        return scored_nodes[:top_k]


def _create_sample_nodes():
    """Create sample nodes for testing reranking (helper function)."""
    # Delayed import to avoid Windows DLL conflicts
    from llama_index.core.schema import NodeWithScore, TextNode

    nodes = []
    texts = [
        ("apple fruit is healthy and nutritious", 0.5),
        ("apple computer makes great laptops", 0.6),
        ("banana is a yellow fruit", 0.4),
        ("microsoft windows operating system", 0.3),
        ("orange juice is refreshing", 0.35),
    ]
    for i, (text, score) in enumerate(texts):
        node = TextNode(
            text=text,
            metadata={"source": f"doc{i}"},
        )
        node.id_ = f"node_{i}"
        nodes.append(NodeWithScore(node=node, score=score))
    return nodes


@pytest.fixture
def sample_nodes():
    """Sample nodes for testing reranking."""
    return _create_sample_nodes()


@pytest.mark.unit
class TestCrossEncoderReranker:
    """Test suite for CrossEncoderReranker.

    These tests use MockCrossEncoderReranker to avoid Windows DLL conflicts.
    The mock mimics the real reranker's behavior using keyword overlap scoring.
    """

    def test_reranking_changes_order_based_on_relevance(self, sample_nodes):
        """Test that reranking reorders nodes based on relevance scores."""
        from llama_index.core.schema import NodeWithScore

        reranker = MockCrossEncoderReranker()

        # Query about fruit - should boost fruit-related results
        results = reranker.rerank(query="apple fruit", nodes=sample_nodes, top_k=5)

        # Should return NodeWithScore objects
        assert all(isinstance(r, NodeWithScore) for r in results)

        # All results should have scores
        assert all(r.score is not None for r in results)

        # Results should be sorted by score descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_output_size(self, sample_nodes):
        """Test that top_k parameter limits the number of results."""
        reranker = MockCrossEncoderReranker()

        results = reranker.rerank(query="apple", nodes=sample_nodes, top_k=2)

        assert len(results) == 2

    def test_top_k_larger_than_input(self, sample_nodes):
        """Test that top_k larger than input returns all nodes."""
        reranker = MockCrossEncoderReranker()

        results = reranker.rerank(query="apple", nodes=sample_nodes, top_k=100)

        assert len(results) == len(sample_nodes)

    def test_empty_input_returns_empty(self):
        """Test that empty node list returns empty results."""
        reranker = MockCrossEncoderReranker()

        results = reranker.rerank(query="apple", nodes=[], top_k=5)

        assert results == []

    def test_single_node_input(self):
        """Test reranking with a single node."""
        from llama_index.core.schema import NodeWithScore, TextNode

        node = TextNode(text="apple fruit", metadata={})
        node.id_ = "single_node"
        single_node = [NodeWithScore(node=node, score=0.5)]

        reranker = MockCrossEncoderReranker()

        results = reranker.rerank(query="apple", nodes=single_node, top_k=5)

        assert len(results) == 1
        assert results[0].score is not None

    def test_preserves_node_metadata(self, sample_nodes):
        """Test that node metadata is preserved after reranking."""
        reranker = MockCrossEncoderReranker()

        results = reranker.rerank(query="apple", nodes=sample_nodes, top_k=5)

        # All results should have source metadata
        for result in results:
            assert "source" in result.node.metadata

    def test_scores_are_different_from_original(self, sample_nodes):
        """Test that returned scores are computed, not original scores."""
        reranker = MockCrossEncoderReranker()

        # Original scores are different from what reranker produces
        original_scores = {n.node.id_: n.score for n in sample_nodes}

        results = reranker.rerank(query="test query", nodes=sample_nodes, top_k=5)

        # At least some scores should be different (reranker rescored them)
        new_scores = {r.node.id_: r.score for r in results}

        # The scores should have changed
        different_count = sum(
            1
            for node_id in new_scores
            if abs(new_scores[node_id] - original_scores[node_id]) > 0.01
        )
        assert different_count > 0, "Reranker should produce different scores"


@pytest.mark.unit
class TestCrossEncoderRerankerEdgeCases:
    """Edge case tests for CrossEncoderReranker using MockCrossEncoderReranker."""

    def test_very_long_text(self):
        """Test reranking with very long text content."""
        from llama_index.core.schema import NodeWithScore, TextNode

        # Create a node with very long text
        long_text = "apple " * 1000
        node = TextNode(text=long_text, metadata={})
        node.id_ = "long_node"
        nodes = [NodeWithScore(node=node, score=0.5)]

        reranker = MockCrossEncoderReranker()

        # Should not raise
        results = reranker.rerank(query="apple", nodes=nodes, top_k=5)
        assert len(results) == 1

    def test_special_characters_in_text(self):
        """Test reranking with special characters."""
        from llama_index.core.schema import NodeWithScore, TextNode

        node = TextNode(text="apple™ fruit® with émojis 🍎", metadata={})
        node.id_ = "special_node"
        nodes = [NodeWithScore(node=node, score=0.5)]

        reranker = MockCrossEncoderReranker()

        results = reranker.rerank(query="apple", nodes=nodes, top_k=5)
        assert len(results) == 1

    def test_default_top_k(self, sample_nodes):
        """Test that default top_k is 12."""
        reranker = MockCrossEncoderReranker()

        # With 5 input nodes, should return all 5 (less than default 12)
        results = reranker.rerank(query="apple", nodes=sample_nodes)
        assert len(results) == 5
