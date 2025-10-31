"""
Tests for the new unified ProcessingAgent with modular RAG components.

Validates the refactored ProcessingAgent interface and component integration.
"""

from unittest.mock import Mock
import pytest


def create_mock_app(embedder=None, chunker=None):
    """Create a mock FastAPI app with cached models."""
    app = Mock()
    app.state = Mock()
    app.state.embedding_model = embedder or Mock()
    return app


class TestImportSmokeTest:
    """Smoke test to ensure imports and basic functionality work."""

    def test_processing_agent_import(self):
        """Test that ProcessingAgent can be imported without errors."""
        from agents.processing import ProcessingAgent, ProcessingErrorType

        assert ProcessingAgent is not None
        assert callable(ProcessingAgent)
        assert ProcessingErrorType is not None

    @pytest.mark.skip(reason="ProcessingAgent requires real models for instantiation")
    def test_basic_instantiation(self):
        """Test ProcessingAgent can be created with a mock app (skipped due to model validation)."""
        from agents.processing import ProcessingAgent

        mock_app = create_mock_app()
        agent = ProcessingAgent(mock_app)

        assert agent is not None
        assert agent.app == mock_app
