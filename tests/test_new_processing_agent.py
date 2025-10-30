"""
Tests for the new unified ProcessingAgent with modular RAG components.

Validates the refactored ProcessingAgent interface and component integration.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
import numpy as np
from agents.processing import ProcessingAgent, ProcessingErrorType
from models.state import State, ProcessingConfig, Paper, PDFContent
from rag.chunking import AdvancedDocumentChunker
from rag.embeddings import EmbeddingGenerator


class TestNewProcessingAgent:
    """Test the new ProcessingAgent with modular components."""

    @pytest.fixture
    def mock_embedder(self):
        """Mock EmbeddingGenerator for testing."""
        embedder = Mock(spec=EmbeddingGenerator)
        embedder.batch_size = 32
        embedder.generate_embeddings_async = AsyncMock()
        embedder.get_statistics = Mock(return_value={"total_processed": 10})
        return embedder

    @pytest.fixture
    def mock_chunker(self):
        """Mock AdvancedDocumentChunker for testing."""
        chunker = Mock(spec=AdvancedDocumentChunker)
        chunker.process_papers = Mock(
            return_value=([], [], {})
        )  # No chunks, no errors, no stats
        chunker.get_statistics = Mock(return_value={"total_chunks": 0})
        return chunker

    @pytest.fixture
    def sample_state(self):
        """Create sample state for testing."""
        state = State(
            original_query="test query",
            filtered_papers=[
                Paper(
                    paper_id="paper1",
                    title="Test Paper",
                    authors=["Author"],
                    year=2023,
                    journal="Test Journal",  # Required field
                    type="article",
                    pdf_url="https://example.com/paper1.pdf",  # Required field
                )
            ],
            pdf_contents=[
                PDFContent(
                    paper_id="paper1",
                    raw_text="Some content",
                    page_count=5,
                    char_count=1000,
                    extraction_time=1.0,
                )
            ],
            config={"batch_size": 16, "top_k": 10},
        )
        return state

    def test_processing_agent_initialization(self, mock_embedder):
        """Test ProcessingAgent initializes with mock components."""
        agent = ProcessingAgent(
            app=None,
            chunker=None,
            embedder=mock_embedder,
            retriever=None,
            reranker=None,
        )

        assert agent.embedder == mock_embedder
        assert isinstance(agent.chunker, AdvancedDocumentChunker)  # Default chunker
        assert agent.app is None

    def test_processing_agent_requires_embedder(self):
        """Test ProcessingAgent requires embedder when no app provided."""
        with pytest.raises(ValueError, match="Either app or embedder must be provided"):
            ProcessingAgent(app=None)

    def test_parse_config_with_valid_dict(self):
        """Test config parsing with valid dictionary."""
        agent = ProcessingAgent(embedder=Mock())
        config_dict = {"batch_size": 8, "top_k": 5}

        config = agent._parse_config(config_dict)

        assert config.batch_size == 8
        assert config.top_k == 5
        assert config.final_k == 10  # Default value

    def test_parse_config_with_invalid_dict(self):
        """Test config parsing falls back to defaults for invalid input."""
        agent = ProcessingAgent(embedder=Mock())
        config_dict = {"invalid_field": "value"}

        config = agent._parse_config(config_dict)

        # Should use defaults
        assert config.batch_size == 32
        assert config.top_k == 20
        assert config.final_k == 10

    def test_handle_empty_results(self):
        """Test empty results handling updates metadata correctly."""
        from models.state import ProcessingMetadata

        agent = ProcessingAgent(embedder=Mock())
        metadata = ProcessingMetadata()
        start_time = 1000.0

        agent._handle_empty_results(Mock(), metadata, start_time, "Test reason")

        assert len(metadata.processing_errors) == 1
        assert metadata.processing_errors[0] == "Test reason"
        assert metadata.total_time > 0

    def test_processing_agent_callable_with_empty_input(
        self, mock_embedder, mock_chunker, sample_state
    ):
        """Test ProcessingAgent handles empty input gracefully."""
        # Setup mocks for empty results
        mock_embedder.generate_embeddings_async.return_value = []

        agent = ProcessingAgent(
            chunker=mock_chunker,
            embedder=mock_embedder,
            retriever=Mock(),  # Will cause failure in later stages
            reranker=Mock(),
        )

        result_state = agent(sample_state)

        # Should return state with empty results due to chunking failure
        assert result_state.chunks == []
        assert result_state.relevant_passages == []
        assert result_state.processing_metadata.total_papers == 1
        assert "No chunks created" in str(
            result_state.processing_metadata.processing_errors
        )

    @pytest.mark.asyncio
    async def test_process_embedding_stage_success(self, mock_embedder):
        """Test embedding stage processes chunks successfully."""
        from models.state import ProcessingMetadata, Chunk

        agent = ProcessingAgent(embedder=mock_embedder)

        # Mock successful embedding - handle dictionaries returned by chunk.model_dump()
        async def mock_generate_embeddings(chunk_dicts):
            return [
                {
                    "content": chunk_dict["content"],
                    "embedding": np.array([0.1] * 384),
                    "embedding_dim": 384,
                }
                for chunk_dict in chunk_dicts
            ]

        mock_embedder.generate_embeddings_async.side_effect = mock_generate_embeddings

        # Create test chunks
        chunks = [
            Chunk(
                content="test content",
                section="introduction",
                paper_id="paper1",
                paper_title="Test Paper",
                start_position=0,
                end_position=13,
                use_fallback=False,
            )
        ]

        config = ProcessingConfig()
        metadata = ProcessingMetadata()  # Use real ProcessingMetadata object

        embedded, errors = await agent._process_embedding_stage(
            chunks, metadata, config
        )

        # Verify embedder was called
        mock_embedder.generate_embeddings_async.assert_called_once()
        assert len(embedded) == 1
        assert embedded[0]["content"] == "test content"
        assert "embedding" in embedded[0]
        assert errors == []  # No errors

    def test_performance_summary(self, mock_embedder, mock_chunker):
        """Test performance summary retrieval."""
        mock_chunker.get_statistics.return_value = {"chunker_metric": 42}

        agent = ProcessingAgent(
            chunker=mock_chunker,
            embedder=mock_embedder,
        )

        summary = agent.get_performance_summary()

        assert "chunker_stats" in summary
        assert "embedder_stats" in summary
        assert "retriever_stats" in summary
        assert "reranker_stats" in summary
        assert summary["chunker_stats"]["chunker_metric"] == 42

    def test_processing_status_enum_import(self):
        """Test that required imports are available."""
        # This test will fail if there are import issues
        from agents.processing import ProcessingAgent, ProcessingAgentV2
        from models.state import ProcessingConfig, ProcessingMetadata, Chunk
        from rag.chunking import AdvancedDocumentChunker
        from rag.embeddings import EmbeddingGenerator

        # Verify classes can be instantiated (basic smoke test)
        assert ProcessingAgentV2 is ProcessingAgent  # Alias check
        assert callable(ProcessingAgent)
        assert ProcessingConfig().batch_size == 32  # Default value check

    def test_memory_usage_checking(self, mock_embedder):
        """Test memory usage monitoring functionality."""
        agent = ProcessingAgent(embedder=mock_embedder)

        # Mock psutil to return 50% memory usage
        import psutil

        with patch.object(psutil, "virtual_memory") as mock_memory:
            mock_memory.return_value.percent = 50.0

            usage = agent._check_memory_usage()
            assert usage == 0.5

            mock_memory.assert_called_once()

    def test_memory_usage_checking_failure(self, mock_embedder):
        """Test memory usage checking handles failures gracefully."""
        agent = ProcessingAgent(embedder=mock_embedder)

        # Mock psutil to raise exception
        import psutil

        with patch.object(psutil, "virtual_memory") as mock_memory:
            mock_memory.side_effect = Exception("Memory check failed")

            usage = agent._check_memory_usage()
            assert usage == 0.0  # Should return safe default

    def test_chunk_content_validation(self, mock_embedder):
        """Test chunk content validation detects various corruption issues."""
        agent = ProcessingAgent(embedder=mock_embedder)

        # Valid content
        valid_chunk = {"content": "This is valid content for testing purposes."}
        is_valid, error = agent._validate_chunk_content(valid_chunk)
        assert is_valid
        assert error == ""

        # Empty content
        empty_chunk = {"content": ""}
        is_valid, error = agent._validate_chunk_content(empty_chunk)
        assert not is_valid
        assert "Empty" in error

        # Whitespace only
        ws_chunk = {"content": "   \n\t   "}
        is_valid, error = agent._validate_chunk_content(ws_chunk)
        assert not is_valid
        assert "whitespace-only" in error

        # Null bytes
        null_chunk = {"content": "Content\x00with\x00nulls"}
        is_valid, error = agent._validate_chunk_content(null_chunk)
        assert not is_valid
        assert "null bytes" in error

        # Excessive line breaks
        line_chunk = {"content": "Line\n" * 600}
        is_valid, error = agent._validate_chunk_content(line_chunk)
        assert not is_valid
        assert "line breaks" in error

        # Content too long
        long_chunk = {"content": "x" * 12000}
        is_valid, error = agent._validate_chunk_content(long_chunk)
        assert not is_valid
        assert "too long" in error

    def test_optimal_batch_size_calculation(self, mock_embedder):
        """Test adaptive batch size calculation based on memory."""
        agent = ProcessingAgent(embedder=mock_embedder)

        # Mock memory usage
        with patch.object(agent, "_check_memory_usage") as mock_mem:
            # Normal memory - should return configured batch size
            mock_mem.return_value = 0.3  # 30%
            optimal = agent._calculate_optimal_batch_size(100, 32)
            assert optimal == 32

            # High memory - should reduce batch size
            mock_mem.return_value = 0.7  # 70%
            optimal = agent._calculate_optimal_batch_size(100, 32)
            assert optimal == 16  # Half

            # Very high memory - should significantly reduce
            mock_mem.return_value = 0.9  # 90%
            optimal = agent._calculate_optimal_batch_size(100, 32)
            assert optimal == 8  # Quarter

            # Should not exceed total chunks
            optimal = agent._calculate_optimal_batch_size(5, 32)
            assert optimal == 5

    def test_embedding_error_classification(self, mock_embedder):
        """Test embedding error classification for different error types."""
        agent = ProcessingAgent(embedder=mock_embedder)

        # OOM errors
        oom_error = Exception("CUDA out of memory")
        assert (
            agent._classify_embedding_error(oom_error)
            == ProcessingErrorType.OUT_OF_MEMORY
        )

        memory_error = Exception("MemoryError: out of memory")
        assert (
            agent._classify_embedding_error(memory_error)
            == ProcessingErrorType.OUT_OF_MEMORY
        )

        # Model loading errors
        model_error = Exception("Model not found in cache")
        assert (
            agent._classify_embedding_error(model_error)
            == ProcessingErrorType.MODEL_LOADING
        )

        # Network errors
        network_error = Exception("Connection timeout 502")
        assert (
            agent._classify_embedding_error(network_error)
            == ProcessingErrorType.NETWORK_FAILURE
        )

        # Unknown errors
        unknown_error = Exception("Some random error")
        assert (
            agent._classify_embedding_error(unknown_error)
            == ProcessingErrorType.UNKNOWN
        )

    def test_embedding_validation(self, mock_embedder):
        """Test embedding validation and filtering."""
        agent = ProcessingAgent(embedder=mock_embedder)

        # Test data
        import numpy as np

        valid_chunk = {
            "content": "test",
            "embedding": np.array([0.1, 0.2, 0.3] * 128),  # 384 dims
            "embedding_dim": 384,
        }

        invalid_chunk_missing = {
            "content": "test",
            # No embedding key
        }

        invalid_chunk_wrong_shape = {
            "content": "test",
            "embedding": np.array([0.1, 0.2]),  # Wrong shape
            "embedding_dim": 384,
        }

        invalid_chunk_non_finite = {
            "content": "test",
            "embedding": np.array([np.nan, 0.2, np.inf] * 128),  # Non-finite values
            "embedding_dim": 384,
        }

        test_chunks = [
            valid_chunk,
            invalid_chunk_missing,
            invalid_chunk_wrong_shape,
            invalid_chunk_non_finite,
        ]

        valid_results, invalid_count = agent._validate_embeddings(test_chunks)

        # Should have 1 valid chunk, 3 invalid
        assert len(valid_results) == 1
        assert invalid_count == 3
        assert valid_results[0]["content"] == "test"  # Valid chunk preserved
