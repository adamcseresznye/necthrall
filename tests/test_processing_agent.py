import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from agents.processing_agent import ProcessingAgent
from models.state import State, Paper, PDFContent
import numpy as np


@pytest.fixture
def mock_app():
    """Mock FastAPI app with cached models"""
    app = Mock()

    # Mock embedding model
    mock_model = Mock()
    mock_model.encode = Mock(
        return_value=np.array([[0.1] * 384])
    )  # Mock query embedding

    # Mock app state
    app.state.embedding_model = mock_model
    return app


@pytest.fixture
def agent(mock_app):
    return ProcessingAgent(mock_app)


@pytest.fixture
def sample_filtered_papers():
    """Create sample filtered papers"""
    return [
        Paper(
            paper_id="paper1",
            title="Test Paper 1",
            authors=["Author A"],
            year=2023,
            journal="Nature",
            abstract="Abstract 1",
            pdf_url="https://example.com/1.pdf",
            type="article",
        ),
        Paper(
            paper_id="paper2",
            title="Test Paper 2",
            authors=["Author B"],
            year=2023,
            journal="Science",
            abstract="Abstract 2",
            pdf_url="https://example.com/2.pdf",
            type="article",
        ),
    ]


@pytest.fixture
def sample_pdf_contents():
    """Create sample PDF contents"""
    return [
        PDFContent(
            paper_id="paper1",
            raw_text="""
1. Introduction

This paper discusses test topics in detail. The introduction covers background material.

2. Methods

We used various methods for testing.

3. Results

The results show clear findings.

4. Discussion

Discussion of implications.
""",
            page_count=5,
            char_count=1000,
            extraction_time=1.0,
        ),
        PDFContent(
            paper_id="paper2",
            raw_text="""
Single section paper without clear divisions.

This paper has content but no numbered sections.
More content here.
""",
            page_count=3,
            char_count=600,
            extraction_time=0.8,
        ),
    ]


def test_processing_agent_callable(agent, sample_filtered_papers, sample_pdf_contents):
    """Test ProcessingAgent is callable and returns State with top_passages and processing_stats"""
    state = State(
        original_query="test query",
        optimized_query="optimized test query",
        filtered_papers=sample_filtered_papers,
        pdf_contents=sample_pdf_contents,
    )

    # Mock all the internal components
    with patch("asyncio.run") as mock_asyncio_run, patch(
        "utils.embedding_manager.isinstance", return_value=True
    ), patch.object(
        agent.section_detector, "detect_sections"
    ) as mock_detect, patch.object(
        agent.embedding_manager, "process_chunks_async"
    ) as mock_embed, patch.object(
        agent.hybrid_retriever, "build_indices"
    ) as mock_build, patch.object(
        agent.hybrid_retriever, "retrieve"
    ) as mock_retrieve, patch.object(
        agent.reranker, "rerank"
    ) as mock_rerank:

        # Setup asyncio.run to return the mock embedding result
        mock_asyncio_run.return_value = [
            {
                "content": "chunk1",
                "section": "introduction",
                "paper_id": "paper1",
                "embedding": [0.1] * 384,
                "embedding_dim": 384,
            },
            {
                "content": "chunk2",
                "section": "methods",
                "paper_id": "paper1",
                "embedding": [0.2] * 384,
                "embedding_dim": 384,
            },
            {
                "content": "chunk3",
                "section": "unknown",
                "paper_id": "paper2",
                "embedding": [0.3] * 384,
                "embedding_dim": 384,
            },
        ]

        # Mock section detection - first paper has multiple sections, second has fallback
        mock_detect.side_effect = [
            [
                {"content": "Intro content", "section": "introduction", "start_pos": 0},
                {"content": "Methods content", "section": "methods", "start_pos": 100},
            ],
            [
                {"content": "Fallback content", "section": "unknown", "start_pos": 0}
            ],  # Fallback
        ]

        # Mock embedding - return chunks with embeddings
        mock_embed.return_value = [
            {
                "content": "chunk1",
                "section": "introduction",
                "paper_id": "paper1",
                "embedding": [0.1] * 384,
                "embedding_dim": 384,
            },
            {
                "content": "chunk2",
                "section": "methods",
                "paper_id": "paper1",
                "embedding": [0.2] * 384,
                "embedding_dim": 384,
            },
            {
                "content": "chunk3",
                "section": "unknown",
                "paper_id": "paper2",
                "embedding": [0.3] * 384,
                "embedding_dim": 384,
            },
        ]

        mock_build.return_value = 0.1
        mock_retrieve.return_value = [
            {
                "content": "chunk1",
                "section": "introduction",
                "paper_id": "paper1",
                "retrieval_score": 0.8,
            },
            {
                "content": "chunk2",
                "section": "methods",
                "paper_id": "paper1",
                "retrieval_score": 0.7,
            },
        ]
        mock_rerank.return_value = [
            {
                "content": "chunk1",
                "section": "introduction",
                "paper_id": "paper1",
                "retrieval_score": 0.8,
                "cross_encoder_score": 0.9,
                "final_score": 0.9,
            },
            {
                "content": "chunk2",
                "section": "methods",
                "paper_id": "paper1",
                "retrieval_score": 0.7,
                "cross_encoder_score": 0.85,
                "final_score": 0.85,
            },
        ]

        # Call the agent
        result_state = agent(state)

        # Verify result is a new State instance (side-effect free)
        assert isinstance(result_state, State)
        assert result_state is not state  # Different object

        # Verify top_passages
        assert len(result_state.top_passages) == 2

        # Convert Passage objects to dicts for compatibility and testing
        passages_dict = [
            p.model_dump() if hasattr(p, "model_dump") else p
            for p in result_state.top_passages
        ]

        # Verify content fields in returned passages
        assert passages_dict[0]["content"] == "chunk1"
        assert passages_dict[0]["section"] == "introduction"
        assert passages_dict[0]["paper_id"] == "paper1"
        assert "retrieval_score" in passages_dict[0]
        assert "cross_encoder_score" in passages_dict[0]
        assert "final_score" in passages_dict[0]

        # Verify processing_stats
        stats = result_state.processing_stats
        assert isinstance(stats, dict)
        assert stats["total_papers"] == 2
        assert stats["processed_papers"] == 2
        assert stats["skipped_papers"] == 0
        assert "total_time" in stats
        assert stats["total_time"] > 0
        assert "stage_times" in stats
        assert "section_detection" in stats["stage_times"]
        assert "chunking" in stats["stage_times"]
        assert "embedding" in stats["stage_times"]
        assert "index_build" in stats["stage_times"]
        assert "retrieval" in stats["stage_times"]
        assert "reranking" in stats["stage_times"]


def test_processing_agent_handles_empty_filtered_papers(agent):
    """Test ProcessingAgent handles empty filtered_papers gracefully"""
    state = State(
        original_query="test query",
        filtered_papers=[],
    )

    result_state = agent(state)

    assert len(result_state.top_passages) == 0
    assert result_state.processing_stats["total_papers"] == 0
    assert result_state.processing_stats["error"] == "No filtered papers"
    assert result_state.processing_stats["reason"] == "Empty filtered_papers list"


def test_processing_agent_handles_missing_pdf_content(agent, sample_filtered_papers):
    """Test ProcessingAgent skips papers without PDF content"""
    # State with papers but no matching PDF contents
    state = State(
        original_query="test query",
        filtered_papers=sample_filtered_papers,
        pdf_contents=[],  # No PDF contents
    )

    result_state = agent(state)

    # Should skip both papers and return empty results with error
    assert len(result_state.top_passages) == 0
    assert result_state.processing_stats["total_papers"] == 2
    assert result_state.processing_stats["processed_papers"] == 0
    assert result_state.processing_stats["skipped_papers"] == 2
    assert "error" in result_state.processing_stats


def test_processing_agent_chunking_fallback(
    agent, sample_filtered_papers, sample_pdf_contents
):
    """Test ProcessingAgent uses fallback chunking when sections < 2"""
    state = State(
        original_query="test query",
        filtered_papers=sample_filtered_papers,
        pdf_contents=sample_pdf_contents,
    )

    with patch("asyncio.run") as mock_asyncio_run, patch.object(
        agent.section_detector, "detect_sections"
    ) as mock_detect, patch.object(
        agent.embedding_manager, "process_chunks_async"
    ) as mock_embed, patch.object(
        agent.hybrid_retriever, "build_indices"
    ) as mock_build, patch.object(
        agent.hybrid_retriever, "retrieve"
    ) as mock_retrieve, patch.object(
        agent.reranker, "rerank"
    ) as mock_rerank:

        # Setup asyncio.run to return the mock embedding result
        mock_asyncio_run.return_value = [
            {
                "content": "fallback chunk",
                "section": "unknown",
                "paper_id": "paper1",
                "embedding": [0.1] * 384,
                "embedding_dim": 384,
            },
        ]

        # Mock detection: one paper with single section (triggers fallback)
        mock_detect.return_value = [
            {"content": "Single section content", "section": "unknown", "start_pos": 0}
        ]

        mock_embed.return_value = [
            {
                "content": "fallback chunk",
                "section": "unknown",
                "paper_id": "paper1",
                "embedding": [0.1] * 384,
                "embedding_dim": 384,
            },
        ]

        mock_build.return_value = 0.1
        mock_retrieve.return_value = [
            {
                "content": "fallback chunk",
                "section": "unknown",
                "paper_id": "paper1",
                "retrieval_score": 0.8,
            }
        ]
        mock_rerank.return_value = [
            {
                "content": "fallback chunk",
                "section": "unknown",
                "paper_id": "paper1",
                "retrieval_score": 0.8,
                "cross_encoder_score": 0.9,
                "final_score": 0.9,
            }
        ]

        result_state = agent(state)

        assert len(result_state.top_passages) == 1
        assert (
            result_state.processing_stats["fallback_used_count"] == 2
        )  # Both papers fallback
        # Convert Passage object to dict for compatibility
        passages_dict = [
            p.model_dump() if hasattr(p, "model_dump") else p
            for p in result_state.top_passages
        ]
        # Passage model normalizes "unknown" sections to "other"
        assert passages_dict[0]["section"] == "other"


def test_processing_agent_stats_structure(
    agent, sample_filtered_papers, sample_pdf_contents
):
    """Test processing_stats contains all required fields and counts"""
    state = State(
        original_query="test query",
        filtered_papers=sample_filtered_papers,
        pdf_contents=sample_pdf_contents,
    )

    with patch("asyncio.run") as mock_asyncio_run, patch.object(
        agent.section_detector, "detect_sections"
    ) as mock_detect, patch.object(
        agent.embedding_manager, "process_chunks_async"
    ) as mock_embed, patch.object(
        agent.hybrid_retriever, "build_indices"
    ) as mock_build, patch.object(
        agent.hybrid_retriever, "retrieve"
    ) as mock_retrieve, patch.object(
        agent.reranker, "rerank"
    ) as mock_rerank:

        # Setup asyncio.run to return the mock embedding result
        mock_asyncio_run.return_value = [
            {
                "content": "chunk",
                "section": "introduction",
                "paper_id": "paper1",
                "embedding": [0.1] * 384,
                "embedding_dim": 384,
            },
        ]

        mock_detect.return_value = [
            {"content": "content", "section": "introduction", "start_pos": 0}
        ]
        mock_embed.return_value = [
            {
                "content": "chunk",
                "section": "introduction",
                "paper_id": "paper1",
                "embedding": [0.1] * 384,
                "embedding_dim": 384,
            },
        ]
        mock_build.return_value = 0.1
        mock_retrieve.return_value = [
            {
                "content": "chunk",
                "section": "introduction",
                "paper_id": "paper1",
                "retrieval_score": 0.8,
            }
        ]
        mock_rerank.return_value = [
            {
                "content": "chunk",
                "section": "introduction",
                "paper_id": "paper1",
                "retrieval_score": 0.8,
                "cross_encoder_score": 0.9,
                "final_score": 0.9,
            }
        ]

        result_state = agent(state)

        stats = result_state.processing_stats

        # Required count fields
        assert "total_papers" in stats
        assert "processed_papers" in stats
        assert "skipped_papers" in stats
        assert "total_sections" in stats
        assert "total_chunks" in stats
        assert "fallback_used_count" in stats
        assert "chunks_embedded" in stats
        assert "retrieval_candidates" in stats
        assert "reranked_passages" in stats

        # Time tracking
        assert "total_time" in stats
        assert "stage_times" in stats
        assert all(
            key in stats["stage_times"]
            for key in [
                "section_detection",
                "chunking",
                "embedding",
                "index_build",
                "retrieval",
                "reranking",
            ]
        )
