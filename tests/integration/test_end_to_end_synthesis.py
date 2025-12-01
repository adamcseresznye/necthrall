"""End-to-end integration tests for the full 10-stage pipeline.

These tests validate the complete pipeline from query to cited answer,
including synthesis (Stage 9) and verification (Stage 10).

Week 1 Stages (1-4):
    1. Query Optimization
    2. Semantic Scholar Search
    3. Quality Gate
    4. Composite Scoring

Week 2 Stages (5-8):
    5. PDF Acquisition
    6. Processing & Embedding
    7. Hybrid Retrieval
    8. Cross-Encoder Reranking

Final Stages (9-10):
    9. Synthesis - Generate cited answer from passages
    10. Verification - Validate citation references

Test Coverage:
    - Happy path: All 10 stages succeed with valid cited answer
    - Assertions for all key PipelineResult fields

Run these tests with:
    pytest tests/integration/test_end_to_end_synthesis.py -v -s
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

# Import llama_index types for fixtures
from llama_index.core.schema import NodeWithScore, TextNode

# Import query service
from services.query_service import QueryService, PipelineResult


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model with the expected interface.

    Returns a mock that mimics both LlamaIndexRetriever and ProcessingAgent
    embedding interfaces with deterministic 384-dim vectors.
    """
    model = MagicMock()
    model.embed_dim = 384

    def mock_embed_batch(texts: List[str]) -> List[List[float]]:
        return [[0.1] * 384 for _ in texts]

    model.get_text_embedding_batch = MagicMock(side_effect=mock_embed_batch)
    model.embed_documents = MagicMock(side_effect=mock_embed_batch)
    return model


@pytest.fixture
def mock_finalists() -> List[Dict[str, Any]]:
    """Sample finalists for testing with realistic Semantic Scholar metadata."""
    return [
        {
            "paperId": "paper1",
            "title": "Effects of Intermittent Fasting on Cardiovascular Health",
            "abstract": "This study examines the effects of intermittent fasting on cardiovascular risk factors.",
            "citationCount": 150,
            "influentialCitationCount": 25,
            "year": 2023,
            "venue": "Nature Medicine",
            "authors": [{"name": "John Smith"}, {"name": "Jane Doe"}],
            "openAccessPdf": {"url": "https://example.com/paper1.pdf"},
            "externalIds": {"DOI": "10.1234/nm.2023.001"},
        },
        {
            "paperId": "paper2",
            "title": "Time-Restricted Eating and Cardiac Function",
            "abstract": "A randomized controlled trial investigating how time-restricted eating affects cardiac function.",
            "citationCount": 200,
            "influentialCitationCount": 40,
            "year": 2022,
            "venue": "Cell Metabolism",
            "authors": [{"name": "Alice Brown"}],
            "openAccessPdf": {"url": "https://example.com/paper2.pdf"},
            "externalIds": {"DOI": "10.1234/cm.2022.001"},
        },
        {
            "paperId": "paper3",
            "title": "Fasting Protocols and Atherosclerosis Risk",
            "abstract": "Long-term follow-up study on fasting interventions and atherosclerosis.",
            "citationCount": 300,
            "influentialCitationCount": 55,
            "year": 2024,
            "venue": "Circulation",
            "authors": [{"name": "Bob Wilson"}, {"name": "Carol Davis"}],
            "openAccessPdf": {"url": "https://example.com/paper3.pdf"},
            "externalIds": {"DOI": "10.1234/c.2024.001"},
        },
    ]


@pytest.fixture
def mock_passages() -> List[Dict[str, Any]]:
    """Sample passages returned from PDF acquisition."""
    return [
        {
            "paperId": "paper1",
            "paper_id": "paper1",
            "title": "Effects of Intermittent Fasting on Cardiovascular Health",
            "text": "Intermittent fasting has gained considerable attention as a dietary intervention for improving cardiovascular health.",
            "citationCount": 150,
            "year": 2023,
            "venue": "Nature Medicine",
        },
        {
            "paperId": "paper2",
            "paper_id": "paper2",
            "title": "Time-Restricted Eating and Cardiac Function",
            "text": "Time-restricted eating has emerged as a popular dietary pattern with potential benefits for cardiac function.",
            "citationCount": 200,
            "year": 2022,
            "venue": "Cell Metabolism",
        },
        {
            "paperId": "paper3",
            "paper_id": "paper3",
            "title": "Fasting Protocols and Atherosclerosis Risk",
            "text": "Fasting protocols have been proposed as interventions to slow atherosclerosis disease progression.",
            "citationCount": 300,
            "year": 2024,
            "venue": "Circulation",
        },
    ]


@pytest.fixture
def mock_chunks() -> List[TextNode]:
    """Sample chunks returned from processing."""
    chunk_texts = [
        "Intermittent fasting has gained considerable attention as a dietary intervention for improving cardiovascular health.",
        "The intermittent fasting group showed significant reductions in blood pressure (systolic: -8.5 mmHg, p<0.001).",
        "LDL cholesterol decreased by 15% (p<0.01) and inflammatory markers including C-reactive protein fell by 25%.",
        "These findings suggest that intermittent fasting can effectively reduce cardiovascular risk factors.",
        "Time-restricted eating has emerged as a popular dietary pattern with potential benefits for cardiac function.",
        "TRE improved left ventricular ejection fraction by 4.2% (95% CI: 2.1-6.3).",
        "Heart rate variability increased, suggesting improved autonomic function.",
        "Atherosclerosis is a chronic inflammatory disease affecting arteries.",
        "Fasting activates AMPK and inhibits mTOR signaling, promoting cellular repair mechanisms.",
        "Enhanced autophagy helps clear damaged cellular components and may reduce atherosclerotic burden.",
        "Using Framingham Risk Score, periodic fasting reduces 10-year CVD risk by approximately 20-25%.",
        "Blood pressure reduction is comparable to first-line antihypertensive medications.",
    ]

    chunks = []
    for i, text in enumerate(chunk_texts):
        paper_idx = i % 3 + 1
        node = TextNode(
            text=text,
            metadata={
                "paper_id": f"paper{paper_idx}",
                "chunk_index": i,
                "paper_title": f"Test Paper {paper_idx}",
                "citation_count": 100 + (paper_idx * 50),
                "section_name": "Results" if i >= 6 else "Introduction",
            },
        )
        node.embedding = [0.1 + i * 0.01] * 384
        chunks.append(node)
    return chunks


@pytest.fixture
def mock_reranked_results(mock_chunks) -> List[NodeWithScore]:
    """Sample reranked results (top 12)."""
    return [
        NodeWithScore(node=chunk, score=0.95 - i * 0.02)
        for i, chunk in enumerate(mock_chunks[:12])
    ]


# ============================================================================
# Test Cases
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_happy_path_with_synthesis(
    mock_embedding_model,
    mock_finalists,
    mock_passages,
    mock_chunks,
    mock_reranked_results,
):
    """Test complete 10-stage pipeline: query → cited answer.

    This is the "happy path" test where all stages succeed:
        - Stages 1-4: Query optimization, search, quality gate, ranking
        - Stages 5-8: PDF acquisition, processing, retrieval, reranking
        - Stage 9: Synthesis generates a cited answer
        - Stage 10: Verification validates citations

    Validates:
        - PipelineResult.success is True
        - PipelineResult.answer is non-empty
        - PipelineResult.citation_verification['valid'] is True
        - PipelineResult.finalists is non-empty
        - PipelineResult.passages is non-empty
        - All 10 timing stages are present
    """
    # Arrange
    service = QueryService(embedding_model=mock_embedding_model)

    # Define the fixed synthesis response with valid citations
    synthesis_answer = (
        "Intermittent fasting has been shown to reduce blood pressure [1] "
        "and improve cardiac function [2]. Studies indicate a significant "
        "reduction in cardiovascular risk factors [3]."
    )

    # Mock all pipeline stages
    with (
        patch.object(service, "_get_optimizer") as mock_optimizer,
        patch.object(service, "_get_client") as mock_client,
        patch.object(service, "_get_ranker") as mock_ranker,
        patch.object(service, "_get_acquisition_agent") as mock_acquisition,
        patch.object(service, "_get_processing_agent") as mock_processing,
        patch.object(service, "_get_retriever") as mock_retriever,
        patch.object(service, "_get_reranker") as mock_reranker,
        patch.object(service, "_get_synthesis_agent") as mock_synthesis,
    ):
        # =====================================================================
        # Setup Week 1 mocks (Stages 1-4)
        # =====================================================================
        optimizer_instance = MagicMock()
        optimizer_instance.generate_dual_queries = AsyncMock(
            return_value={
                "primary": "intermittent fasting cardiovascular",
                "broad": "fasting heart health",
                "alternative": "time-restricted eating cardiovascular",
                "final_rephrase": "intermittent fasting cardiovascular risks",
            }
        )
        mock_optimizer.return_value = optimizer_instance

        client_instance = MagicMock()
        client_instance.multi_query_search = AsyncMock(
            return_value=[{"paperId": "p1", "title": "Test Paper", "citationCount": 50}]
        )
        mock_client.return_value = client_instance

        ranker_instance = MagicMock()
        ranker_instance.rank_papers.return_value = mock_finalists
        mock_ranker.return_value = ranker_instance

        # =====================================================================
        # Setup Week 2 mocks (Stages 5-8)
        # =====================================================================
        acquisition_instance = MagicMock()

        async def mock_acquire(state):
            state.update_fields(passages=mock_passages)
            return state

        acquisition_instance.process = AsyncMock(side_effect=mock_acquire)
        mock_acquisition.return_value = acquisition_instance

        processing_instance = MagicMock()

        def mock_process(state, embedding_model=None, batch_size=32):
            state.update_fields(chunks=mock_chunks)
            return state

        processing_instance.process = MagicMock(side_effect=mock_process)
        mock_processing.return_value = processing_instance

        retriever_instance = MagicMock()
        retriever_instance.retrieve.return_value = mock_reranked_results
        mock_retriever.return_value = retriever_instance

        reranker_instance = MagicMock()
        reranker_instance.rerank.return_value = mock_reranked_results
        mock_reranker.return_value = reranker_instance

        # =====================================================================
        # Setup Synthesis mock (Stage 9)
        # =====================================================================
        synthesis_instance = MagicMock()
        synthesis_instance.synthesize = AsyncMock(return_value=synthesis_answer)
        mock_synthesis.return_value = synthesis_instance

        # =====================================================================
        # Patch validate_quality for Stage 3
        # =====================================================================
        with patch("services.query_service.validate_quality") as mock_quality:
            mock_quality.return_value = {"passed": True, "metrics": {}}

            # Act
            result = await service.process_query(
                "What are the cardiovascular risks of intermittent fasting?"
            )

    # =========================================================================
    # Assertions
    # =========================================================================

    # 1. Pipeline should succeed
    assert result.success is True, (
        f"Pipeline should succeed, got error: {result.error_message} "
        f"at stage: {result.error_stage}"
    )

    # 2. Result should be a PipelineResult
    assert isinstance(result, PipelineResult), "Result should be a PipelineResult"

    # 3. Answer should be non-empty
    assert result.answer is not None, "Answer should not be None"
    assert len(result.answer) > 0, "Answer should not be empty"
    assert result.answer == synthesis_answer, "Answer should match expected synthesis"

    # 4. Citation verification should be valid
    assert (
        result.citation_verification is not None
    ), "Citation verification should not be None"
    assert isinstance(
        result.citation_verification, dict
    ), "Citation verification should be a dict"
    assert (
        result.citation_verification["valid"] is True
    ), f"Citations should be valid, got: {result.citation_verification}"
    assert (
        "citations_found" in result.citation_verification
    ), "Should have citations_found key"
    assert (
        len(result.citation_verification["citations_found"]) > 0
    ), "Should have found citations"

    # 5. Finalists should be non-empty
    assert result.finalists is not None, "Finalists should not be None"
    assert len(result.finalists) > 0, "Finalists should not be empty"

    # 6. Passages should be non-empty
    assert result.passages is not None, "Passages should not be None"
    assert len(result.passages) > 0, "Passages should not be empty"

    # 7. Verify passage structure
    for passage in result.passages:
        assert isinstance(
            passage, NodeWithScore
        ), "Passage should be NodeWithScore object"
        assert hasattr(passage, "node"), "Passage should have node attribute"
        assert hasattr(passage, "score"), "Passage should have score attribute"
        assert passage.score is not None, "Passage score should not be None"

    # 8. Verify timing breakdown includes all 10 stages
    expected_stages = [
        "query_optimization",
        "semantic_scholar_search",
        "quality_gate",
        "composite_scoring",
        "pdf_acquisition",
        "processing",
        "retrieval",
        "reranking",
        "synthesis",
        "verification",
    ]

    for stage in expected_stages:
        assert stage in result.timing_breakdown, f"Missing timing for stage: {stage}"
        assert (
            result.timing_breakdown[stage] >= 0
        ), f"Stage {stage} should have non-negative timing"

    # 9. Verify optimized queries are present
    assert result.optimized_queries is not None, "Optimized queries should not be None"
    assert "primary" in result.optimized_queries, "Should have primary query"
    assert "final_rephrase" in result.optimized_queries, "Should have final_rephrase"

    # 10. Verify quality gate passed
    assert result.quality_gate is not None, "Quality gate should not be None"
    assert result.quality_gate["passed"] is True, "Quality gate should pass"

    # 11. Verify execution time is reasonable
    assert result.execution_time > 0, "Execution time should be positive"
    assert (
        result.execution_time < 60.0
    ), f"Pipeline should complete in <60s, took {result.execution_time:.2f}s"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_with_synthesis_failure_continues_gracefully(
    mock_embedding_model,
    mock_finalists,
    mock_passages,
    mock_chunks,
    mock_reranked_results,
):
    """Test that pipeline continues gracefully when synthesis fails.

    Validates:
        - Pipeline still succeeds even if synthesis throws an error
        - answer is None but other fields are populated
        - citation_verification is None when no answer
    """
    # Arrange
    service = QueryService(embedding_model=mock_embedding_model)

    with (
        patch.object(service, "_get_optimizer") as mock_optimizer,
        patch.object(service, "_get_client") as mock_client,
        patch.object(service, "_get_ranker") as mock_ranker,
        patch.object(service, "_get_acquisition_agent") as mock_acquisition,
        patch.object(service, "_get_processing_agent") as mock_processing,
        patch.object(service, "_get_retriever") as mock_retriever,
        patch.object(service, "_get_reranker") as mock_reranker,
        patch.object(service, "_get_synthesis_agent") as mock_synthesis,
    ):
        # Setup Week 1 mocks
        optimizer_instance = MagicMock()
        optimizer_instance.generate_dual_queries = AsyncMock(
            return_value={
                "primary": "test query",
                "broad": "test broad",
                "alternative": "test alt",
                "final_rephrase": "test rephrase",
            }
        )
        mock_optimizer.return_value = optimizer_instance

        client_instance = MagicMock()
        client_instance.multi_query_search = AsyncMock(
            return_value=[{"paperId": "p1", "title": "Test", "citationCount": 50}]
        )
        mock_client.return_value = client_instance

        ranker_instance = MagicMock()
        ranker_instance.rank_papers.return_value = mock_finalists
        mock_ranker.return_value = ranker_instance

        # Setup Week 2 mocks
        acquisition_instance = MagicMock()

        async def mock_acquire(state):
            state.update_fields(passages=mock_passages)
            return state

        acquisition_instance.process = AsyncMock(side_effect=mock_acquire)
        mock_acquisition.return_value = acquisition_instance

        processing_instance = MagicMock()

        def mock_process(state, embedding_model=None, batch_size=32):
            state.update_fields(chunks=mock_chunks)
            return state

        processing_instance.process = MagicMock(side_effect=mock_process)
        mock_processing.return_value = processing_instance

        retriever_instance = MagicMock()
        retriever_instance.retrieve.return_value = mock_reranked_results
        mock_retriever.return_value = retriever_instance

        reranker_instance = MagicMock()
        reranker_instance.rerank.return_value = mock_reranked_results
        mock_reranker.return_value = reranker_instance

        # Make synthesis fail
        synthesis_instance = MagicMock()
        synthesis_instance.synthesize = AsyncMock(
            side_effect=Exception("LLM API error")
        )
        mock_synthesis.return_value = synthesis_instance

        with patch("services.query_service.validate_quality") as mock_quality:
            mock_quality.return_value = {"passed": True, "metrics": {}}

            # Act
            result = await service.process_query("test query")

    # Assertions
    assert result.success is True, "Pipeline should still succeed"
    assert result.answer is None, "Answer should be None when synthesis fails"
    assert (
        result.citation_verification is None
    ), "Verification should be None when no answer"
    assert len(result.finalists) > 0, "Finalists should still be present"
    assert len(result.passages) > 0, "Passages should still be present"
    assert (
        "synthesis" in result.timing_breakdown
    ), "Synthesis timing should be recorded even on failure"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_with_invalid_citations(
    mock_embedding_model,
    mock_finalists,
    mock_passages,
    mock_chunks,
    mock_reranked_results,
):
    """Test citation verification detects invalid citations.

    Validates:
        - Pipeline succeeds but citation_verification['valid'] is False
        - Invalid citations are captured in the result
    """
    # Arrange
    service = QueryService(embedding_model=mock_embedding_model)

    # Answer with invalid citation [99] that doesn't exist in passages
    invalid_answer = "Fasting is beneficial [1] and has many effects [99]."

    with (
        patch.object(service, "_get_optimizer") as mock_optimizer,
        patch.object(service, "_get_client") as mock_client,
        patch.object(service, "_get_ranker") as mock_ranker,
        patch.object(service, "_get_acquisition_agent") as mock_acquisition,
        patch.object(service, "_get_processing_agent") as mock_processing,
        patch.object(service, "_get_retriever") as mock_retriever,
        patch.object(service, "_get_reranker") as mock_reranker,
        patch.object(service, "_get_synthesis_agent") as mock_synthesis,
    ):
        # Setup mocks (simplified for this test)
        optimizer_instance = MagicMock()
        optimizer_instance.generate_dual_queries = AsyncMock(
            return_value={
                "primary": "test",
                "broad": "test",
                "alternative": "test",
                "final_rephrase": "test",
            }
        )
        mock_optimizer.return_value = optimizer_instance

        client_instance = MagicMock()
        client_instance.multi_query_search = AsyncMock(
            return_value=[{"paperId": "p1", "title": "Test", "citationCount": 50}]
        )
        mock_client.return_value = client_instance

        ranker_instance = MagicMock()
        ranker_instance.rank_papers.return_value = mock_finalists
        mock_ranker.return_value = ranker_instance

        acquisition_instance = MagicMock()

        async def mock_acquire(state):
            state.update_fields(passages=mock_passages)
            return state

        acquisition_instance.process = AsyncMock(side_effect=mock_acquire)
        mock_acquisition.return_value = acquisition_instance

        processing_instance = MagicMock()

        def mock_process(state, embedding_model=None, batch_size=32):
            state.update_fields(chunks=mock_chunks)
            return state

        processing_instance.process = MagicMock(side_effect=mock_process)
        mock_processing.return_value = processing_instance

        retriever_instance = MagicMock()
        retriever_instance.retrieve.return_value = mock_reranked_results
        mock_retriever.return_value = retriever_instance

        reranker_instance = MagicMock()
        reranker_instance.rerank.return_value = mock_reranked_results
        mock_reranker.return_value = reranker_instance

        # Return answer with invalid citation
        synthesis_instance = MagicMock()
        synthesis_instance.synthesize = AsyncMock(return_value=invalid_answer)
        mock_synthesis.return_value = synthesis_instance

        with patch("services.query_service.validate_quality") as mock_quality:
            mock_quality.return_value = {"passed": True, "metrics": {}}

            # Act
            result = await service.process_query("test query")

    # Assertions
    assert result.success is True, "Pipeline should succeed"
    assert result.answer == invalid_answer, "Answer should be present"
    assert result.citation_verification is not None, "Verification should run"
    assert (
        result.citation_verification["valid"] is False
    ), "Should detect invalid citation [99]"
    assert (
        99 in result.citation_verification["invalid_citations"]
    ), "Should list 99 as invalid"
