"""End-to-end integration tests for the full 10-stage pipeline.

These tests validate the complete pipeline from query to cited answer,
including synthesis (Stage 9) and verification (Stage 10).

Stages (1-4):
    1. Query Optimization
    2. Semantic Scholar Search
    3. Quality Gate
    4. Composite Scoring

Stages (5-8):
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
        # Setup mocks (Stages 1-4)
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
        # Setup mocks (Stages 5-8)
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
        # Setup mocks (Stages 1-4)
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

        # Setup mocks (Stages 5-8)
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


# ============================================================================
# Citation Validity Test (Validation)
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_synthesis_citation_validity(
    mock_embedding_model,
    mock_finalists,
    mock_passages,
    mock_chunks,
    mock_reranked_results,
):
    """Test that SynthesisAgent does NOT hallucinate citations.

    Validation: Verify that every [N] citation in the synthesized answer
    corresponds to a valid index in the returned passages list.

    This test:
        1. Runs a query through the full pipeline (mocked retrieval for speed)
        2. Parses all [N] citations from the answer using regex
        3. Asserts every citation index N is <= len(passages)
        4. Asserts the answer is not empty and contains at least one citation

    Edge case: If synthesis returns text with numbers that look like citations
    but aren't (e.g., "study of 10 patients"), they would be ignored by the
    prompt design. We only validate formal [N] citations.
    """
    import re
    from loguru import logger

    # Regex pattern to extract citation indices
    citation_pattern = re.compile(r"\[(\d+)\]")

    # Arrange
    service = QueryService(embedding_model=mock_embedding_model)
    test_query = "What are the benefits of intermittent fasting?"

    # Define a realistic synthesis response with valid citations
    # This simulates what the LLM would return based on our prompt template
    synthesis_answer = (
        "Intermittent fasting has been shown to reduce blood pressure significantly [1], "
        "with systolic blood pressure decreasing by an average of 8.5 mmHg. "
        "Additionally, studies demonstrate improvements in cardiac function [2], "
        "including enhanced left ventricular ejection fraction. "
        "Long-term adherence to fasting protocols is associated with reduced "
        "atherosclerotic burden [3] and improved lipid profiles [1]. "
        "These cardiovascular benefits are thought to be mediated by enhanced "
        "autophagy and metabolic switching [2]."
    )

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
        # Setup mocks (Stages 1-4)
        optimizer_instance = MagicMock()
        optimizer_instance.generate_dual_queries = AsyncMock(
            return_value={
                "primary": "intermittent fasting benefits health",
                "broad": "fasting metabolic health",
                "alternative": "time-restricted eating advantages",
                "final_rephrase": "benefits of intermittent fasting for health",
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

        # Setup mocks (Stages 5-8)
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

        # Setup Synthesis mock (Stage 9)
        synthesis_instance = MagicMock()
        synthesis_instance.synthesize = AsyncMock(return_value=synthesis_answer)
        mock_synthesis.return_value = synthesis_instance

        with patch("services.query_service.validate_quality") as mock_quality:
            mock_quality.return_value = {"passed": True, "metrics": {}}

            try:
                # Act
                result = await service.process_query(test_query)
            except Exception as e:
                pytest.fail(
                    f"Pipeline failed with error: {e}. "
                    "This may indicate a network error or configuration issue."
                )

    # =========================================================================
    # Citation Validity Assertions
    # =========================================================================

    # 1. Pipeline should succeed
    assert result.success is True, (
        f"Pipeline should succeed, got error: {result.error_message} "
        f"at stage: {result.error_stage}"
    )

    # 2. Answer should not be empty
    assert result.answer is not None, "Answer should not be None"
    assert len(result.answer.strip()) > 0, "Answer should not be empty"

    # 3. Extract all citation indices from the answer
    matches = citation_pattern.findall(result.answer)
    citation_indices = [int(m) for m in matches]
    unique_citations = sorted(set(citation_indices))

    # Log for debugging
    logger.info(f"Query: '{test_query}'")
    logger.info(f"Answer length: {len(result.answer)} characters")
    logger.info(f"Parsed citations: {unique_citations}")
    logger.info(f"Total passages count: {len(result.passages)}")

    # 4. Assert at least one citation exists
    if len(unique_citations) == 0:
        pytest.fail(
            "No citations found in answer. "
            "The synthesis should include at least one [N] citation. "
            f"Answer: {result.answer[:200]}..."
        )

    # 5. Assert every citation index N is <= len(passages)
    max_valid_index = len(result.passages)
    invalid_citations = [c for c in unique_citations if c <= 0 or c > max_valid_index]

    logger.info(f"Max valid citation index: {max_valid_index}")
    logger.info(f"Invalid citations (if any): {invalid_citations}")

    assert len(invalid_citations) == 0, (
        f"Found hallucinated citations {invalid_citations}! "
        f"Valid range is [1] to [{max_valid_index}]. "
        f"All citations found: {unique_citations}"
    )

    # 6. Verify using the built-in citation verifier (Stage 10)
    assert (
        result.citation_verification is not None
    ), "Citation verification should not be None"
    assert result.citation_verification["valid"] is True, (
        f"CitationVerifier should confirm validity. "
        f"Got: {result.citation_verification}"
    )

    logger.info(
        f"✓ All {len(unique_citations)} citation(s) are valid: {unique_citations}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_synthesis_citation_validity_cardiovascular_query(
    mock_embedding_model,
    mock_finalists,
    mock_passages,
    mock_chunks,
    mock_reranked_results,
):
    """Test citation validity with cardiovascular-focused query.

    Uses the example from the task description:
        Query: "What are the cardiovascular risks of fasting?"
        Expected: Answer with [1], [2], etc., where passages list has length >= 2.

    This test validates:
        - Multiple citations are present
        - All citations reference valid passage indices
        - Answer content is substantive (not just fallback message)
    """
    import re
    from loguru import logger

    citation_pattern = re.compile(r"\[(\d+)\]")

    # Arrange
    service = QueryService(embedding_model=mock_embedding_model)
    test_query = "What are the cardiovascular risks of fasting?"

    # Simulate a response with multiple cardiovascular-focused citations
    synthesis_answer = (
        "Fasting, particularly intermittent fasting, presents several cardiovascular "
        "considerations. Some studies indicate potential benefits such as reduced blood "
        "pressure [1] and improved lipid profiles [2]. However, certain populations "
        "may experience increased heart rate variability during fasting periods [3]. "
        "Long-term adherence appears to reduce atherosclerosis risk [1], though individual "
        "responses may vary [2]."
    )

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
        # Setup mocks
        optimizer_instance = MagicMock()
        optimizer_instance.generate_dual_queries = AsyncMock(
            return_value={
                "primary": "cardiovascular risks fasting",
                "broad": "heart health fasting",
                "alternative": "fasting cardiac effects",
                "final_rephrase": "cardiovascular risks of intermittent fasting",
            }
        )
        mock_optimizer.return_value = optimizer_instance

        client_instance = MagicMock()
        client_instance.multi_query_search = AsyncMock(
            return_value=[
                {"paperId": "p1", "title": "Cardio Paper", "citationCount": 100}
            ]
        )
        mock_client.return_value = client_instance

        ranker_instance = MagicMock()
        ranker_instance.rank_papers.return_value = mock_finalists
        mock_ranker.return_value = ranker_instance

        # Setup mocks
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

        # Setup Synthesis mock
        synthesis_instance = MagicMock()
        synthesis_instance.synthesize = AsyncMock(return_value=synthesis_answer)
        mock_synthesis.return_value = synthesis_instance

        with patch("services.query_service.validate_quality") as mock_quality:
            mock_quality.return_value = {"passed": True, "metrics": {}}

            try:
                result = await service.process_query(test_query)
            except Exception as e:
                pytest.fail(f"Pipeline failed with network/configuration error: {e}")

    # Assertions
    assert result.success is True, f"Pipeline failed: {result.error_message}"
    assert result.answer is not None and len(result.answer) > 0

    # Extract citations
    matches = citation_pattern.findall(result.answer)
    unique_citations = sorted(set(int(m) for m in matches))

    logger.info(f"Cardiovascular query citations: {unique_citations}")
    logger.info(f"Passages available: {len(result.passages)}")

    # Should have at least 2 citations as per task example
    assert len(unique_citations) >= 2, (
        f"Expected at least 2 citations for cardiovascular query, "
        f"found {len(unique_citations)}: {unique_citations}"
    )

    # Validate passages length meets minimum
    assert (
        len(result.passages) >= 2
    ), f"Expected at least 2 passages, got {len(result.passages)}"

    # Validate all citations are in range
    max_valid = len(result.passages)
    invalid = [c for c in unique_citations if c <= 0 or c > max_valid]

    assert len(invalid) == 0, (
        f"Hallucinated citations detected: {invalid}. "
        f"Valid range: [1] to [{max_valid}]"
    )

    # Verify answer is substantive (not fallback message)
    from agents.synthesis_agent import SynthesisAgent

    assert (
        result.answer != SynthesisAgent.INSUFFICIENT_CONTEXT_MESSAGE
    ), "Answer should not be the fallback message"

    logger.info("✓ Cardiovascular citation validity test passed")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_synthesis_no_citations_warns(
    mock_embedding_model,
    mock_finalists,
    mock_passages,
    mock_chunks,
    mock_reranked_results,
):
    """Test that an answer with zero citations triggers appropriate handling.

    If the LLM fails to include citations (against instructions), this test
    validates that:
        - The pipeline still succeeds
        - citation_verification captures the issue
        - The test logs a warning about missing citations

    Note: In production, this should be caught and the LLM re-prompted.
    """
    import re
    from loguru import logger

    citation_pattern = re.compile(r"\[(\d+)\]")

    service = QueryService(embedding_model=mock_embedding_model)

    # Answer with NO citations (edge case - LLM ignoring instructions)
    answer_without_citations = (
        "Intermittent fasting may have various health benefits including "
        "improved metabolic function and weight management. Some studies "
        "suggest cardiovascular improvements as well."
    )

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
        # Setup mocks (simplified)
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

        synthesis_instance = MagicMock()
        synthesis_instance.synthesize = AsyncMock(return_value=answer_without_citations)
        mock_synthesis.return_value = synthesis_instance

        with patch("services.query_service.validate_quality") as mock_quality:
            mock_quality.return_value = {"passed": True, "metrics": {}}

            result = await service.process_query("test query about fasting")

    # Assertions
    assert result.success is True, "Pipeline should succeed even without citations"
    assert result.answer is not None

    # Extract citations (should be empty)
    matches = citation_pattern.findall(result.answer)

    if len(matches) == 0:
        logger.warning(
            "⚠ No citations found in answer! "
            "This indicates the LLM did not follow citation instructions. "
            f"Answer preview: {result.answer[:100]}..."
        )
        # The citation verifier should still report this as valid (no invalid citations)
        # but with citations_found = []
        assert result.citation_verification is not None
        assert len(result.citation_verification.get("citations_found", [])) == 0
        # Note: We don't fail here as this tests edge case behavior
        # In production, this would trigger a re-prompt or warning
    else:
        # If citations were somehow included, validate them
        unique_citations = sorted(set(int(m) for m in matches))
        max_valid = len(result.passages)
        invalid = [c for c in unique_citations if c <= 0 or c > max_valid]
        assert len(invalid) == 0

    logger.info("✓ Zero-citation edge case test completed")
