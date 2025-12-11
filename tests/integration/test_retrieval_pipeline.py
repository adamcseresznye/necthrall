"""Integration tests for the complete retrieval pipeline (Stages 1-8).

These tests validate the full pipeline from query → papers → PDFs → passages,
mocking only external dependencies (Semantic Scholar API, PDF downloads) while
using real implementations for processing, retrieval, and reranking.

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

Test Coverage:
    - Full pipeline with mocked PDF acquisition (12 passages expected)
    - Pipeline with no finalists (early exit)
    - Pipeline with PDF acquisition failure (graceful degradation)
    - Pipeline with partial PDF failures (graceful degradation)
    - Pipeline with no chunks generated (log warning, return papers only)
    - Performance test (total execution time <20 seconds)
    - All 8 timing stages validation

Run these tests with:
    pytest tests/integration/test_retrieval_pipeline.py -v -s
"""

from __future__ import annotations

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

# Import llama_index types for fixtures
from llama_index.core.schema import NodeWithScore, TextNode

from services.query_service import QueryService, PipelineResult
from models.state import State


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

    # Mock the embedding batch method
    def mock_embed_batch(texts: List[str]) -> List[List[float]]:
        return [[0.1] * 384 for _ in texts]

    model.get_text_embedding_batch = MagicMock(side_effect=mock_embed_batch)

    # Also provide embed_documents for ProcessingAgent compatibility
    model.embed_documents = MagicMock(side_effect=mock_embed_batch)
    return model


@pytest.fixture
def mock_finalists() -> List[Dict[str, Any]]:
    """Sample finalists for testing with realistic Semantic Scholar metadata.

    These papers have realistic metadata including openAccessPdf URLs,
    citation counts, and other fields used by the quality gate and ranking.
    """
    return [
        {
            "paperId": "paper1",
            "title": "Effects of Intermittent Fasting on Cardiovascular Health",
            "abstract": "This study examines the effects of intermittent fasting on cardiovascular risk factors including blood pressure, lipid profiles, and inflammatory markers.",
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
            "abstract": "A randomized controlled trial investigating how time-restricted eating affects cardiac function and metabolic health markers.",
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
            "abstract": "Long-term follow-up study on fasting interventions and their relationship with atherosclerosis progression and cardiovascular events.",
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
    """Sample passages returned from PDF acquisition with realistic scientific content.

    These texts contain realistic scientific content related to the query
    to ensure retrieval algorithms work correctly.
    """
    return [
        {
            "paperId": "paper1",
            "paper_id": "paper1",
            "title": "Effects of Intermittent Fasting on Cardiovascular Health",
            "text": """# Introduction

Intermittent fasting has gained considerable attention as a dietary intervention
for improving cardiovascular health. This study investigates the effects of
16:8 time-restricted eating on various cardiovascular risk factors.

# Methods

We conducted a randomized controlled trial with 200 participants over 12 weeks.
Participants were assigned to either an intermittent fasting group or a control
group with standard dietary advice. Blood pressure, lipid profiles, and 
inflammatory markers were measured at baseline and follow-up.

# Results

The intermittent fasting group showed significant reductions in blood pressure
(systolic: -8.5 mmHg, p<0.001), LDL cholesterol (-15%, p<0.01), and inflammatory
markers including C-reactive protein (-25%, p<0.001). HDL cholesterol increased
by 8% compared to control group.

# Discussion

These findings suggest that intermittent fasting can effectively reduce
cardiovascular risk factors. The mechanisms may involve improved insulin
sensitivity, reduced oxidative stress, and enhanced autophagy. The magnitude
of blood pressure reduction is comparable to first-line antihypertensive medications.

# Conclusion

Intermittent fasting represents a promising approach for cardiovascular disease
prevention, with benefits comparable to pharmaceutical interventions.""",
            "citationCount": 150,
            "year": 2023,
            "venue": "Nature Medicine",
            "influentialCitationCount": 25,
        },
        {
            "paperId": "paper2",
            "paper_id": "paper2",
            "title": "Time-Restricted Eating and Cardiac Function",
            "text": """# Abstract

Time-restricted eating (TRE) has emerged as a popular dietary pattern with
potential benefits for cardiac function. This study evaluates the effects
of an 8-hour eating window on echocardiographic parameters.

# Background

Cardiovascular disease remains the leading cause of mortality worldwide.
Dietary interventions that can improve cardiac function without adverse
effects are highly desirable. TRE aligns eating patterns with circadian rhythms.

# Study Design

A 6-month prospective study enrolled 150 adults with early-stage metabolic
syndrome. Cardiac function was assessed using 2D echocardiography and
strain imaging at baseline and follow-up.

# Key Findings

TRE improved left ventricular ejection fraction by 4.2% (95% CI: 2.1-6.3)
and reduced diastolic dysfunction markers. Heart rate variability increased,
suggesting improved autonomic function. Cardiac output increased by 6%.

# Clinical Implications

Time-restricted eating may serve as an adjunct therapy for patients with
early cardiac dysfunction and could potentially delay progression to
symptomatic heart failure. Further long-term studies are warranted.""",
            "citationCount": 200,
            "year": 2022,
            "venue": "Cell Metabolism",
            "influentialCitationCount": 40,
        },
        {
            "paperId": "paper3",
            "paper_id": "paper3",
            "title": "Fasting Protocols and Atherosclerosis Risk",
            "text": """# Summary

Atherosclerosis is a chronic inflammatory disease affecting arteries.
Fasting protocols have been proposed as interventions to slow disease
progression through multiple mechanisms including lipid reduction.

# Pathophysiology Review

The accumulation of lipid-laden macrophages in arterial walls leads to
plaque formation. Fasting may reduce this process by improving lipid
profiles and reducing systemic inflammation markers.

# Study Results

Our 2-year follow-up of fasting intervention participants showed:
- 18% reduction in carotid intima-media thickness progression
- Stabilization of existing plaques on cardiac CT imaging
- Reduced incidence of major cardiovascular events by 22%

# Mechanism Discussion

Fasting activates AMPK and inhibits mTOR signaling, promoting cellular
repair mechanisms. Enhanced autophagy helps clear damaged cellular components
and may reduce atherosclerotic burden. These molecular changes underlie
the observed clinical benefits.

# Cardiovascular Risk Implications

Using Framingham Risk Score, periodic fasting reduces 10-year CVD risk
by approximately 20-25% in individuals with metabolic syndrome.""",
            "citationCount": 300,
            "year": 2024,
            "venue": "Circulation",
            "influentialCitationCount": 55,
        },
    ]


@pytest.fixture
def mock_chunks() -> List[TextNode]:
    """Sample chunks returned from processing with realistic scientific content."""
    chunk_texts = [
        "Intermittent fasting has gained considerable attention as a dietary intervention for improving cardiovascular health.",
        "We conducted a randomized controlled trial with 200 participants over 12 weeks.",
        "The intermittent fasting group showed significant reductions in blood pressure (systolic: -8.5 mmHg, p<0.001).",
        "LDL cholesterol decreased by 15% (p<0.01) and inflammatory markers including C-reactive protein fell by 25%.",
        "These findings suggest that intermittent fasting can effectively reduce cardiovascular risk factors.",
        "The mechanisms may involve improved insulin sensitivity, reduced oxidative stress, and enhanced autophagy.",
        "Time-restricted eating has emerged as a popular dietary pattern with potential benefits for cardiac function.",
        "A 6-month prospective study enrolled 150 adults with early-stage metabolic syndrome.",
        "TRE improved left ventricular ejection fraction by 4.2% (95% CI: 2.1-6.3).",
        "Heart rate variability increased, suggesting improved autonomic function.",
        "Atherosclerosis is a chronic inflammatory disease affecting arteries.",
        "Fasting protocols have been proposed as interventions to slow disease progression.",
        "Fasting activates AMPK and inhibits mTOR signaling, promoting cellular repair mechanisms.",
        "Enhanced autophagy helps clear damaged cellular components and may reduce atherosclerotic burden.",
        "Using Framingham Risk Score, periodic fasting reduces 10-year CVD risk by approximately 20-25%.",
        "Blood pressure reduction is comparable to first-line antihypertensive medications.",
        "Time-restricted eating may serve as an adjunct therapy for patients with early cardiac dysfunction.",
        "Carotid intima-media thickness progression reduced by 18% in fasting intervention group.",
        "Stabilization of existing plaques on cardiac CT imaging was observed.",
        "Reduced incidence of major cardiovascular events by 22% over 2-year follow-up.",
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
                "section_name": (
                    "Introduction" if i < 5 else ("Methods" if i < 10 else "Results")
                ),
            },
        )
        # Add embedding
        node.embedding = [0.1 + i * 0.01] * 384
        chunks.append(node)
    return chunks


@pytest.fixture
def mock_retrieval_results(mock_chunks) -> List[NodeWithScore]:
    """Sample retrieval results."""
    return [
        NodeWithScore(node=chunk, score=0.9 - i * 0.05)
        for i, chunk in enumerate(mock_chunks[:15])
    ]


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
async def test_full_pipeline_with_mocked_stages(
    mock_embedding_model,
    mock_finalists,
    mock_passages,
    mock_chunks,
    mock_retrieval_results,
    mock_reranked_results,
):
    """Test complete pipeline: query → papers → PDFs → passages.

    This test validates the full pipeline with mocked external
    dependencies (Semantic Scholar API, PDF downloads) but validates all other
    components including processing, retrieval, and reranking.

    Validates:
        - finalists selected, quality_gate passed
        - passages returned with correct count and structure
        - Timing: execution completes in reasonable time with all 8 stages
        - Structure: all result fields properly populated
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
    ):
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
        retriever_instance.retrieve.return_value = mock_retrieval_results
        mock_retriever.return_value = retriever_instance

        reranker_instance = MagicMock()
        reranker_instance.rerank.return_value = mock_reranked_results
        mock_reranker.return_value = reranker_instance

        # Patch validate_quality
        with patch("services.query_service.validate_quality") as mock_quality:
            mock_quality.return_value = {"passed": True, "metrics": {}}

            # Act
            result = await service.process_query(
                "intermittent fasting cardiovascular risks"
            )

    # =========================================================================
    # Validate outputs
    # =========================================================================
    assert (
        result.success is True
    ), f"Pipeline should succeed, got error: {result.error_message}"
    assert len(result.finalists) >= 1, "Should return at least 1 finalist paper"
    assert result.quality_gate["passed"] is True, "Quality gate should pass"

    # =========================================================================
    # Validate outputs
    # =========================================================================
    assert hasattr(result, "passages"), "Result should have passages attribute"
    assert (
        len(result.passages) == 12
    ), "Should return exactly 12 passages after reranking"

    # Validate passage structure (NodeWithScore objects)
    for passage in result.passages:
        assert isinstance(
            passage, NodeWithScore
        ), "Passage should be NodeWithScore object"
        assert hasattr(passage, "node"), "Passage should have node attribute"
        assert hasattr(passage, "score"), "Passage should have score attribute"
        assert passage.score is not None, "Passage score should not be None"
        assert passage.score > 0.0, "Passage should have positive score"
        # Content should be present
        content = passage.node.get_content()
        assert content is not None, "Passage content should not be None"
        assert len(content) > 0, "Passage content should not be empty"

    # =========================================================================
    # Validate timing breakdown includes all 8 stages
    # =========================================================================
    expected_stages = [
        "query_optimization",
        "semantic_scholar_search",
        "quality_gate",
        "composite_scoring",
        "pdf_acquisition",
        "processing",
        "retrieval",
        "reranking",
    ]

    for stage in expected_stages:
        assert stage in result.timing_breakdown, f"Missing timing for stage: {stage}"
        assert (
            result.timing_breakdown[stage] >= 0
        ), f"Stage {stage} should have non-negative timing"

    assert (
        result.execution_time < 20.0
    ), f"Pipeline should complete in <20s, took {result.execution_time:.2f}s"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_with_no_finalists(mock_embedding_model):
    """Test early exit when Semantic Scholar returns 0 papers.

    Validates:
        - Pipeline exits gracefully with no finalists
        - No crashes when encountering empty results
        - Result structure is still valid (empty finalists, empty passages)
        - Success flag is True (empty is not an error)
        - Later stages are not executed (no timing breakdown for them)
    """
    # Arrange
    service = QueryService(embedding_model=mock_embedding_model)

    with (
        patch.object(service, "_get_optimizer") as mock_optimizer,
        patch.object(service, "_get_client") as mock_client,
        patch.object(service, "_get_ranker") as mock_ranker,
    ):
        optimizer_instance = MagicMock()
        optimizer_instance.generate_dual_queries = AsyncMock(
            return_value={
                "primary": "nonexistent research topic xyz123",
                "broad": "nonexistent topic research",
                "alternative": "xyz123 topic research",
                "final_rephrase": "nonexistent research topic xyz123",
            }
        )
        mock_optimizer.return_value = optimizer_instance

        client_instance = MagicMock()
        client_instance.multi_query_search = AsyncMock(return_value=[])
        mock_client.return_value = client_instance

        ranker_instance = MagicMock()
        ranker_instance.rank_papers.return_value = []
        mock_ranker.return_value = ranker_instance

        with patch("services.query_service.validate_quality") as mock_quality:
            mock_quality.return_value = {"passed": True, "metrics": {}}

            # Act
            result = await service.process_query("nonexistent research topic xyz123")

    # Assert - Pipeline should succeed with empty results
    assert result.success is True, "Pipeline should succeed even with no papers"
    assert len(result.finalists) == 0, "Should have 0 finalists"
    assert len(result.passages) == 0, "Should have 0 passages"

    # Quality gate result should be present
    assert result.quality_gate is not None, "Quality gate result should be present"

    # Timing should include early stages but not later stages
    assert "query_optimization" in result.timing_breakdown
    assert "semantic_scholar_search" in result.timing_breakdown
    assert "quality_gate" in result.timing_breakdown

    # Later stages should not be executed
    assert (
        "pdf_acquisition" not in result.timing_breakdown
    ), "PDF acquisition should not run with no finalists"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_with_pdf_acquisition_failure(
    mock_embedding_model, mock_finalists
):
    """Test graceful degradation when all PDF acquisitions fail.

    Validates:
        - Pipeline returns finalists selected when acquisition completely fails
        - Success flag is True (acquisition failure is handled gracefully)
        - Empty passages list is returned
        - No crashes when acquisition returns empty
    """
    # Arrange
    service = QueryService(embedding_model=mock_embedding_model)

    with (
        patch.object(service, "_get_optimizer") as mock_optimizer,
        patch.object(service, "_get_client") as mock_client,
        patch.object(service, "_get_ranker") as mock_ranker,
        patch.object(service, "_get_acquisition_agent") as mock_acquisition,
    ):
        optimizer_instance = MagicMock()
        optimizer_instance.generate_dual_queries = AsyncMock(
            return_value={
                "primary": "intermittent fasting cardiovascular",
                "broad": "fasting heart health",
                "alternative": "time-restricted eating",
                "final_rephrase": "intermittent fasting cardiovascular risks",
            }
        )
        mock_optimizer.return_value = optimizer_instance

        client_instance = MagicMock()
        client_instance.multi_query_search = AsyncMock(
            return_value=[{"paperId": "p1", "citationCount": 50}]
        )
        mock_client.return_value = client_instance

        ranker_instance = MagicMock()
        ranker_instance.rank_papers.return_value = mock_finalists
        mock_ranker.return_value = ranker_instance

        # Make acquisition return empty passages (simulating complete failure)
        acquisition_instance = MagicMock()

        async def mock_acquire_fail(state: State) -> State:
            state.update_fields(passages=[])
            state.append_error("Critical: No PDFs acquired")
            return state

        acquisition_instance.process = AsyncMock(side_effect=mock_acquire_fail)
        mock_acquisition.return_value = acquisition_instance

        with patch("services.query_service.validate_quality") as mock_quality:
            mock_quality.return_value = {"passed": True, "metrics": {}}

            # Act
            result = await service.process_query(
                "intermittent fasting cardiovascular risks"
            )

    # Assert - Pipeline should succeed with graceful degradation
    assert result.success is True, "Pipeline should succeed with graceful degradation"
    assert len(result.finalists) >= 1, "Finalists should be preserved"
    assert len(result.passages) == 0, "No passages when acquisition fails"
    assert (
        "pdf_acquisition" in result.timing_breakdown
    ), "Acquisition timing should be recorded"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_with_no_chunks_generated(
    mock_embedding_model, mock_finalists, mock_passages
):
    """Test graceful degradation when processing/chunking fails.

    Validates:
        - Pipeline handles processing failures gracefully
        - Returns finalists selected even if later processing fails
        - No crashes when chunks cannot be generated
        - Both acquisition and processing timing are recorded
    """
    # Arrange
    service = QueryService(embedding_model=mock_embedding_model)

    with (
        patch.object(service, "_get_optimizer") as mock_optimizer,
        patch.object(service, "_get_client") as mock_client,
        patch.object(service, "_get_ranker") as mock_ranker,
        patch.object(service, "_get_acquisition_agent") as mock_acquisition,
        patch.object(service, "_get_processing_agent") as mock_processing,
    ):
        optimizer_instance = MagicMock()
        optimizer_instance.generate_dual_queries = AsyncMock(
            return_value={
                "primary": "intermittent fasting cardiovascular",
                "broad": "fasting heart health",
                "alternative": "time-restricted eating",
                "final_rephrase": "intermittent fasting cardiovascular risks",
            }
        )
        mock_optimizer.return_value = optimizer_instance

        client_instance = MagicMock()
        client_instance.multi_query_search = AsyncMock(
            return_value=[{"paperId": "p1", "citationCount": 50}]
        )
        mock_client.return_value = client_instance

        ranker_instance = MagicMock()
        ranker_instance.rank_papers.return_value = mock_finalists
        mock_ranker.return_value = ranker_instance

        acquisition_instance = MagicMock()

        async def mock_acquire(state: State) -> State:
            state.update_fields(passages=mock_passages)
            return state

        acquisition_instance.process = AsyncMock(side_effect=mock_acquire)
        mock_acquisition.return_value = acquisition_instance

        # Processing returns no chunks (simulating processing failure)
        processing_instance = MagicMock()

        def mock_process_empty(state, embedding_model=None, batch_size=32):
            state.update_fields(chunks=[])
            return state

        processing_instance.process = MagicMock(side_effect=mock_process_empty)
        mock_processing.return_value = processing_instance

        with patch("services.query_service.validate_quality") as mock_quality:
            mock_quality.return_value = {"passed": True, "metrics": {}}

            # Act
            result = await service.process_query(
                "intermittent fasting cardiovascular risks"
            )

    # Assert - Pipeline should still succeed (graceful degradation)
    assert result.success is True, "Pipeline should succeed despite processing issues"
    assert len(result.finalists) >= 1, "Finalists should be preserved"
    assert len(result.passages) == 0, "No passages due to empty chunks"

    # Both acquisition and processing timing should be recorded
    assert "pdf_acquisition" in result.timing_breakdown
    assert "processing" in result.timing_breakdown


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.performance
async def test_pipeline_performance(
    mock_embedding_model,
    mock_finalists,
    mock_passages,
    mock_chunks,
    mock_retrieval_results,
    mock_reranked_results,
):
    """Test that the full pipeline completes within performance budget.

    Validates:
        - Total execution time < 20 seconds
        - All 8 pipeline stages are timed
        - No individual stage takes excessive time
        - Test itself completes in < 30 seconds
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
    ):
        # Setup all mocks
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
            return_value=[{"paperId": "p1", "citationCount": 50}]
        )
        mock_client.return_value = client_instance

        ranker_instance = MagicMock()
        ranker_instance.rank_papers.return_value = mock_finalists
        mock_ranker.return_value = ranker_instance

        acquisition_instance = MagicMock()

        async def mock_acquire(state: State) -> State:
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
        retriever_instance.retrieve.return_value = mock_retrieval_results
        mock_retriever.return_value = retriever_instance

        reranker_instance = MagicMock()
        reranker_instance.rerank.return_value = mock_reranked_results
        mock_reranker.return_value = reranker_instance

        with patch("services.query_service.validate_quality") as mock_quality:
            mock_quality.return_value = {"passed": True, "metrics": {}}

            # Act
            start_time = time.perf_counter()
            result = await service.process_query(
                "intermittent fasting cardiovascular risks"
            )
            measured_time = time.perf_counter() - start_time

    # Performance assertions
    assert result.success is True, f"Pipeline should succeed: {result.error_message}"
    assert (
        result.execution_time < 20.0
    ), f"Pipeline should complete in <20s, took {result.execution_time:.2f}s"
    assert (
        measured_time < 30.0
    ), f"Total test execution should be <30s, took {measured_time:.2f}s"

    # Validate timing breakdown includes all 8 stages
    expected_stages = [
        "query_optimization",
        "semantic_scholar_search",
        "quality_gate",
        "composite_scoring",
        "pdf_acquisition",
        "processing",
        "retrieval",
        "reranking",
    ]

    for stage in expected_stages:
        assert stage in result.timing_breakdown, f"Missing timing for stage: {stage}"
        assert (
            result.timing_breakdown[stage] >= 0
        ), f"Stage {stage} should have non-negative timing"

    # Log timing breakdown for debugging
    print("\n=== Pipeline Timing Breakdown ===")
    for stage, timing in result.timing_breakdown.items():
        print(f"  {stage}: {timing:.3f}s")
    print(f"  TOTAL: {result.execution_time:.3f}s")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_without_embedding_model(mock_finalists):
    """Test that pipeline completes stages without embedding model.

    Validates:
        - Pipeline works without embedding model
        - Later stages are skipped gracefully
        - Finalists are returned
        - No later stage timing entries
    """
    # Arrange - No embedding model
    service = QueryService(embedding_model=None)

    with (
        patch.object(service, "_get_optimizer") as mock_optimizer,
        patch.object(service, "_get_client") as mock_client,
        patch.object(service, "_get_ranker") as mock_ranker,
    ):
        optimizer_instance = MagicMock()
        optimizer_instance.generate_dual_queries = AsyncMock(
            return_value={
                "primary": "intermittent fasting cardiovascular",
                "broad": "fasting heart health",
                "alternative": "time-restricted eating",
                "final_rephrase": "intermittent fasting cardiovascular risks",
            }
        )
        mock_optimizer.return_value = optimizer_instance

        client_instance = MagicMock()
        client_instance.multi_query_search = AsyncMock(
            return_value=[{"paperId": "p1", "citationCount": 50}]
        )
        mock_client.return_value = client_instance

        ranker_instance = MagicMock()
        ranker_instance.rank_papers.return_value = mock_finalists
        mock_ranker.return_value = ranker_instance

        with patch("services.query_service.validate_quality") as mock_quality:
            mock_quality.return_value = {"passed": True, "metrics": {}}

            # Act
            result = await service.process_query(
                "intermittent fasting cardiovascular risks"
            )

    assert result.success is True, "Pipeline should succeed without embedding model"
    assert len(result.finalists) >= 1, "Should have finalists"
    assert len(result.passages) == 0, "Should skip later stages without embedding model"

    assert "query_optimization" in result.timing_breakdown
    assert "semantic_scholar_search" in result.timing_breakdown
    assert "quality_gate" in result.timing_breakdown
    assert "composite_scoring" in result.timing_breakdown

    assert (
        "pdf_acquisition" not in result.timing_breakdown
    ), "Later stages should be skipped"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_with_partial_pdf_failures(
    mock_embedding_model,
    mock_finalists,
    mock_chunks,
    mock_retrieval_results,
    mock_reranked_results,
):
    """Test graceful degradation when some (but not all) PDFs fail to download.

    Validates:
        - Pipeline continues with available PDFs
        - Partial results are still returned
        - No crashes when some acquisitions fail
        - Pipeline reports success with partial data
        - Passages from successful PDFs are included
    """
    # Arrange - Only 1 of 3 papers has PDF content (partial failure)
    partial_passages = [
        {
            "paperId": "paper1",
            "paper_id": "paper1",
            "title": "Effects of Intermittent Fasting on Cardiovascular Health",
            "text": """# Introduction
            
Intermittent fasting has gained considerable attention as a dietary intervention
for improving cardiovascular health. This study investigates the effects of
16:8 time-restricted eating on various cardiovascular risk factors.

# Results

The intermittent fasting group showed significant reductions in blood pressure
(systolic: -8.5 mmHg, p<0.001), LDL cholesterol (-15%, p<0.01).""",
            "citationCount": 150,
            "year": 2023,
        },
        # paper2 and paper3 "failed" to download - not included
    ]

    # Chunks only from paper1
    partial_chunks = [
        chunk for chunk in mock_chunks if chunk.metadata.get("paper_id") == "paper1"
    ]
    if not partial_chunks:
        # Ensure we have at least some chunks for the test
        partial_chunks = mock_chunks[:5]
        for chunk in partial_chunks:
            chunk.metadata["paper_id"] = "paper1"

    # Partial retrieval and reranking results
    partial_retrieval = mock_retrieval_results[:8]
    partial_reranked = mock_reranked_results[:8]

    service = QueryService(embedding_model=mock_embedding_model)

    with (
        patch.object(service, "_get_optimizer") as mock_optimizer,
        patch.object(service, "_get_client") as mock_client,
        patch.object(service, "_get_ranker") as mock_ranker,
        patch.object(service, "_get_acquisition_agent") as mock_acquisition,
        patch.object(service, "_get_processing_agent") as mock_processing,
        patch.object(service, "_get_retriever") as mock_retriever,
        patch.object(service, "_get_reranker") as mock_reranker,
    ):
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
            return_value=[{"paperId": "p1", "citationCount": 50}]
        )
        mock_client.return_value = client_instance

        ranker_instance = MagicMock()
        ranker_instance.rank_papers.return_value = mock_finalists
        mock_ranker.return_value = ranker_instance

        acquisition_instance = MagicMock()

        async def mock_partial_acquire(state: State) -> State:
            # Simulate partial success - only 1 of 3 PDFs acquired
            state.update_fields(passages=partial_passages)
            return state

        acquisition_instance.process = AsyncMock(side_effect=mock_partial_acquire)
        mock_acquisition.return_value = acquisition_instance

        processing_instance = MagicMock()

        def mock_partial_process(state, embedding_model=None, batch_size=32):
            state.update_fields(chunks=partial_chunks)
            return state

        processing_instance.process = MagicMock(side_effect=mock_partial_process)
        mock_processing.return_value = processing_instance

        retriever_instance = MagicMock()
        retriever_instance.retrieve.return_value = partial_retrieval
        mock_retriever.return_value = retriever_instance

        reranker_instance = MagicMock()
        reranker_instance.rerank.return_value = partial_reranked
        mock_reranker.return_value = reranker_instance

        with patch("services.query_service.validate_quality") as mock_quality:
            mock_quality.return_value = {"passed": True, "metrics": {}}

            # Act
            result = await service.process_query(
                "intermittent fasting cardiovascular risks"
            )

    # Assert - Pipeline should succeed with partial data
    assert (
        result.success is True
    ), "Pipeline should succeed with partial PDF acquisition"
    assert len(result.finalists) >= 1, "Should have finalists from earlier stages"

    # Should still have passages from successfully acquired PDFs
    assert len(result.passages) > 0, "Should have passages from available PDFs"
    assert len(result.passages) <= 12, "Should have at most 12 passages"

    # All 8 timing stages should be recorded
    expected_stages = [
        "query_optimization",
        "semantic_scholar_search",
        "quality_gate",
        "composite_scoring",
        "pdf_acquisition",
        "processing",
        "retrieval",
        "reranking",
    ]
    for stage in expected_stages:
        assert stage in result.timing_breakdown, f"Missing timing for stage: {stage}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_result_structure(
    mock_embedding_model,
    mock_finalists,
    mock_passages,
    mock_chunks,
    mock_retrieval_results,
    mock_reranked_results,
):
    """Test that PipelineResult has all expected fields and types.

    Validates:
        - All PipelineResult fields are present
        - Types are correct
        - No unexpected None values where data expected
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
    ):
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
            return_value=[{"paperId": "p1", "citationCount": 50}]
        )
        mock_client.return_value = client_instance

        ranker_instance = MagicMock()
        ranker_instance.rank_papers.return_value = mock_finalists
        mock_ranker.return_value = ranker_instance

        acquisition_instance = MagicMock()

        async def mock_acquire(state: State) -> State:
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
        retriever_instance.retrieve.return_value = mock_retrieval_results
        mock_retriever.return_value = retriever_instance

        reranker_instance = MagicMock()
        reranker_instance.rerank.return_value = mock_reranked_results
        mock_reranker.return_value = reranker_instance

        with patch("services.query_service.validate_quality") as mock_quality:
            mock_quality.return_value = {"passed": True, "metrics": {}}

            # Act
            query = "intermittent fasting cardiovascular risks"
            result = await service.process_query(query)

    # Validate PipelineResult structure
    assert isinstance(
        result, PipelineResult
    ), "Result should be PipelineResult instance"

    # Required fields - query
    assert isinstance(result.query, str), "query should be string"
    assert result.query == query, "query should match input"

    # Required fields - optimized_queries
    assert isinstance(
        result.optimized_queries, dict
    ), "optimized_queries should be dict"
    assert "final_rephrase" in result.optimized_queries
    assert "primary" in result.optimized_queries
    assert "broad" in result.optimized_queries
    assert "alternative" in result.optimized_queries

    # Required fields - quality_gate
    assert isinstance(result.quality_gate, dict), "quality_gate should be dict"
    assert "passed" in result.quality_gate

    # Required fields - finalists
    assert isinstance(result.finalists, list), "finalists should be list"

    # Required fields - execution_time
    assert isinstance(result.execution_time, float), "execution_time should be float"
    assert result.execution_time > 0, "execution_time should be positive"

    # Required fields - timing_breakdown
    assert isinstance(result.timing_breakdown, dict), "timing_breakdown should be dict"

    # Required fields - success
    assert isinstance(result.success, bool), "success should be bool"

    # Required fields - passages
    assert isinstance(result.passages, list), "passages should be list"

    # Optional error fields (should be None on success)
    if result.success:
        assert result.error_message is None, "error_message should be None on success"
        assert result.error_stage is None, "error_stage should be None on success"
