import pytest

pytestmark = [pytest.mark.unit]
from pydantic import ValidationError
from models.state import State, Paper, Passage, Score


def test_state_creation():
    """Tests basic State object creation with new optimized fields."""
    state = State(query="test query")
    assert state.original_query == "test query"
    assert state.query == "test query"  # Test alias backward compatibility
    assert isinstance(state.request_id, str)
    assert state.papers_metadata == []
    assert state.filtered_papers == []
    assert state.optimized_query is None
    assert state.filtering_scores is None
    assert state.dedup_stats is None
    assert state.search_quality is None
    assert state.refinement_count == 0


def test_state_creation_with_original_query():
    """Tests State creation using original_query field directly."""
    state = State(original_query="direct query test")
    assert state.original_query == "direct query test"
    assert state.query == "direct query test"  # Alias should work both ways


def test_state_with_papers():
    """Tests State object creation with paper metadata."""
    papers = [
        Paper(
            paper_id="1",
            title="Paper 1",
            authors=["Author 1"],
            year=2023,
            journal="N/A",
            citation_count=0,
            pdf_url="http://example.com/1.pdf",
            type="article",
        ),
        Paper(
            paper_id="2",
            title="Paper 2",
            authors=["Author 2"],
            year=2023,
            journal="N/A",
            citation_count=0,
            pdf_url="http://example.com/2.pdf",
            type="review",
        ),
    ]
    state = State(query="test", papers_metadata=papers)
    assert len(state.papers_metadata) == 2
    assert state.papers_metadata[0].pdf_url is not None


def test_state_validator_no_pdf_urls():
    """Tests the validator when no papers have PDF URLs."""
    papers = [
        Paper(
            paper_id="1",
            title="Paper 1",
            authors=["Author 1"],
            year=2023,
            journal="N/A",
            citation_count=0,
            pdf_url=None,
            type="article",
        )
    ]
    state = State(query="test", papers_metadata=papers)
    assert "No papers with PDF URLs found." in state.validation_errors


def test_state_validator_empty_papers_list():
    """Tests the validator with an empty list of papers."""
    state = State(query="test", papers_metadata=[])
    assert state.papers_metadata == []


def test_state_serialization():
    """Tests that the State object can be serialized to a dict."""
    papers = [
        Paper(
            paper_id="1",
            title="Paper 1",
            authors=["Author 1"],
            year=2023,
            journal="N/A",
            citation_count=0,
            pdf_url="http://example.com/1.pdf",
            type="article",
        )
    ]
    state = State(query="test", papers_metadata=papers)
    state_dict = state.model_dump()
    # Default serialization uses field names, not aliases
    assert state_dict["original_query"] == "test"
    assert "query" not in state_dict  # Alias not used by default
    assert len(state_dict["papers_metadata"]) == 1
    assert state_dict["papers_metadata"][0]["pdf_url"] == "http://example.com/1.pdf"


def test_state_serialization_by_alias():
    """Tests serialization with by_alias parameter."""
    state = State(query="test")
    # Test serialization with alias (legacy format)
    state_dict = state.model_dump(by_alias=True)
    assert "query" in state_dict
    assert "original_query" not in state_dict

    # Test serialization without alias (new format)
    state_dict = state.model_dump(by_alias=False)
    assert "original_query" in state_dict
    assert "query" not in state_dict


def test_optimized_query_validation():
    """Test optimized_query field validation enforces length constraints."""
    state = State(query="test")

    # Valid optimized query
    state.optimized_query = "CRISPR-Cas9 off-target effects in mammalian cells"
    assert state.optimized_query is not None

    # Invalid: too short
    with pytest.raises(ValueError, match="20-200 characters"):
        state.optimized_query = "short"

    # Invalid: too long
    with pytest.raises(ValueError, match="20-200 characters"):
        state.optimized_query = "x" * 201


def test_query_length_validation():
    """Test query length validation for both original and optimized queries."""
    # Valid query
    state = State(query="This is a valid query with sufficient length")
    assert state.original_query is not None

    # Invalid: too short
    with pytest.raises(ValueError, match="at least 3 characters"):
        State(query="hi")

    # Invalid: too long
    with pytest.raises(ValueError, match="less than 256 characters"):
        State(query="x" * 257)

    # Invalid: just whitespace - this will fail length validation first (3 chars required)
    with pytest.raises(ValueError, match="at least 3 characters"):
        State(query="   ")


def test_filtered_papers_count_validation():
    """Test filtered_papers cannot exceed 25 papers."""
    state = State(query="test")

    # Valid: exactly 25 papers
    state.filtered_papers = [
        Paper(
            paper_id=str(i),
            title=f"Paper {i}",
            authors=[f"Author {i}"],
            year=2023,
            journal="N/A",
            citation_count=0,
            pdf_url=f"https://example.com/{i}.pdf",
            type="article",
        )
        for i in range(25)
    ]
    assert len(state.filtered_papers) == 25

    # Invalid: more than 25
    with pytest.raises(ValueError, match="≤25"):
        state.filtered_papers = [
            Paper(
                paper_id=str(i),
                title=f"Paper {i}",
                authors=[f"Author {i}"],
                year=2023,
                journal="N/A",
                citation_count=0,
                pdf_url=f"https://example.com/{i}.pdf",
                type="article",
            )
            for i in range(26)
        ]


def test_search_quality_structure():
    """Test search_quality dict has correct structure."""
    state = State(
        query="test",
        search_quality={
            "passed": True,
            "reason": "Found 78 papers with avg_relevance 0.65",
            "paper_count": 78,
            "avg_relevance": 0.65,
        },
    )

    assert state.search_quality["passed"] is True
    assert state.search_quality["paper_count"] == 78
    assert 0 <= state.search_quality["avg_relevance"] <= 1

    # Test validation failures
    with pytest.raises(
        ValueError, match="must contain: passed, reason, paper_count, avg_relevance"
    ):
        State(query="test", search_quality={"incomplete": "dict"})

    with pytest.raises(ValueError, match="must be boolean"):
        State(
            query="test",
            search_quality={
                "passed": "not_boolean",
                "reason": "test",
                "paper_count": 1,
                "avg_relevance": 0.5,
            },
        )

    with pytest.raises(ValueError, match="between 0 and 1"):
        State(
            query="test",
            search_quality={
                "passed": True,
                "reason": "test",
                "paper_count": 1,
                "avg_relevance": 1.5,
            },
        )


def test_dedup_stats_structure():
    """Test dedup_stats tracks deduplication correctly."""
    state = State(
        query="test",
        dedup_stats={"raw_count": 200, "unique_count": 145, "duplicates_removed": 55},
    )

    assert state.dedup_stats["raw_count"] == 200
    assert state.dedup_stats["unique_count"] == 145
    assert state.dedup_stats["duplicates_removed"] == 55

    # Test validation failures - missing required keys
    with pytest.raises(
        ValueError, match="must contain: raw_count, unique_count, duplicates_removed"
    ):
        State(
            query="test", dedup_stats={"raw_count": 100}
        )  # Missing other required keys

    with pytest.raises(ValueError, match="cannot be greater than raw_count"):
        State(
            query="test",
            dedup_stats={
                "raw_count": 100,
                "unique_count": 150,
                "duplicates_removed": 0,
            },
        )


def test_filtering_scores_structure():
    """Test filtering_scores has expected structure."""
    state = State(
        query="test",
        filtering_scores={
            "bm25_top_50": ["paper1", "paper2"],
            "semantic_top_25": ["paper1", "paper3"],
            "avg_bm25_score": 8.2,
            "avg_semantic_score": 0.78,
        },
    )

    assert "bm25_top_50" in state.filtering_scores
    assert "avg_bm25_score" in state.filtering_scores

    # Test validation - should pass with at least one expected key
    state2 = State(query="test", filtering_scores={"avg_bm25_score": 7.5})
    assert state2.filtering_scores is not None

    # Test validation failure
    with pytest.raises(ValueError, match="must contain at least one of"):
        State(query="test", filtering_scores={"unknown_key": "value"})


def test_passage_and_score_models():
    """Test the new Passage and Score models work correctly."""
    passage = Passage(
        content="This is a test passage.",
        section="introduction",
        paper_id="p1",
        retrieval_score=0.85,
        cross_encoder_score=0.78,
        final_score=0.81,
    )

    assert passage.content == "This is a test passage."
    assert passage.section == "introduction"
    assert passage.paper_id == "p1"
    assert passage.retrieval_score == 0.85

    score = Score(
        score_type="relevance",
        value=0.95,
        justification="Highly relevant to query",
        confidence=0.9,
    )

    assert score.score_type == "relevance"
    assert score.value == 0.95
    assert score.justification == "Highly relevant to query"


def test_backward_compatibility():
    """Test old State usage patterns still work."""
    # Simulate old code that uses 'query' field
    state = State(query="What is intermittent fasting?")

    # Both alias and field should work
    assert state.query == state.original_query
    assert state.original_query == "What is intermittent fasting?"

    # Test that old serialization still works
    state_dict = state.model_dump(by_alias=True)
    assert state_dict["query"] == "What is intermittent fasting?"
    assert "original_query" not in state_dict


def test_state_flows_through_optimized_pipeline():
    """Test State correctly tracks data through new optimized flow."""
    state = State(query="fasting benefits")

    # After QueryOptimizationAgent
    state.optimized_query = "metabolic effects of intermittent fasting protocols"
    assert state.optimized_query is not None

    # After SearchAgent (100 papers)
    state.papers_metadata = [
        Paper(
            paper_id=str(i),
            title=f"Paper {i}",
            authors=[f"Author {i}"],
            year=2023,
            journal="N/A",
            citation_count=0,
            pdf_url=f"https://example.com/{i}.pdf",
            type="article",
        )
        for i in range(100)
    ]
    state.search_quality = {
        "passed": True,
        "reason": "Found 100 papers",
        "paper_count": 100,
        "avg_relevance": 0.72,
    }

    # After DeduplicationAgent
    state.dedup_stats = {"raw_count": 100, "unique_count": 95, "duplicates_removed": 5}

    # After FilteringAgent
    state.filtered_papers = state.papers_metadata[:25]
    state.filtering_scores = {"avg_bm25_score": 8.2, "avg_semantic_score": 0.78}

    assert len(state.filtered_papers) == 25
    assert state.dedup_stats["duplicates_removed"] == 5
    assert state.search_quality["paper_count"] == 100


def test_enhanced_processing_models():
    """Test case 1: Model validation with valid and invalid input data."""
    from models.state import (
        ProcessingStatus,
        RetrievalScores,
        Chunk,
        Passage,
        ProcessingMetadata,
        ProcessingConfig,
        State,
    )
    import pytest

    # Test ProcessingStatus enum
    assert ProcessingStatus.PENDING.value == "pending"
    assert ProcessingStatus.IN_PROGRESS.value == "in_progress"
    assert ProcessingStatus.COMPLETED.value == "completed"
    assert ProcessingStatus.FAILED.value == "failed"
    assert ProcessingStatus.PARTIAL.value == "partial"

    # Test RetrievalScores validation
    valid_scores = RetrievalScores(
        bm25_score=0.8,
        semantic_score=0.7,
        rrf_score=0.9,
        reranking_score=0.85,
        final_rank=5,
    )
    assert valid_scores.bm25_score == 0.8
    assert valid_scores.final_rank == 5

    # Invalid scores should raise ValueError
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        RetrievalScores(bm25_score=1.5)

    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        RetrievalScores(semantic_score=-0.1)

    # Test Chunk model
    valid_chunk = Chunk(
        chunk_id="chunk_123",
        paper_id="paper_456",
        text="This is sample chunk text.",
        chunk_index=0,
        section="INTRODUCTION",
        token_count=100,
        char_start=0,
        char_end=24,
    )
    assert valid_chunk.chunk_id == "chunk_123"
    assert valid_chunk.section == "INTRODUCTION"

    # Invalid chunk (char_end < char_start)
    with pytest.raises(ValueError, match="char_end must be >= char_start"):
        Chunk(
            chunk_id="chunk_123",
            paper_id="paper_456",
            text="Sample text",
            chunk_index=0,
            char_start=10,
            char_end=5,
        )

    # Test ProcessingConfig validation
    valid_config = ProcessingConfig(
        chunk_size=600, chunk_overlap=100, enable_reranking=True, batch_size=16
    )
    assert valid_config.chunk_size == 600
    assert valid_config.batch_size == 16

    # Invalid config should raise ValueError
    with pytest.raises(ValueError):  # chunk_size too large
        ProcessingConfig(chunk_size=1200)

    with pytest.raises(ValueError):  # chunk_overlap exceeds limit
        ProcessingConfig(chunk_overlap=250)

    with pytest.raises(ValueError):  # batch_size too small
        ProcessingConfig(batch_size=0)


def test_processing_state_transitions():
    """Test case 2: State transitions and status updates work correctly."""
    from models.state import State, ProcessingStatus, ProcessingMetadata, Chunk, Passage
    import time

    # Create state with processing metadata
    state = State(original_query="test query")
    metadata = ProcessingMetadata(
        total_papers=2, processed_papers=2, total_chunks=50, chunks_embedded=50
    )
    state.processing_metadata = metadata

    # Test initial status
    assert state.processing_status == ProcessingStatus.PENDING

    # Test state transitions
    state.start_processing()
    assert state.processing_status == ProcessingStatus.IN_PROGRESS

    # Cannot start processing again
    state.start_processing()  # Should not change status
    assert state.processing_status == ProcessingStatus.IN_PROGRESS

    state.complete_processing()
    assert state.processing_status == ProcessingStatus.COMPLETED

    # Test failure handling
    failed_state = State(original_query="failed query")
    failed_state.start_processing()
    failed_state.fail_processing("Test error")
    assert failed_state.processing_status == ProcessingStatus.FAILED

    # Test partial completion
    partial_state = State(original_query="partial query")
    partial_state.start_processing()
    partial_state.mark_partial()
    assert partial_state.processing_status == ProcessingStatus.PARTIAL

    partial_state.complete_processing()
    assert partial_state.processing_status == ProcessingStatus.COMPLETED


def test_processing_serialization_deserialization():
    """Test case 3: Serialization/deserialization preserves all data."""
    from models.state import State, ProcessingMetadata, Chunk, Passage, RetrievalScores
    import json

    # Create comprehensive state with all enhanced fields
    state = State(
        original_query="What are the effects of fasting on cardiovascular health?",
        processing_status="in_progress",
    )

    # Add processing metadata
    metadata = ProcessingMetadata(
        total_papers=2,
        processed_papers=2,
        total_chunks=45,
        chunks_embedded=45,
        retrieval_candidates=20,
        reranked_passages=10,
        stage_times={"embedding": 2.1, "retrieval": 0.8, "reranking": 0.3},
        total_time=3.2,
        memory_usage_mb=120.5,
        throughput_chunks_per_second=21.4,
    )
    state.processing_metadata = metadata

    # Add chunks
    chunks = []
    for i in range(50):
        chunk = Chunk(
            chunk_id=f"chunk_{i:03d}",
            paper_id=f"paper_{(i % 2) + 1}",
            text=f"This is chunk number {i} with some content for testing purposes.",
            chunk_index=i,
            section="INTRODUCTION" if i < 25 else "METHODS",
            token_count=50 + (i % 100),
            char_start=i * 200,
            char_end=(i + 1) * 200,
        )
        chunks.append(chunk)
    state.chunks = chunks

    # Add relevant passages
    passages = []
    for i in range(10):
        passage = Passage(
            content=f"Passage content {i} from cardiovascular research study.",
            section="introduction",
            paper_id="paper_1",
            chunk_id=f"chunk_{i:03d}",
            retrieval_score=0.8 - (i * 0.02),
            final_score=0.9 - (i * 0.05),
            char_start=i * 50,
            char_end=(i + 1) * 50,
        )
        # Set scores
        passage.scores.bm25_score = 0.8 - (i * 0.02)
        passage.scores.semantic_score = 0.7 - (i * 0.03)
        passage.scores.rrf_score = 0.75 - (i * 0.025)
        passage.scores.reranking_score = 0.85 - (i * 0.04)
        passage.scores.final_rank = i + 1

        passages.append(passage)
    state.relevant_passages = passages

    # Serialize to JSON
    state_dict = state.model_dump()
    json_str = json.dumps(state_dict, default=str)
    assert json_str

    # Deserialize back
    loaded_dict = json.loads(json_str)
    reconstructed_state = State.model_validate(loaded_dict)

    # Verify all data preserved
    assert reconstructed_state.original_query == state.original_query
    assert reconstructed_state.processing_status == state.processing_status
    assert reconstructed_state.config.chunk_size == state.config.chunk_size

    # Verify processing metadata
    assert reconstructed_state.processing_metadata.total_papers == 2
    assert reconstructed_state.processing_metadata.memory_usage_mb == 120.5
    assert reconstructed_state.processing_metadata.throughput_chunks_per_second == 21.4

    # Verify chunks
    assert len(reconstructed_state.chunks) == 50
    for i, chunk in enumerate(reconstructed_state.chunks):
        assert chunk.chunk_id == f"chunk_{i:03d}"
        assert chunk.chunk_index == i
        assert chunk.char_start == i * 200
        assert chunk.char_end == (i + 1) * 200

    # Verify passages
    assert len(reconstructed_state.relevant_passages) == 10
    for i, passage in enumerate(reconstructed_state.relevant_passages):
        assert f"Passage content {i}" in passage.content
        assert passage.scores.final_rank == i + 1
        assert abs(passage.scores.bm25_score - (0.8 - (i * 0.02))) < 1e-6


def test_enhanced_validator_error_handling():
    """Test case 4: Custom validators catch configuration errors."""
    from models.state import ProcessingConfig, RetrievalScores, Chunk, Passage
    import pytest

    # ProcessingConfig validator tests
    with pytest.raises(ValueError, match="chunk_size"):  # Below minimum
        ProcessingConfig(chunk_size=50)

    with pytest.raises(ValueError, match="chunk_size"):  # Above maximum
        ProcessingConfig(chunk_size=1100)

    with pytest.raises(ValueError, match="chunk_overlap"):  # Below minimum
        ProcessingConfig(chunk_overlap=-10)

    with pytest.raises(ValueError, match="chunk_overlap"):  # Above maximum
        ProcessingConfig(chunk_overlap=250)

    with pytest.raises(ValueError, match="top_k"):  # Below minimum
        ProcessingConfig(top_k=3)

    with pytest.raises(ValueError, match="final_k"):  # Above maximum
        ProcessingConfig(final_k=25)

    with pytest.raises(ValueError, match="batch_size"):  # Below minimum
        ProcessingConfig(batch_size=0)

    with pytest.raises(ValueError, match="max_chunk_tokens"):  # Below minimum
        ProcessingConfig(max_chunk_tokens=50)

    # RetrievalScores validator tests
    with pytest.raises(
        ValueError, match="must be between 0.0 and 1.0"
    ):  # bm25_score too high
        RetrievalScores(bm25_score=1.1)

    with pytest.raises(
        ValueError, match="must be between 0.0 and 1.0"
    ):  # Negative score
        RetrievalScores(semantic_score=-0.5)

    with pytest.raises(
        ValueError, match="must be between 0.0 and 1.0"
    ):  # rrf_score too high
        RetrievalScores(rrf_score=1.5)

    with pytest.raises(
        ValueError, match="must be between 0.0 and 1.0"
    ):  # reranking_score too low
        RetrievalScores(reranking_score=-0.1)

    # Chunk validator tests
    with pytest.raises(ValueError, match="greater_than_equal"):  # Negative chunk_index
        Chunk(chunk_id="chunk_1", paper_id="paper_1", text="test", chunk_index=-1)

    with pytest.raises(ValueError, match="greater_than_equal"):  # Negative token_count
        Chunk(
            chunk_id="chunk_1",
            paper_id="paper_1",
            text="test",
            chunk_index=0,
            token_count=-5,
        )

    with pytest.raises(ValueError, match="greater_than_equal"):  # Negative char_start
        Chunk(
            chunk_id="chunk_1",
            paper_id="paper_1",
            text="test",
            chunk_index=0,
            char_start=-10,
        )

        # Passage validator tests
        with pytest.raises(
            ValueError, match="should be less than or equal to 1"
        ):  # retrieval_score too high
            Passage(
                content="test content",
                section="introduction",
                paper_id="paper_1",
                retrieval_score=1.5,
            )

    with pytest.raises(
        ValueError, match="char_end must be >= char_start"
    ):  # Invalid char positions
        Passage(
            content="test",
            section="introduction",
            paper_id="paper_1",
            retrieval_score=0.8,
            char_start=20,
            char_end=10,
        )


def test_comprehensive_processing_integration():
    """Example Test Case: State with processing metadata, 50 chunks, 10 passages with scores."""
    from models.state import State, ProcessingMetadata, Chunk, Passage, RetrievalScores
    import time

    # Create the comprehensive test case specified in requirements
    state = State(
        original_query="cardiovascular effects of intermittent fasting",
        processing_status="completed",
    )

    # Set up processing config
    state.config.chunk_size = 512
    state.config.chunk_overlap = 64
    state.config.enable_reranking = True

    # Add processing metadata
    metadata = ProcessingMetadata(
        total_papers=3,
        processed_papers=3,
        total_sections=9,
        total_chunks=57,
        chunks_embedded=57,
        retrieval_candidates=42,
        reranked_passages=15,
        stage_times={
            "section_detection": 0.5,
            "chunking": 1.2,
            "embedding": 3.4,
            "index_build": 0.1,
            "retrieval": 0.8,
            "reranking": 0.3,
        },
        total_time=6.3,
        memory_usage_mb=142.8,
        throughput_chunks_per_second=16.7,
    )
    state.processing_metadata = metadata

    # Add 50+ chunks with realistic data
    chunks = []
    paper_ids = ["fasting_study_2023", "cardio_research_2024", "meta_analysis_2023"]

    for i in range(57):
        chunk = Chunk(
            chunk_id=f"chunk_{i:03d}",
            paper_id=paper_ids[i % 3],
            text=f"Intermittent fasting has been shown to improve cardiovascular health markers including reduced blood pressure and improved lipid profiles. Study data indicates significant improvements in cholesterol levels after 12 weeks of time-restricted eating protocols.",
            chunk_index=i,
            section="INTRODUCTION" if i < 20 else ("METHODS" if i < 40 else "RESULTS"),
            token_count=85 + (i % 20),
            char_start=i * 250,
            char_end=(i + 1) * 250,
        )
        chunks.append(chunk)
    state.chunks = chunks

    # Add 10 passages with comprehensive scoring
    passages = []
    for i in range(10):
        scores = RetrievalScores(
            bm25_score=min(1.0, 0.85 - (i * 0.02)),
            semantic_score=min(1.0, 0.75 - (i * 0.03)),
            rrf_score=min(1.0, 0.80 - (i * 0.025)),
            reranking_score=min(1.0, 0.90 - (i * 0.01)),
            final_rank=i + 1,
        )

        passage = Passage(
            content=f"Passage {i+1}: Intermittent fasting demonstrates significant cardiovascular benefits through multiple mechanisms including reduced inflammation, improved metabolic flexibility, and decreased oxidative stress. Clinical evidence supports the efficacy of time-restricted feeding protocols.",
            section="introduction",
            paper_id="fasting_study_2023",
            chunk_id=f"chunk_{i*5:03d}",
            scores=scores,
            retrieval_score=scores.bm25_score,
            cross_encoder_score=scores.reranking_score,
            final_score=scores.reranking_score,
            char_start=i * 120,
            char_end=(i + 1) * 120,
        )
        passages.append(passage)

    state.relevant_passages = passages

    # Validation tests
    assert state.processing_status.value == "completed"
    assert len(state.chunks) == 57
    assert len(state.relevant_passages) == 10

    # Test property access
    assert state.total_execution_time > 0
    assert state.total_execution_time == state.processing_metadata.total_time

    # Test helper methods
    summary = state.get_processing_summary()
    assert summary["status"] == "completed"
    assert summary["chunks_created"] == 57
    assert summary["passages_retrieved"] == 10
    assert "INTRODUCTION" in summary["section_distribution"]

    metrics = state.get_performance_metrics()
    assert metrics["status"] == "completed"
    assert "stage_times" in metrics
    assert "memory_usage_mb" in metrics

    # Verify scoring integration
    top_passage = state.relevant_passages[0]
    assert top_passage.bm25_score == 0.85
    assert top_passage.semantic_score == 0.75
    assert top_passage.rrf_score == 0.80
    assert top_passage.reranking_score == 0.90
    assert top_passage.scores.final_rank == 1

    # Test serialization roundtrip (brief test)
    serialized = state.model_dump()
    reconstructed = State.model_validate(serialized)

    assert reconstructed.processing_status == state.processing_status
    assert len(reconstructed.chunks) == 57
    assert len(reconstructed.relevant_passages) == 10
    assert reconstructed.total_execution_time == state.total_execution_time
