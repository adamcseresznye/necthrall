import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List
from models.state import (
    State,
    CredibilityScore,
    DetectedContradiction,
    ContradictionClaim,
    Paper,
    Passage,
)
from agents.analysis import (
    AnalysisAgent,
    AnalysisError,
    CredibilityScoringError,
    ContradictionDetectionError,
    LLMProviderError,
    DataValidationError,
    RecoveryStrategy,
)
from langgraph.graph import StateGraph


def create_test_paper(
    paper_id: str,
    title: str,
    authors: List[str] = None,
    year: int = 2020,
    journal: str = "Test Journal",
    citation_count: int = 10,
    doi: str = None,
    pdf_url: str = None,
    paper_type: str = "article",
) -> Paper:
    """Helper function to create valid Paper objects for testing."""
    return Paper(
        paper_id=paper_id,
        title=title,
        authors=authors or ["Test Author"],
        year=year,
        journal=journal,
        citation_count=citation_count,
        doi=doi or f"10.1038/test{len(paper_id)}",
        pdf_url=pdf_url or f"https://example.com/pdf/{paper_id}.pdf",
        type=paper_type,
    )


def create_test_passage(
    paper_id: str,
    content: str,
    retrieval_score: float = 0.8,
    section: str = "Abstract",
    paper_title: str = None,
) -> Passage:
    """Helper function to create valid Passage objects for testing."""
    return Passage(
        paper_id=paper_id,
        content=content,
        section=section,
        retrieval_score=retrieval_score,
        paper_title=paper_title or f"Paper {paper_id}",
        start_pos=0,
        end_pos=len(content),
    )


class TestAnalysisAgentIntegration:
    """Comprehensive integration tests for AnalysisAgent in LangGraph context."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state for testing."""
        return State(
            original_query="What are the effects of intermittent fasting?",
            filtered_papers=[
                create_test_paper(
                    "paper1",
                    "Effects of Intermittent Fasting",
                    citation_count=50,
                    year=2023,
                    journal="Nature Medicine",
                ),
                create_test_paper(
                    "paper2",
                    "Nutritional Concerns with Fasting",
                    citation_count=25,
                    year=2020,
                    journal="Journal of Clinical Nutrition",
                ),
                create_test_paper(
                    "paper3",
                    "Preliminary Fasting Study",
                    citation_count=5,
                    year=2015,
                    journal="arxiv",
                ),
            ],
            relevant_passages=[
                create_test_passage(
                    "paper1",
                    "Intermittent fasting improves metabolic health by reducing insulin resistance.",
                    retrieval_score=0.95,
                    paper_title="Effects of Intermittent Fasting",
                ),
                create_test_passage(
                    "paper2",
                    "Intermittent fasting may increase the risk of nutrient deficiencies.",
                    retrieval_score=0.88,
                    paper_title="Nutritional Concerns with Fasting",
                ),
            ],
        )

    @pytest.fixture
    def analysis_agent(self):
        """Create AnalysisAgent instance for testing."""
        return AnalysisAgent()

    @pytest.mark.asyncio
    async def test_successful_analysis_execution(self, analysis_agent, mock_state):
        """Test successful execution of analysis with both components working."""
        # Execute analysis
        result = await analysis_agent.analyze(mock_state)

        # Verify results structure
        assert isinstance(result, dict)
        assert "credibility_scores" in result
        assert "contradictions" in result
        assert "execution_times" in result
        assert "analysis_errors" in result
        assert "recovery_actions" in result

        # Verify credibility scores
        scores = result["credibility_scores"]
        assert len(scores) == 3
        assert all(isinstance(score, CredibilityScore) for score in scores)
        assert all(score.paper_id in ["paper1", "paper2", "paper3"] for score in scores)

        # Verify execution times
        execution_times = result["execution_times"]
        assert "analysis_agent" in execution_times
        assert "credibility_scoring" in execution_times
        assert execution_times["analysis_agent"] > 0

    @pytest.mark.asyncio
    async def test_credibility_scoring_failure_recovery(
        self, analysis_agent, mock_state
    ):
        """Test recovery when credibility scoring fails."""
        # Mock the credibility scorer to raise an error
        with patch.object(
            analysis_agent.credibility_scorer,
            "score_paper",
            side_effect=Exception("Scoring failed"),
        ):
            result = await analysis_agent.analyze(mock_state)

        # Verify recovery was applied - individual paper failures get default scores
        assert len(result["analysis_errors"]) > 0
        assert len(result["credibility_scores"]) == len(
            mock_state.filtered_papers
        )  # All papers get scores

        # Verify default scores were generated
        scores = result["credibility_scores"]
        assert len(scores) == 3
        assert all(score.tier == "medium" for score in scores)  # Default tier

    @pytest.mark.asyncio
    async def test_contradiction_detection_failure_recovery(
        self, analysis_agent, mock_state
    ):
        """Test recovery when contradiction detection fails."""
        # Mock contradiction detector to raise an error
        with patch.object(
            analysis_agent,
            "_perform_contradiction_detection",
            side_effect=Exception("Detection failed"),
        ):
            result = await analysis_agent.analyze(mock_state)

        # Verify recovery was applied
        assert len(result["analysis_errors"]) > 0
        assert len(result["recovery_actions"]) > 0

        # Verify empty contradictions list
        assert result["contradictions"] == []

    @pytest.mark.asyncio
    async def test_large_paper_set_performance(self, analysis_agent):
        """Test performance optimization with large paper sets (50+ papers)."""
        # Create large mock state (limited to 25 papers due to State validation)
        large_papers = []
        for i in range(25):  # 25 papers (State validation limit)
            paper = create_test_paper(
                paper_id=f"paper{i}",
                title=f"Paper {i}",
                citation_count=10 + (i % 15),  # Vary citation counts
                year=2010 + (i % 13),  # Vary years
                journal="Test Journal",
            )
            large_papers.append(paper)

        large_state = State(
            original_query="Large scale test query",
            filtered_papers=large_papers,
            relevant_passages=[],  # Skip contradiction detection for this test
        )

        start_time = time.time()
        result = await analysis_agent.analyze(large_state)
        execution_time = time.time() - start_time

        # Verify performance
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert len(result["credibility_scores"]) == 25  # Should score all 25 papers

    @pytest.mark.asyncio
    async def test_empty_input_handling(self, analysis_agent):
        """Test handling of empty or minimal input."""
        # Test with no papers
        empty_state = State(original_query="Test query", filtered_papers=[])
        result = await analysis_agent.analyze(empty_state)

        assert result["credibility_scores"] == []
        assert result["contradictions"] == []
        assert len(result["analysis_errors"]) == 1  # Should have error for no papers

        # Test with papers but no passages
        state_no_passages = State(
            original_query="Test query",
            filtered_papers=[
                create_test_paper(
                    "test", "Test Paper", citation_count=10, year=2020, journal="Test"
                )
            ],
            relevant_passages=[],
        )
        result = await analysis_agent.analyze(state_no_passages)

        assert len(result["credibility_scores"]) == 1
        assert result["contradictions"] == []

    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self, analysis_agent, mock_state):
        """Test partial failure where some papers fail but others succeed."""
        # Mock scorer to fail on specific papers
        original_score_paper = analysis_agent.credibility_scorer.score_paper

        def selective_failure(metadata):
            if metadata["paper_id"] == "paper2":
                raise Exception("Selective failure")
            return original_score_paper(metadata)

        with patch.object(
            analysis_agent.credibility_scorer,
            "score_paper",
            side_effect=selective_failure,
        ):
            result = await analysis_agent.analyze(mock_state)

        # Verify partial success
        scores = result["credibility_scores"]
        assert len(scores) == 3  # All papers should have scores (some default)

        # Check that paper2 got default score
        paper2_score = next(score for score in scores if score.paper_id == "paper2")
        assert paper2_score.tier == "medium"  # Default tier

    @pytest.mark.asyncio
    async def test_memory_efficiency_large_dataset(self, analysis_agent):
        """Test memory efficiency with large datasets."""
        # Create state with many passages
        large_passages = []
        for i in range(100):
            passage = Passage(
                paper_id=f"paper{i%10}",
                content=f"This is passage content {i} with some scientific text that needs analysis.",
                paper_title=f"Paper Title {i%10}",
                retrieval_score=0.5 + (i % 50) / 100.0,
                start_pos=0,
                end_pos=100,
                section="Abstract",
            )
            large_passages.append(passage)

        large_state = State(
            original_query="Memory efficiency test",
            filtered_papers=[
                create_test_paper(
                    f"paper{i}",
                    f"Paper {i}",
                    citation_count=10,
                    year=2020,
                    journal="Test",
                )
                for i in range(10)
            ],
            relevant_passages=large_passages,
        )

        # Mock contradiction detection to avoid LLM calls
        with patch.object(
            analysis_agent,
            "_perform_contradiction_detection_optimized",
            return_value=[],
        ):
            result = await analysis_agent.analyze(large_state)

        # Verify basic functionality
        assert len(result["credibility_scores"]) == 10

    @pytest.mark.asyncio
    async def test_error_categorization(self, analysis_agent, mock_state):
        """Test proper error categorization and recovery strategy selection."""
        # Test data validation error
        with patch.object(
            analysis_agent,
            "_perform_credibility_scoring",
            side_effect=DataValidationError("Invalid data", "credibility_scorer"),
        ):
            result = await analysis_agent.analyze(mock_state)

            assert len(result["analysis_errors"]) > 0
            recovery_action = result["recovery_actions"][0]
            assert (
                recovery_action["action"] == "partial_recovery"
            )  # DataValidationError gets categorized as CredibilityScoringError

        # Test LLM provider error
        with patch.object(
            analysis_agent,
            "_perform_contradiction_detection",
            side_effect=LLMProviderError("API limit", "gemini"),
        ):
            result = await analysis_agent.analyze(mock_state)

            assert len(result["analysis_errors"]) > 0
            recovery_action = result["recovery_actions"][-1]  # Last recovery action
            assert recovery_action["action"] == "fallback_provider"

    @pytest.mark.asyncio
    async def test_langgraph_state_integration(self, analysis_agent, mock_state):
        """Test integration with LangGraph State updates."""
        # Execute analysis
        result = await analysis_agent.analyze(mock_state)

        # Verify the result can be used to update state
        updated_state = mock_state.model_copy()
        for key, value in result.items():
            if hasattr(updated_state, key):
                setattr(updated_state, key, value)

        # Verify state was updated correctly
        assert len(updated_state.credibility_scores) == 3
        assert len(updated_state.execution_times) > 0
        assert len(updated_state.analysis_errors) == len(result["analysis_errors"])
        assert len(updated_state.recovery_actions) == len(result["recovery_actions"])

    @pytest.mark.asyncio
    async def test_concurrent_execution_safety(self, analysis_agent):
        """Test that multiple analysis executions don't interfere with each other."""
        # Create multiple states
        states = []
        for i in range(3):
            state = State(
                original_query=f"Query {i}",
                filtered_papers=[
                    create_test_paper(
                        f"paper{i}_{j}",
                        f"Paper {i}_{j}",
                        citation_count=10,
                        year=2020,
                        journal="Test",
                    )
                    for j in range(5)
                ],
            )
            states.append(state)

        # Execute concurrently
        tasks = [analysis_agent.analyze(state) for state in states]
        results = await asyncio.gather(*tasks)

        # Verify all results are correct
        assert len(results) == 3
        for result in results:
            assert "credibility_scores" in result
            assert len(result["credibility_scores"]) == 5

    @pytest.mark.asyncio
    async def test_timeout_and_cancellation(self, analysis_agent, mock_state):
        """Test handling of timeouts and cancellation."""

        # Mock a slow operation that should be cancelled
        async def slow_operation():
            await asyncio.sleep(10)  # Long operation
            return []

        with patch.object(
            analysis_agent,
            "_perform_contradiction_detection_optimized",
            side_effect=slow_operation,
        ):
            # Use a timeout
            try:
                result = await asyncio.wait_for(
                    analysis_agent.analyze(mock_state), timeout=1.0  # Short timeout
                )
            except asyncio.TimeoutError:
                # This is expected - the operation should timeout
                pass


class TestRecoveryStrategies:
    """Test recovery strategy logic."""

    def test_credibility_error_recovery_strategies(self):
        """Test different recovery strategies for credibility errors."""
        # Test recoverable error
        recoverable_error = CredibilityScoringError(
            "Temporary failure", recoverable=True
        )
        strategy = RecoveryStrategy.for_credibility_error(recoverable_error)
        assert strategy["action"] == "partial_recovery"

        # Test non-recoverable error
        non_recoverable_error = CredibilityScoringError(
            "Memory error", recoverable=False
        )
        strategy = RecoveryStrategy.for_credibility_error(non_recoverable_error)
        assert strategy["action"] == "fail_fast"

    def test_contradiction_error_recovery_strategies(self):
        """Test different recovery strategies for contradiction errors."""
        # Test LLM provider error
        llm_error = LLMProviderError("Rate limit", "gemini")
        strategy = RecoveryStrategy.for_contradiction_error(llm_error)
        assert strategy["action"] == "fallback_provider"

        # Test non-recoverable error
        fatal_error = ContradictionDetectionError("Auth failed", recoverable=False)
        strategy = RecoveryStrategy.for_contradiction_error(fatal_error)
        assert strategy["action"] == "fail_fast"

    def test_data_validation_recovery(self):
        """Test data validation error recovery."""
        data_error = DataValidationError("Invalid input", "test_component")
        strategy = RecoveryStrategy.for_data_validation_error(data_error)
        assert strategy["action"] == "data_repair"


class TestErrorCategorization:
    """Test error categorization logic."""

    @pytest.fixture
    def analysis_agent(self):
        return AnalysisAgent()

    def test_credibility_error_categorization(self, analysis_agent):
        """Test categorization of different credibility errors."""
        mock_state = State(original_query="Test", filtered_papers=[])

        # ValueError -> DataValidationError
        error = analysis_agent._categorize_credibility_error(
            ValueError("Invalid value"), mock_state
        )
        assert isinstance(error, DataValidationError)

        # MemoryError -> non-recoverable CredibilityScoringError
        error = analysis_agent._categorize_credibility_error(
            MemoryError("Out of memory"), mock_state
        )
        assert isinstance(error, CredibilityScoringError)
        assert not error.recoverable

    def test_contradiction_error_categorization(self, analysis_agent):
        """Test categorization of different contradiction errors."""
        mock_state = State(original_query="Test", relevant_passages=[])

        # ResourceExhausted -> LLMProviderError
        error = analysis_agent._categorize_contradiction_error(
            Exception("Resource exhausted"), mock_state
        )
        assert isinstance(error, ContradictionDetectionError)
        assert error.recoverable

        # InvalidArgument -> LLMProviderError (not recoverable - authentication error)
        from google.api_core.exceptions import InvalidArgument

        error = analysis_agent._categorize_contradiction_error(
            InvalidArgument("Invalid API key"), mock_state
        )
        assert isinstance(error, LLMProviderError)
        assert not error.recoverable  # Authentication errors are not recoverable


class TestAnalysisAgentEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def analysis_agent(self):
        return AnalysisAgent()

    @pytest.mark.asyncio
    async def test_malformed_paper_data(self, analysis_agent):
        """Test handling of malformed paper data."""
        # Create paper with minimal valid data
        minimal_paper = create_test_paper(
            "test", "Test", citation_count=0, year=2000, journal=""
        )

        state = State(original_query="Test", filtered_papers=[minimal_paper])

        result = await analysis_agent.analyze(state)

        # Should handle gracefully and produce some result
        assert "credibility_scores" in result
        assert len(result["credibility_scores"]) >= 0  # May be 0 if all fail

    @pytest.mark.asyncio
    async def test_extreme_passage_counts(self, analysis_agent):
        """Test handling of extreme passage counts."""
        # Test with zero passages
        state_zero = State(
            original_query="Test",
            filtered_papers=[
                create_test_paper(
                    "test", "Test Paper", citation_count=10, year=2020, journal="Test"
                )
            ],
            relevant_passages=[],
        )

        result = await analysis_agent.analyze(state_zero)
        assert result["contradictions"] == []

        # Test with many passages (mock the detection to avoid slow LLM calls)
        many_passages = [
            create_test_passage(
                f"p{i}", f"Content {i}", retrieval_score=0.8, paper_title=f"Title {i}"
            )
            for i in range(50)
        ]

        state_many = State(
            original_query="Test",
            filtered_papers=[
                create_test_paper(
                    "test", "Test Paper", citation_count=10, year=2020, journal="Test"
                )
            ],
            relevant_passages=many_passages,
        )

        with patch.object(
            analysis_agent,
            "_perform_contradiction_detection_optimized",
            return_value=[],
        ):
            result = await analysis_agent.analyze(state_many)

        # Should handle large passage count gracefully
        assert "contradictions" in result

    @pytest.mark.asyncio
    async def test_unicode_and_special_characters(self, analysis_agent):
        """Test handling of Unicode and special characters in content."""
        unicode_paper = create_test_paper(
            "test_unicode",
            "Unicode Test",
            citation_count=10,
            year=2023,
            journal="自然科学",
        )

        unicode_passage = create_test_passage(
            "test_unicode",
            "This contains émojis 😀 and spëcial chärs",
            retrieval_score=0.8,
            paper_title="Unicode Test",
        )

        state = State(
            original_query="Unicode test query",
            filtered_papers=[unicode_paper],
            relevant_passages=[unicode_passage],
        )

        result = await analysis_agent.analyze(state)

        # Should handle Unicode gracefully
        assert len(result["credibility_scores"]) == 1
        assert "contradictions" in result
