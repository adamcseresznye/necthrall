import pytest
from unittest.mock import Mock, patch
from models.state import State
from orchestrator.graph import build_workflow, should_refine_query
from agents.fallback_refinement_agent import FallbackRefinementAgent


def test_should_refine_query_triggers_on_failure():
    """Test routing function triggers refinement when search quality fails"""
    state = State(
        original_query="test",
        search_quality={
            "passed": False,
            "reason": "Only 5 papers found",
            "paper_count": 5,
            "avg_relevance": 0.3,
        },
        refinement_count=0,
    )

    assert should_refine_query(state) == "refine"


def test_should_refine_query_continues_on_success():
    """Test routing function continues when search quality passes"""
    state = State(
        original_query="test",
        search_quality={
            "passed": True,
            "reason": "Found 78 papers",
            "paper_count": 78,
            "avg_relevance": 0.7,
        },
        refinement_count=0,
    )

    assert should_refine_query(state) == "continue"


def test_should_refine_query_stops_after_max_attempts():
    """Test routing function stops refinement after 2 attempts"""
    state = State(
        original_query="test",
        search_quality={
            "passed": False,
            "reason": "Still insufficient",
            "paper_count": 3,
            "avg_relevance": 0.2,
        },
        refinement_count=2,
    )

    assert should_refine_query(state) == "continue"  # Max attempts reached


def test_should_refine_query_handles_missing_search_quality():
    """Test routing function defaults to continue when search_quality is None"""
    state = State(original_query="test", search_quality=None, refinement_count=0)

    assert should_refine_query(state) == "continue"


def test_fallback_refinement_updates_query():
    """Test FallbackRefinementAgent refines query with context"""
    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(
        content="AMPK and mTOR pathway modulation during intermittent fasting protocols"
    )
    agent = FallbackRefinementAgent(llm=mock_llm)

    state = State(
        original_query="fasting",
        optimized_query="metabolic effects of intermittent fasting",
        search_quality={
            "passed": False,
            "paper_count": 5,
            "avg_relevance": 0.2,
            "reason": "Poor results",
        },
        refinement_count=0,
    )

    updated_state = agent.refine(state)

    assert updated_state.optimized_query != "metabolic effects of intermittent fasting"
    assert updated_state.refinement_count == 1


def test_fallback_refinement_handles_short_response():
    """Test FallbackRefinementAgent falls back to optimized query when response too short"""
    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="ok")  # Too short
    agent = FallbackRefinementAgent(llm=mock_llm)

    state = State(
        original_query="fasting",
        optimized_query="metabolic effects of intermittent fasting",
        search_quality={
            "passed": False,
            "paper_count": 5,
            "avg_relevance": 0.2,
            "reason": "Poor results",
        },
        refinement_count=0,
    )

    updated_state = agent.refine(state)

    assert (
        updated_state.optimized_query
        == "AMPK and mTOR pathway modulation during intermittent fasting protocols"
    )
    assert updated_state.refinement_count == 1


def test_fallback_refinement_increments_count_on_error():
    """Test FallbackRefinementAgent increments refinement_count even on error"""
    mock_llm = Mock()
    mock_llm.invoke.side_effect = Exception("API error")
    agent = FallbackRefinementAgent(llm=mock_llm)

    state = State(
        original_query="fasting",
        optimized_query="metabolic effects of intermittent fasting",
        search_quality={
            "passed": False,
            "paper_count": 5,
            "avg_relevance": 0.2,
            "reason": "Poor results",
        },
        refinement_count=0,
    )

    updated_state = agent.refine(state)

    assert updated_state.refinement_count == 1  # Still incremented


def test_fallback_refinement_no_llm():
    """Test FallbackRefinementAgent works when LLM is not available (for testing)"""
    agent = FallbackRefinementAgent(llm=None)  # Explicitly set to None

    state = State(
        original_query="fasting",
        optimized_query="metabolic effects of intermittent fasting",
        search_quality={
            "passed": False,
            "paper_count": 5,
            "avg_relevance": 0.2,
            "reason": "Poor results",
        },
        refinement_count=0,
    )

    updated_state = agent.refine(state)

    # Should add "refined " prefix when LLM is None
    expected = "refined metabolic effects of intermittent fasting"
    assert updated_state.optimized_query == expected
    assert updated_state.refinement_count == 1


@pytest.mark.integration
def test_workflow_executes_without_refinement():
    """Test workflow completes happy path without triggering refinement"""
    # Mock request with embedding model
    request = Mock()
    request.app.state.embedding_model = Mock()

    # Mock agents to avoid Google API credential issues
    mock_query_optimizer = Mock()
    mock_search_agent = Mock()
    mock_dedup_agent = Mock()
    mock_filtering_agent = Mock()
    mock_fallback_refiner = Mock()
    mock_acquisition_agent = Mock()

    mock_agents = {
        "query_optimizer": mock_query_optimizer,
        "search_agent": mock_search_agent,
        "dedup_agent": mock_dedup_agent,
        "filtering_agent": mock_filtering_agent,
        "fallback_refiner": mock_fallback_refiner,
        "acquisition_agent": mock_acquisition_agent,
    }

    # Build workflow with mock agents
    workflow = build_workflow(request, mock_agents)

    # Create initial state
    initial_state = State(original_query="CRISPR gene editing")

    # Configure mock agent behaviors to simulate successful flow
    mock_query_optimizer.optimize.return_value = initial_state
    mock_search_agent.search.return_value = State(
        **{
            **initial_state.model_dump(),
            "search_quality": {
                "passed": True,
                "paper_count": 78,
                "reason": "Good results",
                "avg_relevance": 0.8,
            },
        }
    )
    mock_dedup_agent.deduplicate.return_value = initial_state
    mock_filtering_agent.filter_candidates.return_value = initial_state
    mock_acquisition_agent.return_value = initial_state

    # Execute workflow
    final_result = workflow.invoke(initial_state)
    final_state = State(**final_result)

    # Verify no refinement triggered
    assert final_state.refinement_count == 0


@pytest.mark.integration
def test_workflow_triggers_refinement_loop():
    """Test workflow triggers fallback refinement when search quality fails"""
    request = Mock()
    request.app.state.embedding_model = Mock()

    # Mock agents to avoid Google API credential issues
    mock_query_optimizer = Mock()
    mock_search_agent = Mock()
    mock_dedup_agent = Mock()
    mock_filtering_agent = Mock()
    mock_fallback_refiner = Mock()
    mock_acquisition_agent = Mock()

    mock_agents = {
        "query_optimizer": mock_query_optimizer,
        "search_agent": mock_search_agent,
        "dedup_agent": mock_dedup_agent,
        "filtering_agent": mock_filtering_agent,
        "fallback_refiner": mock_fallback_refiner,
        "acquisition_agent": mock_acquisition_agent,
    }

    workflow = build_workflow(request, mock_agents)
    initial_state = State(original_query="vague query")

    # First search fails
    mock_query_optimizer.optimize.return_value = initial_state
    mock_search_agent.search.return_value = State(
        **{
            **initial_state.model_dump(),
            "search_quality": {
                "passed": False,
                "paper_count": 5,
                "reason": "Poor results",
                "avg_relevance": 0.2,
            },
        }
    )
    mock_fallback_refiner.refine.return_value = State(
        **{**initial_state.model_dump(), "refinement_count": 1}
    )

    # Second search succeeds (after refinement)
    mock_search_agent.search.side_effect = [
        State(
            **{
                **initial_state.model_dump(),
                "search_quality": {
                    "passed": False,
                    "paper_count": 5,
                    "reason": "Poor results",
                    "avg_relevance": 0.2,
                },
            }
        ),  # First call (fails)
        State(
            **{
                **initial_state.model_dump(),
                "search_quality": {
                    "passed": True,
                    "paper_count": 75,
                    "reason": "Good after refinement",
                    "avg_relevance": 0.8,
                },
            }
        ),  # Second call (succeeds)
    ]
    mock_dedup_agent.deduplicate.return_value = State(
        **{**initial_state.model_dump(), "refinement_count": 1}
    )
    mock_filtering_agent.filter_candidates.return_value = State(
        **{**initial_state.model_dump(), "refinement_count": 1}
    )
    mock_acquisition_agent.return_value = State(
        **{**initial_state.model_dump(), "refinement_count": 1}
    )

    # Execute workflow
    final_result = workflow.invoke(initial_state)
    final_state = State(**final_result)

    # Verify refinement was triggered
    assert final_state.refinement_count >= 1


@pytest.mark.integration
def test_workflow_stops_after_max_refinements():
    """Test workflow stops refinement after 2 attempts"""
    request = Mock()
    request.app.state.embedding_model = Mock()

    # Mock agents to avoid Google API credential issues
    mock_query_optimizer = Mock()
    mock_search_agent = Mock()
    mock_dedup_agent = Mock()
    mock_filtering_agent = Mock()
    mock_fallback_refiner = Mock()
    mock_acquisition_agent = Mock()

    mock_agents = {
        "query_optimizer": mock_query_optimizer,
        "search_agent": mock_search_agent,
        "dedup_agent": mock_dedup_agent,
        "filtering_agent": mock_filtering_agent,
        "fallback_refiner": mock_fallback_refiner,
        "acquisition_agent": mock_acquisition_agent,
    }

    workflow = build_workflow(request, mock_agents)
    initial_state = State(original_query="vague query", refinement_count=2)

    # Configure mocks
    mock_query_optimizer.optimize.return_value = initial_state
    mock_search_agent.search.return_value = State(
        **{
            **initial_state.model_dump(),
            "search_quality": {
                "passed": False,
                "paper_count": 5,
                "reason": "Poor results",
                "avg_relevance": 0.2,
            },
        }
    )
    mock_dedup_agent.deduplicate.return_value = State(
        **{**initial_state.model_dump(), "refinement_count": 2}
    )
    mock_filtering_agent.filter_candidates.return_value = State(
        **{**initial_state.model_dump(), "refinement_count": 2}
    )
    mock_acquisition_agent.return_value = State(
        **{**initial_state.model_dump(), "refinement_count": 2}
    )

    # Execute workflow
    final_result = workflow.invoke(initial_state)
    final_state = State(**final_result)

    # Verify no additional refinement attempted (count stays at 2)
    assert final_state.refinement_count == 2
