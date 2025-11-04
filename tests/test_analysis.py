import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio
from google.api_core.exceptions import ResourceExhausted, InvalidArgument
import httpx

pytestmark = [pytest.mark.unit]

from agents.analysis import (
    AnalysisCredibilityScorer,
    ContradictionDetector,
    AnalysisAgent,
)
from models.state import (
    State,
    CredibilityScore,
    DetectedContradiction,
    ContradictionClaim,
    Paper,
    Passage,
    ProcessingConfig,
)


def test_high_credibility_nature_2023():
    metadata = {
        "paper_id": "high_001",
        "citation_count": 250,
        "year": 2023,
        "journal": "Nature Medicine",
    }

    score = AnalysisCredibilityScorer.score_paper(metadata)
    assert isinstance(score, CredibilityScore)
    assert score.score >= 75
    assert score.tier == "high"


def test_low_credibility_arxiv_2015():
    metadata = {
        "paper_id": "low_001",
        "citation_count": 5,
        "year": 2015,
        "journal": "arXiv",
    }

    score = AnalysisCredibilityScorer.score_paper(metadata)
    assert isinstance(score, CredibilityScore)
    assert score.score < 50
    assert score.tier == "low"


def test_missing_metadata_defaults_medium():
    metadata = {"paper_id": "missing_001"}  # missing citation_count/year/journal
    score = AnalysisCredibilityScorer.score_paper(metadata)
    assert isinstance(score, CredibilityScore)
    assert score.score == 50
    assert score.tier == "medium"
    assert isinstance(score.rationale, str) and len(score.rationale) > 0


@pytest.mark.asyncio
async def test_successful_contradiction_detection():
    """Test successful contradiction detection with mocked Gemini response."""
    detector = ContradictionDetector()

    # Mock successful response
    mock_response = [
        {
            "topic": "cardiovascular effects",
            "claim_1": {
                "paper_id": "p1",
                "text": "IF reduces heart disease risk by 20%",
            },
            "claim_2": {
                "paper_id": "p2",
                "text": "IF increases arrhythmia risk in elderly",
            },
            "severity": "major",
        }
    ]

    with patch.object(
        detector, "_call_llm_with_fallback", new_callable=AsyncMock
    ) as mock_call:
        mock_call.return_value = json.dumps(mock_response)

        query = "intermittent fasting cardiovascular effects"
        passages = [
            {
                "paper_id": "p1",
                "text": "IF reduces heart disease risk by 20%",
                "paper_title": "Study 1",
            },
            {
                "paper_id": "p2",
                "text": "IF increases arrhythmia risk in elderly",
                "paper_title": "Study 2",
            },
        ]
        llm_config = {"GOOGLE_API_KEY": "test_key"}

        result = await detector.detect_contradictions(query, passages, llm_config)

        assert len(result) == 1
        assert isinstance(result[0], DetectedContradiction)
        assert result[0].topic == "cardiovascular effects"
        assert result[0].severity == "major"
        assert result[0].claim_1.paper_id == "p1"
        assert result[0].claim_2.paper_id == "p2"


@pytest.mark.asyncio
async def test_provider_fallback_mechanism():
    """Test fallback from Gemini to Groq when primary fails."""
    detector = ContradictionDetector()

    mock_response = []  # No contradictions

    with patch("agents.analysis.ChatGoogleGenerativeAI") as mock_gemini_class, patch(
        "agents.analysis.ChatGroq"
    ) as mock_groq_class:

        # Mock Gemini to fail
        mock_gemini_instance = AsyncMock()
        mock_gemini_instance.ainvoke.side_effect = Exception("Gemini API error")
        mock_gemini_class.return_value = mock_gemini_instance

        # Mock Groq to succeed
        mock_groq_instance = AsyncMock()
        mock_groq_instance.ainvoke.return_value = AsyncMock(
            content=json.dumps(mock_response)
        )
        mock_groq_class.return_value = mock_groq_instance

        query = "test query"
        passages = [{"paper_id": "p1", "text": "test text", "paper_title": "Test"}]
        llm_config = {"GOOGLE_API_KEY": "test", "GROQ_API_KEY": "test"}

        result = await detector.detect_contradictions(query, passages, llm_config)

        assert result == []
        # Verify Gemini was called and failed, then Groq was called
        mock_gemini_instance.ainvoke.assert_called()
        mock_groq_instance.ainvoke.assert_called()


@pytest.mark.asyncio
async def test_both_providers_fail():
    """Test graceful handling when both providers fail."""
    detector = ContradictionDetector()

    with patch("agents.analysis.ChatGoogleGenerativeAI") as mock_gemini_class, patch(
        "agents.analysis.ChatGroq"
    ) as mock_groq_class:

        # Both fail
        mock_gemini_instance = AsyncMock()
        mock_gemini_instance.ainvoke.side_effect = Exception("API error")
        mock_gemini_class.return_value = mock_gemini_instance

        mock_groq_instance = AsyncMock()
        mock_groq_instance.ainvoke.side_effect = Exception("API error")
        mock_groq_class.return_value = mock_groq_instance

        query = "test query"
        passages = [{"paper_id": "p1", "text": "test text", "paper_title": "Test"}]
        llm_config = {"GOOGLE_API_KEY": "test", "GROQ_API_KEY": "test"}

        result = await detector.detect_contradictions(query, passages, llm_config)

        assert result == []  # Should return empty list


@pytest.mark.asyncio
async def test_malformed_json_response():
    """Test handling of malformed JSON responses."""
    detector = ContradictionDetector()

    # Malformed response
    malformed_response = "Not a valid JSON response"

    with patch.object(
        detector, "_call_llm_with_fallback", new_callable=AsyncMock
    ) as mock_call:
        mock_call.return_value = malformed_response

        query = "test query"
        passages = [{"paper_id": "p1", "text": "test text", "paper_title": "Test"}]
        llm_config = {"GOOGLE_API_KEY": "test_key"}

        result = await detector.detect_contradictions(query, passages, llm_config)

        assert result == []  # Should return empty list on parse failure


@pytest.mark.asyncio
async def test_passage_token_optimization():
    """Test passage formatting with token optimization."""
    detector = ContradictionDetector()

    # Create a long passage with scientific content
    long_text = (
        """
    This study demonstrates significant results showing that intermittent fasting reduces cardiovascular risk by 25%.
    The research involved 500 participants over 2 years with rigorous methodology.
    Statistical analysis revealed p < 0.001 for the primary endpoint.
    However, subgroup analysis indicated potential risks in elderly patients with comorbidities.
    The findings suggest that while beneficial for healthy adults, intermittent fasting may increase arrhythmia risk in vulnerable populations.
    Further research is needed to clarify these contradictory findings.
    """
        * 10
    )  # Make it very long

    passages = [
        {
            "paper_id": "p1",
            "text": long_text,
            "paper_title": "Long Study on Intermittent Fasting",
        }
    ]

    formatted = detector._format_passages(passages)

    # Should be truncated to fit token limits
    assert len(formatted) < len(long_text)
    assert "reduces cardiovascular risk" in formatted  # Important claim preserved
    assert "increase arrhythmia risk" in formatted  # Contradictory claim preserved


@pytest.mark.asyncio
async def test_realistic_contradiction_scenario():
    """Test with realistic scientific contradiction scenario."""
    detector = ContradictionDetector()

    # Realistic contradictory passages
    mock_response = [
        {
            "topic": "intermittent fasting effects",
            "claim_1": {
                "paper_id": "study_A_2023",
                "text": "Intermittent fasting reduces cardiovascular disease risk by 22% (HR 0.78, 95% CI 0.65-0.94)",
            },
            "claim_2": {
                "paper_id": "study_B_2024",
                "text": "Intermittent fasting increases atrial fibrillation risk by 37% in patients over 65 years",
            },
            "severity": "major",
        },
        {
            "topic": "metabolic benefits",
            "claim_1": {
                "paper_id": "study_C_2022",
                "text": "Time-restricted eating improves insulin sensitivity by 15-20%",
            },
            "claim_2": {
                "paper_id": "study_D_2023",
                "text": "No significant difference in insulin sensitivity between intermittent fasting and continuous calorie restriction",
            },
            "severity": "minor",
        },
    ]

    with patch.object(
        detector, "_call_llm_with_fallback", new_callable=AsyncMock
    ) as mock_call:
        mock_call.return_value = json.dumps(mock_response)

        query = (
            "What are the cardiovascular and metabolic effects of intermittent fasting?"
        )
        passages = [
            {
                "paper_id": "study_A_2023",
                "text": "Our randomized controlled trial of 2400 participants found that intermittent fasting reduces cardiovascular disease risk by 22% (HR 0.78, 95% CI 0.65-0.94). This effect was most pronounced in participants under 65 years old.",
                "paper_title": "Cardiovascular Benefits of Intermittent Fasting",
            },
            {
                "paper_id": "study_B_2024",
                "text": "In a prospective cohort study of elderly patients, intermittent fasting was associated with a 37% increased risk of atrial fibrillation (OR 1.37, 95% CI 1.12-1.68), particularly in those with preexisting cardiovascular conditions.",
                "paper_title": "Intermittent Fasting and Arrhythmia Risk in Elderly",
            },
            {
                "paper_id": "study_C_2022",
                "text": "Time-restricted eating for 16 hours daily improved insulin sensitivity by 15-20% compared to ad libitum eating, with benefits sustained over 12 months of follow-up.",
                "paper_title": "Metabolic Effects of Time-Restricted Eating",
            },
            {
                "paper_id": "study_D_2023",
                "text": "A meta-analysis of 23 RCTs found no significant difference in insulin sensitivity between intermittent fasting and continuous calorie restriction (MD -0.12, 95% CI -0.35 to 0.11).",
                "paper_title": "Comparative Effectiveness of Dietary Interventions",
            },
        ]
        llm_config = {"GOOGLE_API_KEY": "test_key"}

        result = await detector.detect_contradictions(query, passages, llm_config)

        assert len(result) == 2
        assert all(isinstance(c, DetectedContradiction) for c in result)
        assert result[0].severity == "major"
        assert result[1].severity == "minor"
        assert "intermittent fasting" in result[0].topic.lower()
        assert "metabolic" in result[1].topic.lower()


@pytest.mark.asyncio
async def test_empty_passages_handling():
    """Test handling of empty or invalid passages."""
    detector = ContradictionDetector()

    mock_response = []

    with patch.object(
        detector, "_call_llm_with_fallback", new_callable=AsyncMock
    ) as mock_call:
        mock_call.return_value = json.dumps(mock_response)

        query = "test query"
        passages = [
            {"paper_id": "", "text": "", "paper_title": ""},  # Empty
            {"paper_id": "p1", "text": None, "paper_title": None},  # None values
            {"paper_id": "p2"},  # Missing text/title
        ]
        llm_config = {"GOOGLE_API_KEY": "test_key"}

        result = await detector.detect_contradictions(query, passages, llm_config)

        assert result == []
        # Should handle gracefully without crashing


@pytest.mark.asyncio
async def test_max_contradictions_limit():
    """Test that only maximum 3 contradictions are returned."""
    detector = ContradictionDetector()

    # Create 5 contradictions in response
    mock_response = [
        {
            "topic": f"topic_{i}",
            "claim_1": {"paper_id": f"p{i}a", "text": f"claim {i} a"},
            "claim_2": {"paper_id": f"p{i}b", "text": f"claim {i} b"},
            "severity": "major",
        }
        for i in range(5)
    ]

    with patch.object(
        detector, "_call_llm_with_fallback", new_callable=AsyncMock
    ) as mock_call:
        mock_call.return_value = json.dumps(mock_response)

        query = "test query"
        passages = [
            {"paper_id": f"p{i}", "text": f"text {i}", "paper_title": f"Title {i}"}
            for i in range(10)
        ]
        llm_config = {"GOOGLE_API_KEY": "test_key"}

        result = await detector.detect_contradictions(query, passages, llm_config)

        assert len(result) == 3  # Should be limited to 3


@pytest.mark.asyncio
async def test_analysis_agent_successful_execution():
    """Test case 1: Successful analysis node execution with mock state containing papers and passages."""
    agent = AnalysisAgent()

    # Create mock state
    state = State(
        original_query="test query",
        optimized_query="optimized test query",
        filtered_papers=[
            Paper(
                paper_id="p1",
                title="Test Paper 1",
                authors=["Author 1"],
                year=2023,
                citation_count=100,
                journal="Nature",
                type="article",
                pdf_url="https://example.com/paper1.pdf",
            )
        ],
        relevant_passages=[
            Passage(
                content="test content",
                paper_id="p1",
                retrieval_score=0.9,
            )
        ],
        config=ProcessingConfig(),
        execution_times={"previous": 1.0},
    )

    # Mock contradiction detection to return empty list
    with patch.object(
        agent.contradiction_detector, "detect_contradictions", new_callable=AsyncMock
    ) as mock_detect:
        mock_detect.return_value = []

        result = await agent.analyze(state)

        # Verify result structure
        assert "credibility_scores" in result
        assert "contradictions" in result
        assert "execution_times" in result

        # Verify credibility scores
        assert len(result["credibility_scores"]) == 1
        score = result["credibility_scores"][0]
        assert isinstance(score, CredibilityScore)
        assert score.paper_id == "p1"
        assert score.score >= 70  # Should be high credibility

        # Verify contradictions
        assert result["contradictions"] == []

        # Verify execution times
        assert "analysis_agent" in result["execution_times"]
        assert isinstance(result["execution_times"]["analysis_agent"], float)
        assert (
            result["execution_times"]["previous"] == 1.0
        )  # Should preserve existing times


@pytest.mark.asyncio
async def test_analysis_agent_partial_failure():
    """Test case 2: Partial failure handling when credibility scoring succeeds but contradiction detection fails."""
    agent = AnalysisAgent()

    # Create mock state
    state = State(
        original_query="test query",
        filtered_papers=[
            Paper(
                paper_id="p1",
                title="Test Paper 1",
                authors=["Author 1"],
                year=2023,
                citation_count=100,
                journal="Nature",
                type="article",
                pdf_url="https://example.com/paper1.pdf",
            )
        ],
        relevant_passages=[
            Passage(
                content="test content",
                paper_id="p1",
                retrieval_score=0.9,
            )
        ],
        config=ProcessingConfig(),
        execution_times={},
    )

    # Mock contradiction detection to fail
    with patch.object(
        agent.contradiction_detector, "detect_contradictions", new_callable=AsyncMock
    ) as mock_detect:
        mock_detect.side_effect = Exception("LLM API failure")

        result = await agent.analyze(state)

        # Should still return credibility scores
        assert len(result["credibility_scores"]) == 1
        assert isinstance(result["credibility_scores"][0], CredibilityScore)

        # Should return empty contradictions on failure
        assert result["contradictions"] == []

        # Should still have execution time
        assert "analysis_agent" in result["execution_times"]


@pytest.mark.asyncio
async def test_analysis_agent_complete_failure():
    """Test case 3: Complete failure recovery returning empty results but valid state structure."""
    agent = AnalysisAgent()

    # Create mock state with invalid data that might cause failures
    state = State(
        original_query="test query",
        filtered_papers=[
            Paper(
                paper_id="p1",
                title="Test Paper 1",
                authors=["Author 1"],
                year=2023,
                citation_count=100,
                journal="Nature",
                type="article",
                pdf_url="https://example.com/paper1.pdf",
            )
        ],
        relevant_passages=[
            Passage(
                content="test content",
                paper_id="p1",
                retrieval_score=0.9,
            )
        ],
        config=ProcessingConfig(),
        execution_times={},
    )

    # Mock both components to fail
    with patch.object(
        agent.credibility_scorer, "score_paper"
    ) as mock_score, patch.object(
        agent.contradiction_detector, "detect_contradictions", new_callable=AsyncMock
    ) as mock_detect:

        mock_score.side_effect = Exception("Scoring failure")
        mock_detect.side_effect = Exception("Detection failure")

        result = await agent.analyze(state)

        # Should return default scores and empty contradictions on complete failure
        assert len(result["credibility_scores"]) == 1
        assert isinstance(result["credibility_scores"][0], CredibilityScore)
        assert result["credibility_scores"][0].score == 50  # Default score
        assert result["credibility_scores"][0].tier == "medium"
        assert result["contradictions"] == []
        assert "analysis_agent" in result["execution_times"]
        assert isinstance(result["execution_times"]["analysis_agent"], float)


@pytest.mark.asyncio
async def test_analysis_agent_state_transition_validation():
    """Test case 4: State transition validation ensuring proper field updates."""
    agent = AnalysisAgent()

    # Create initial state
    initial_times = {"processing": 2.5}
    state = State(
        original_query="test query",
        filtered_papers=[
            Paper(
                paper_id="p1",
                title="Test Paper 1",
                authors=["Author 1"],
                year=2023,
                citation_count=100,
                journal="Nature",
                type="article",
                pdf_url="https://example.com/paper1.pdf",
            )
        ],
        relevant_passages=[],
        config=ProcessingConfig(),
        execution_times=initial_times.copy(),
    )

    # Mock contradiction detection
    with patch.object(
        agent.contradiction_detector, "detect_contradictions", new_callable=AsyncMock
    ) as mock_detect:
        mock_detect.return_value = []

        result = await agent.analyze(state)

        # Verify execution times are updated correctly
        assert result["execution_times"]["processing"] == 2.5  # Preserved
        assert "analysis_agent" in result["execution_times"]
        assert result["execution_times"]["analysis_agent"] > 0

        # Verify all required fields are present
        required_fields = ["credibility_scores", "contradictions", "execution_times"]
        for field in required_fields:
            assert field in result

        # Verify types
        assert isinstance(result["credibility_scores"], list)
        assert isinstance(result["contradictions"], list)
        assert isinstance(result["execution_times"], dict)
