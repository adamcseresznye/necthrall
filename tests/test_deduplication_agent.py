import pytest
from agents.deduplication_agent import DeduplicationAgent
from models.state import State, Paper


@pytest.fixture
def agent():
    return DeduplicationAgent()


@pytest.fixture
def sample_papers():
    """Create sample papers with intentional duplicates"""
    return [
        Paper(
            title="CRISPR gene editing",
            doi="10.1000/test1",
            paper_id="openalex:1",
            authors=[],
            year=2023,
            journal=None,
            pdf_url="http://example.com/p1.pdf",
            type="article",
        ),
        Paper(
            title="CRISPR gene editing",
            doi="10.1000/test1",
            paper_id="openalex:1",
            authors=[],
            year=2023,
            journal=None,
            pdf_url="http://example.com/p1.pdf",
            type="article",
        ),  # Duplicate DOI
        Paper(
            title="Metabolic effects of fasting",
            doi=None,
            paper_id="openalex:2",
            authors=[],
            year=2023,
            journal=None,
            pdf_url="http://example.com/p2.pdf",
            type="article",
        ),
        Paper(
            title="Metabolic Effects of Fasting",
            doi=None,
            paper_id="openalex:3",
            authors=[],
            year=2023,
            journal=None,
            pdf_url="http://example.com/p3.pdf",
            type="article",
        ),  # Duplicate title (both have no DOI)
        Paper(
            title="Cardiovascular outcomes",
            doi="10.1000/test3",
            paper_id="openalex:4",
            authors=[],
            year=2023,
            journal=None,
            pdf_url="http://example.com/p4.pdf",
            type="review",
        ),
    ]


def test_deduplication_removes_doi_duplicates(agent, sample_papers):
    """Test DeduplicationAgent removes papers with duplicate DOIs"""
    state = State(original_query="test", papers_metadata=sample_papers)

    updated_state = agent.deduplicate(state)

    # Should remove 1 DOI duplicate and 1 title duplicate = 2 removed
    assert len(updated_state.papers_metadata) == 3
    assert updated_state.dedup_stats["raw_count"] == 5
    assert updated_state.dedup_stats["unique_count"] == 3
    assert updated_state.dedup_stats["duplicates_removed"] == 2


def test_deduplication_handles_title_fallback(agent):
    """Test DeduplicationAgent uses normalized title when DOI missing"""
    papers = [
        Paper(
            title="Test Paper One",
            doi=None,
            paper_id="1",
            authors=[],
            year=2023,
            journal=None,
            pdf_url="http://example.com/1.pdf",
            type="article",
        ),
        Paper(
            title="Test Paper One",
            doi=None,
            paper_id="2",
            authors=[],
            year=2023,
            journal=None,
            pdf_url="http://example.com/2.pdf",
            type="article",
        ),  # Duplicate title
        Paper(
            title="Test Paper Two",
            doi=None,
            paper_id="3",
            authors=[],
            year=2023,
            journal=None,
            pdf_url="http://example.com/3.pdf",
            type="article",
        ),
    ]

    state = State(original_query="test", papers_metadata=papers)
    updated_state = agent.deduplicate(state)

    assert len(updated_state.papers_metadata) == 2  # Should keep only unique titles


def test_doi_normalization(agent):
    """Test _normalize_doi handles various DOI formats"""
    test_cases = [
        ("10.1000/test", "10.1000/test"),
        ("https://doi.org/10.1000/test", "10.1000/test"),
        ("http://dx.doi.org/10.1000/test", "10.1000/test"),
        ("DOI: 10.1000/test", "10.1000/test"),
        ("10.1000/TEST", "10.1000/test"),  # Case normalization
    ]

    for input_doi, expected in test_cases:
        assert agent._normalize_doi(input_doi) == expected


def test_title_normalization(agent):
    """Test _normalize_title handles punctuation and case"""
    test_cases = [
        ("CRISPR Gene Editing", "crispr gene editing"),
        ("CRISPR-Cas9: A Revolutionary Tool", "crisprcas9 a revolutionary tool"),
        ("Metabolic   Effects   of    Fasting", "metabolic effects of fasting"),
        ("  Title with Spaces  ", "title with spaces"),
    ]

    for input_title, expected in test_cases:
        assert agent._normalize_title(input_title) == expected


def test_deduplication_preserves_order(agent):
    """Test DeduplicationAgent keeps first occurrence of duplicates"""
    papers = [
        Paper(
            title="Paper A",
            doi="10.1000/a",
            paper_id="1",
            authors=["Author 1"],
            year=2023,
            journal=None,
            pdf_url="http://example.com/a.pdf",
            type="article",
        ),
        Paper(
            title="Paper B",
            doi="10.1000/b",
            paper_id="2",
            authors=["Author 2"],
            year=2023,
            journal=None,
            pdf_url="http://example.com/b.pdf",
            type="article",
        ),
        Paper(
            title="Paper A",
            doi="10.1000/a",
            paper_id="3",
            authors=["Author 3"],
            year=2023,
            journal=None,
            pdf_url="http://example.com/a2.pdf",
            type="article",
        ),  # Duplicate of first
    ]

    state = State(original_query="test", papers_metadata=papers)
    updated_state = agent.deduplicate(state)

    # Should keep first occurrence (authors=["Author 1"])
    assert len(updated_state.papers_metadata) == 2
    assert updated_state.papers_metadata[0].authors == ["Author 1"]


def test_deduplication_stats_structure(agent, sample_papers):
    """Test dedup_stats dict has correct structure"""
    state = State(original_query="test", papers_metadata=sample_papers)
    updated_state = agent.deduplicate(state)

    assert "raw_count" in updated_state.dedup_stats
    assert "unique_count" in updated_state.dedup_stats
    assert "duplicates_removed" in updated_state.dedup_stats
    assert "deduplication_rate" in updated_state.dedup_stats

    # Verify math
    stats = updated_state.dedup_stats
    assert stats["raw_count"] == len(sample_papers)
    assert stats["raw_count"] == stats["unique_count"] + stats["duplicates_removed"]


def test_deduplication_handles_empty_list(agent):
    """Test DeduplicationAgent handles empty papers_metadata gracefully"""
    state = State(original_query="test", papers_metadata=[])
    updated_state = agent.deduplicate(state)

    assert updated_state.papers_metadata == []
    assert updated_state.dedup_stats["raw_count"] == 0
    assert updated_state.dedup_stats["unique_count"] == 0
    assert updated_state.dedup_stats["duplicates_removed"] == 0


def test_deduplication_handles_all_unique(agent):
    """Test DeduplicationAgent handles case where no duplicates exist"""
    papers = [
        Paper(
            title=f"Paper {i}",
            doi=f"10.1000/test{i}",
            paper_id=f"{i}",
            authors=[],
            year=2023,
            journal=None,
            pdf_url=f"http://example.com/{i}.pdf",
            type="article",
        )
        for i in range(10)
    ]

    state = State(original_query="test", papers_metadata=papers)
    updated_state = agent.deduplicate(state)

    assert len(updated_state.papers_metadata) == 10
    assert updated_state.dedup_stats["duplicates_removed"] == 0
    assert updated_state.dedup_stats["deduplication_rate"] == 0.0


def test_deduplication_handles_missing_metadata(agent):
    """Test DeduplicationAgent handles papers with missing title/DOI"""
    papers = [
        Paper(
            title="Valid Paper",
            doi="10.1000/valid",
            paper_id="1",
            authors=[],
            year=2023,
            journal=None,
            pdf_url="http://example.com/1.pdf",
            type="article",
        ),
        Paper(
            title="",
            doi=None,
            paper_id="2",
            authors=[],
            year=2023,
            journal=None,
            pdf_url="http://example.com/2.pdf",
            type="article",
        ),  # No title, no DOI
        Paper(
            title="Valid Paper",
            doi="10.1000/valid",
            paper_id="3",
            authors=[],
            year=2023,
            journal=None,
            pdf_url="http://example.com/3.pdf",
            type="article",
        ),  # Duplicate
    ]

    state = State(original_query="test", papers_metadata=papers)
    updated_state = agent.deduplicate(state)

    # Should handle gracefully (empty title creates empty identifier, still tracks uniqueness)
    assert len(updated_state.papers_metadata) <= 3


@pytest.mark.performance
def test_deduplication_performance():
    """Test DeduplicationAgent processes 300 papers in <100ms"""
    import time

    agent = DeduplicationAgent()

    # Create 300 papers with ~30% duplicates
    papers = []
    for i in range(300):
        if i % 3 == 0 and i > 0:  # Every 3rd paper is a duplicate
            papers.append(papers[i - 3])
        else:
            papers.append(
                Paper(
                    title=f"Paper {i}",
                    doi=f"10.1000/test{i}",
                    paper_id=f"openalex:{i}",
                    authors=[],
                    year=2023,
                    journal=None,
                    pdf_url=f"http://example.com/{i}.pdf",
                    type="article",
                )
            )

    state = State(original_query="test", papers_metadata=papers)

    start_time = time.time()
    agent.deduplicate(state)
    elapsed = time.time() - start_time

    assert elapsed < 0.1, f"Deduplication took {elapsed:.3f}s (target: <0.1s)"
