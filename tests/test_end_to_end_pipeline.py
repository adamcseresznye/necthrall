import pytest
import time
import sys
import os
from typing import List, Dict, Any

# Add project root to Python path for imports
sys.path.insert(0, os.path.abspath(".."))

from scripts.test_end_to_end_pipeline import main

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# Allow configuring the acceptable pipeline time via env var for CI variability
_ALLOWED_TIME_SEC = int(os.getenv("TEST_PIPELINE_TIME_LIMIT", "10"))


@pytest.mark.timeout(_ALLOWED_TIME_SEC + 2)
def test_end_to_end_pipeline_functionality():
    """Test complete pipeline functionality with performance and output validation."""
    # Record start time
    start_time = time.time()

    # Run the pipeline
    results = main()

    # Measure total time
    total_time = time.time() - start_time

    # Assert functionality
    assert isinstance(results, list), "Pipeline should return list of passages"
    assert len(results) == 10, f"Expected exactly 10 passages, got {len(results)}"

    # Validate output structure and section names
    valid_sections = {
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion",
        "fallback",
    }
    required_fields = {
        "content",
        "section",
        "retrieval_score",
        "cross_encoder_score",
        "final_score",
    }

    for i, passage in enumerate(results):
        assert isinstance(passage, dict), f"Passage {i} should be a dictionary"

        # Check required fields
        for field in required_fields:
            assert field in passage, f"Passage {i} missing required field: {field}"

        # Validate section names
        assert (
            passage["section"] in valid_sections
        ), f"Passage {i} has invalid section: {passage['section']}"

        # Validate scores are numeric
        assert isinstance(
            passage["retrieval_score"], (int, float)
        ), f"Passage {i} retrieval_score not numeric"
        assert isinstance(
            passage["cross_encoder_score"], (int, float)
        ), f"Passage {i} cross_encoder_score not numeric"
        assert isinstance(
            passage["final_score"], (int, float)
        ), f"Passage {i} final_score not numeric"

    # Performance validation: allow configurable threshold with 10% buffer
    allowed = float(_ALLOWED_TIME_SEC) * 1.1
    assert (
        total_time < allowed
    ), f"Pipeline took too long: {total_time:.3f}s (target: <{allowed:.1f}s)"

    # Log performance for monitoring
    print(
        f"Test completed in {total_time:.3f}s - {'PASS' if total_time < 4.4 else 'SLOW'}: {len(results)} passages returned"
    )
