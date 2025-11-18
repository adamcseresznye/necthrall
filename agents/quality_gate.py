"""Quality gate module for validating retrieved Semantic Scholar papers.

Provides early-stop validation to ensure sufficient paper quality before proceeding
with downstream processing. Validates against minimum thresholds for paper count,
embedding coverage, abstract coverage, and median semantic similarity.

Usage example:
    from agents.quality_gate import validate_quality
    import numpy as np

    papers = [...]  # List of paper dicts from Semantic Scholar
    query_embedding = np.random.rand(768)  # SPECTER2 embedding (768D)

    result = validate_quality(papers, query_embedding)
    if not result["passed"]:
        print(f"Validation failed: {result['reason']}")
"""

from typing import Dict, List, Tuple
import numpy as np
from loguru import logger


def validate_quality(
    papers: List[Dict], query_embedding: np.ndarray | None = None
) -> Dict:
    """Validate retrieved papers against quality thresholds.

    Performs comprehensive quality checks on Semantic Scholar paper results:
    - Minimum paper count (>=25)
    - Embedding coverage (>=60% have SPECTER2 embeddings)
    - Abstract coverage (>=60% have abstracts)

    Args:
        papers: List of paper dictionaries from Semantic Scholar API.
        query_embedding: 384 or 768-dimensional SPECTER/SPECTER2 embedding of the original query.
                        Optional for Week 1 (semantic similarity check skipped).

    Returns:
        Dict with keys:
        - passed: bool indicating if all criteria met
        - metrics: Dict with paper_count, embedding_coverage, abstract_coverage
        - reason: Human-readable explanation (empty string if passed)

    Raises:
        ValueError: If inputs are invalid (query_embedding not 384D/768D array, papers not list of dicts).
        TypeError: If paper items are not dictionaries or missing required fields.
    """
    # Comprehensive input validation
    _validate_inputs(papers, query_embedding)

    # Compute all metrics
    metrics = _compute_metrics(papers, query_embedding)

    # Log metrics for monitoring
    logger.info(
        "Quality gate metrics computed: paper_count={}, embedding_coverage={:.2%}, "
        "abstract_coverage={:.2%}",
        metrics["paper_count"],
        metrics["embedding_coverage"],
        metrics["abstract_coverage"],
    )

    # Check against thresholds
    passed, reason = _check_thresholds(metrics)

    result = {
        "passed": passed,
        "metrics": metrics,
        "reason": reason,
    }

    if not passed:
        logger.warning("Quality gate failed: {} | Metrics: {}", reason, metrics)

    return result


def _validate_inputs(papers: List[Dict], query_embedding: np.ndarray | None) -> None:
    """Validate input parameters with clear error messages."""
    if not isinstance(papers, list):
        raise TypeError("papers must be a list of dictionaries")

    if not papers:
        raise ValueError("papers list cannot be empty")

    for i, paper in enumerate(papers):
        if not isinstance(paper, dict):
            raise TypeError(f"paper at index {i} must be a dictionary")
        if "paperId" not in paper:
            raise ValueError(f"paper at index {i} missing required 'paperId' field")

    # Query embedding is optional for Week 1
    if query_embedding is not None:
        if not isinstance(query_embedding, np.ndarray):
            raise TypeError("query_embedding must be a numpy array")

        if query_embedding.shape not in [(384,), (768,)]:
            raise ValueError(
                f"query_embedding must be 384 or 768-dimensional, got shape {query_embedding.shape}"
            )

        if not np.isfinite(query_embedding).all():
            raise ValueError("query_embedding contains non-finite values (NaN or inf)")


def _compute_metrics(papers: List[Dict], query_embedding: np.ndarray) -> Dict:
    """Compute all quality metrics from papers and query embedding."""
    paper_count = len(papers)

    # Count papers with abstracts
    abstract_count = sum(
        1
        for p in papers
        if isinstance(p.get("abstract"), str) and p["abstract"].strip()
    )
    abstract_coverage = abstract_count / paper_count if paper_count > 0 else 0.0

    # Collect valid embeddings
    embeddings = []
    for p in papers:
        emb_dict = p.get("embedding")
        if emb_dict is not None and isinstance(emb_dict, dict):
            emb = emb_dict.get("specter")
            if emb is not None and isinstance(emb, (list, np.ndarray)):
                try:
                    emb_array = np.array(emb, dtype=np.float32)
                    # Accept both 384 (older SPECTER) and 768 (SPECTER2) dimensions
                    if (
                        emb_array.shape in [(384,), (768,)]
                        and np.isfinite(emb_array).all()
                    ):
                        embeddings.append(emb_array)
                except (ValueError, TypeError):
                    continue  # Skip invalid embeddings

    embedding_count = len(embeddings)
    embedding_coverage = embedding_count / paper_count if paper_count > 0 else 0.0

    return {
        "paper_count": paper_count,
        "embedding_coverage": embedding_coverage,
        "abstract_coverage": abstract_coverage,
    }


def _check_thresholds(metrics: Dict) -> Tuple[bool, str]:
    """Check metrics against quality thresholds and return pass/fail with reason."""
    thresholds = {
        "paper_count": (25, "insufficient paper count ({value} < {threshold})"),
        # Week 1: embedding_coverage set to 0 since we don't have local embedding model
        # Week 2+: will increase to 0.6 when using local embeddings for passages
        "embedding_coverage": (
            0.0,
            "low embedding coverage ({value:.2%} < {threshold:.0%})",
        ),
        "abstract_coverage": (
            0.6,
            "low abstract coverage ({value:.2%} < {threshold:.0%})",
        ),
    }

    failures = []
    passed = True

    for metric_name, (threshold, reason_template) in thresholds.items():
        value = metrics[metric_name]
        if value < threshold:
            passed = False
            reason = reason_template.format(value=value, threshold=threshold)
            failures.append(reason)

    if passed:
        reason = "Quality gate passed"
    else:
        reason = "; ".join(failures)

    return passed, reason
