"""Quality gate module for validating retrieved Semantic Scholar papers.

Provides early-stop validation to ensure sufficient paper quality before proceeding
with downstream processing. Validates against minimum thresholds for paper count
and abstract coverage.

Usage example:
    from agents.quality_gate import validate_quality

    papers = [...]  # List of paper dicts from Semantic Scholar

    result = validate_quality(papers)
    if not result["passed"]:
        print(f"Validation failed: {result['reason']}")
"""

from typing import Dict, List, Tuple

from loguru import logger


def validate_quality(papers: List[Dict]) -> Dict:
    """Validate retrieved papers against quality thresholds.

    Performs comprehensive quality checks on Semantic Scholar paper results:
    - Minimum paper count (>=25)
    - Abstract coverage (>=60% have abstracts)

    Args:
        papers: List of paper dictionaries from Semantic Scholar API.

    Returns:
        Dict with keys:
        - passed: bool indicating if all criteria met
        - metrics: Dict with paper_count, abstract_coverage
        - reason: Human-readable explanation (empty string if passed)

    Raises:
        ValueError: If inputs are invalid.
        TypeError: If paper items are not dictionaries or missing required fields.
    """
    # Comprehensive input validation
    _validate_inputs(papers)

    # Compute all metrics
    metrics = _compute_metrics(papers)

    # Log metrics for monitoring
    logger.info(
        "Quality gate metrics computed: paper_count={}, abstract_coverage={:.2%}",
        metrics["paper_count"],
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


def _validate_inputs(papers: List[Dict]) -> None:
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


def _compute_metrics(papers: List[Dict]) -> Dict:
    """Compute all quality metrics from papers."""
    paper_count = len(papers)

    # Count papers with abstracts
    abstract_count = sum(
        1
        for p in papers
        if isinstance(p.get("abstract"), str) and p["abstract"].strip()
    )
    abstract_coverage = abstract_count / paper_count if paper_count > 0 else 0.0

    return {
        "paper_count": paper_count,
        "abstract_coverage": abstract_coverage,
    }


def _check_thresholds(metrics: Dict) -> Tuple[bool, str]:
    """Check metrics against quality thresholds and return pass/fail with reason."""
    thresholds = {
        "paper_count": (25, "insufficient paper count ({value} < {threshold})"),
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
