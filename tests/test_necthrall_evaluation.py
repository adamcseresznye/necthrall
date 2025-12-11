"""
Validation tests to assess if the golden_dataset is accurate.
"""

import pytest
import json
import os
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from tests.eval_config import LLMJudge
from tests.citation_metrics import CitationValidityMetric, CitationAccuracyMetric

# Load dataset
DATASET_PATH = os.path.join("tests", "data", "golden_dataset.json")


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


dataset = load_dataset()


@pytest.fixture
def llm_judge():
    return LLMJudge()


@pytest.mark.parametrize("entry", dataset)
def test_faithfulness(entry, llm_judge):
    input_text = entry["input"]
    actual_output = entry["actual_output"]
    retrieval_context = entry["retrieval_context"]

    metric = FaithfulnessMetric(threshold=0.7, model=llm_judge, include_reason=True)

    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )

    # Measure manually so we can print the score
    metric.measure(test_case)

    # --- PRINT THE RESULTS ---
    print(f"\n[Faithfulness] Score: {metric.score}")
    print(f"[Faithfulness] Reason: {metric.reason}")
    # -------------------------

    # Assert success based on the threshold
    assert metric.is_successful(), f"Faithfulness failed. Score: {metric.score}"


@pytest.mark.parametrize("entry", dataset)
def test_citations(entry, llm_judge):
    input_text = entry["input"]
    actual_output = entry["actual_output"]
    retrieval_context = entry["retrieval_context"]

    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )

    # 1. Check Validity (Syntax)
    validity_metric = CitationValidityMetric()
    validity_metric.measure(test_case)
    print(f"\n[Citation Validity] Score: {validity_metric.score}")
    print(f"[Citation Validity] Reason: {validity_metric.reason}")

    # Assert validity first
    assert (
        validity_metric.is_successful()
    ), f"Citation Validity Failed: {validity_metric.reason}"

    # 2. Check Accuracy (Semantics) - Only if validity passes
    accuracy_metric = CitationAccuracyMetric(model=llm_judge)
    accuracy_metric.measure(test_case)
    print(f"[Citation Accuracy] Score: {accuracy_metric.score}")
    print(f"[Citation Accuracy] Reason: {accuracy_metric.reason}")

    assert (
        accuracy_metric.is_successful()
    ), f"Citation Accuracy Failed: {accuracy_metric.reason}"


@pytest.mark.parametrize("entry", dataset)
def test_answer_relevancy(entry, llm_judge):
    input_text = entry["input"]
    actual_output = entry["actual_output"]
    retrieval_context = entry["retrieval_context"]

    metric = AnswerRelevancyMetric(threshold=0.7, model=llm_judge, include_reason=True)

    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )

    # Measure manually so we can print the score
    metric.measure(test_case)

    # --- PRINT THE RESULTS ---
    print(f"\n[Relevancy] Score: {metric.score}")
    print(f"[Relevancy] Reason: {metric.reason}")
    # -------------------------

    assert metric.is_successful(), f"Relevancy failed. Score: {metric.score}"
