import asyncio
import json
import os
import sys

import pytest

# python -m tests.test_necthrall_real
# --- WINDOWS DLL FIX START ---
# This must be done BEFORE importing torch
if os.name == "nt":
    try:
        torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
        if os.path.isdir(torch_lib):
            try:
                os.add_dll_directory(torch_lib)
            except Exception:
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
# --- WINDOWS DLL FIX END ---

# CRITICAL: Import torch FIRST before any other library that might load conflicting DLLs
# This prevents DLL conflicts with onnxruntime, sentence-transformers, etc.
try:
    import torch
except ImportError:
    pass

from unittest.mock import patch

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

import agents.quality_gate
from config.config import get_settings
from services.query_service import QueryService
from tests.citation_metrics import CitationAccuracyMetric, CitationValidityMetric
from tests.eval_config import LLMJudge


# Monkey-patch quality gate thresholds for testing
def lenient_check_thresholds(metrics):
    # Allow fewer papers for testing
    thresholds = {
        "paper_count": (1, "insufficient paper count ({value} < {threshold})"),
        "embedding_coverage": (0.0, "low embedding coverage"),
        "abstract_coverage": (0.0, "low abstract coverage"),
    }
    failures = []
    passed = True
    for metric_name, (threshold, reason_template) in thresholds.items():
        value = metrics[metric_name]
        if value < threshold:
            passed = False
            reason = reason_template.format(value=value, threshold=threshold)
            failures.append(reason)

    return passed, "; ".join(failures) if not passed else "Quality gate passed"


agents.quality_gate._check_thresholds = lenient_check_thresholds


# Load dataset
DATASET_PATH = os.path.join("tests", "data", "golden_dataset.json")


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# Load only the first 2 entries to save costs/time
full_dataset = load_dataset()
dataset = full_dataset[:2] if full_dataset else []


@pytest.fixture(scope="module")
def llm_judge():
    return LLMJudge()


@pytest.fixture(scope="module")
def app_service():
    # Initialize the service (embedding model loads lazily)
    return QueryService(get_settings(), None)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", dataset)
async def test_necthrall_end_to_end(entry, app_service, llm_judge):
    input_query = entry["input"]
    expected_output = entry[
        "actual_output"
    ]  # Not strictly used for RAG eval but good for reference

    print(f"\n\n--- Testing Query: {input_query} ---")

    # 1. Run the Application (The "Student")
    result = await app_service.process_query(input_query)

    # Check for pipeline failure
    assert result.success, f"Pipeline failed: {result.error}"
    assert result.answer, "Pipeline returned empty answer"

    actual_answer = result.answer
    # Extract text from passage objects
    retrieved_passages = [p.text for p in result.passages] if result.passages else []

    print(f"Answer Generated: {actual_answer[:100]}...")
    print(f"Passages Retrieved: {len(retrieved_passages)}")

    # 2. Evaluate (The "Teacher")
    test_case = LLMTestCase(
        input=input_query,
        actual_output=actual_answer,
        retrieval_context=retrieved_passages,
        expected_output=expected_output,
    )

    # Initialize Metrics
    metrics = [
        # FaithfulnessMetric(threshold=0.7, model=llm_judge, include_reason=True),
        # AnswerRelevancyMetric(threshold=0.7, model=llm_judge, include_reason=True),
        CitationValidityMetric(threshold=1.0),
        CitationAccuracyMetric(model=llm_judge, threshold=0.7),
    ]

    # Measure and Print Results
    for metric in metrics:
        print(f"\nRunning {metric.__name__}...")
        # Handle async measurement for CitationAccuracyMetric if needed,
        # though deepeval usually handles it via .measure() wrapper.
        # However, since we are in an async test, we can call measure directly.
        metric.measure(test_case)
        print(f"[{metric.__name__}] Score: {metric.score}")
        print(f"[{metric.__name__}] Reason: {metric.reason}")

    # 3. Assert Success
    # We use assert_test to check all metrics at once
    # assert_test(test_case, metrics)

    print("\n--- Final Results ---")
    failed_metrics = []
    for metric in metrics:
        if not metric.is_successful():
            failed_metrics.append(f"{metric.__name__} (Score: {metric.score})")
            print(f"❌ {metric.__name__} FAILED")
        else:
            print(f"✅ {metric.__name__} PASSED")

    if failed_metrics:
        raise AssertionError(
            f"The following metrics failed: {', '.join(failed_metrics)}"
        )


if __name__ == "__main__":
    # Manual test execution
    async def run_manual_test():
        print("Running manual test...")
        try:
            # Setup
            os.environ["LLM_MODEL_PRIMARY"] = "cerebras/llama3.1-8b"
            os.environ["LLM_MODEL_FALLBACK"] = "groq/llama-3.1-8b-instant"

            # Initialize fixtures
            judge = LLMJudge()

            # Load embedding model
            print("Loading embedding model...")
            from config.onnx_embedding import initialize_embedding_model

            embedding_model = initialize_embedding_model()

            service = QueryService(get_settings(), embedding_model)

            # Load data
            data = load_dataset()
            if not data:
                print("No data found.")
                return

            # Run for all entries and report per-entry results
            failures = []
            for i, entry in enumerate(data):
                print(
                    f"\n--- Running manual test {i+1}/{len(data)}: {entry['input']} ---"
                )
                try:
                    await test_necthrall_end_to_end(entry, service, judge)
                    print(f"Test {i+1} passed!")
                except Exception as e:
                    print(f"Test {i+1} failed: {e}")
                    failures.append(
                        {"index": i, "input": entry.get("input"), "error": str(e)}
                    )
                    import traceback

                    traceback.print_exc()
            if failures:
                raise AssertionError(
                    f"{len(failures)} manual tests failed: {[f['index'] for f in failures]}"
                )
        except Exception as e:
            print(f"Manual test run failed: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(run_manual_test())
            traceback.print_exc()

    asyncio.run(run_manual_test())
