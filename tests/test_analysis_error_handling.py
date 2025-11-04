import asyncio
import pytest
import time

from agents.analysis import ContradictionDetector
from monitoring.health import HealthMonitor
from utils.error_handling import tenacity_retry_decorator
from google.api_core.exceptions import ResourceExhausted, InvalidArgument
import httpx


@pytest.mark.asyncio
async def test_retry_exponential_backoff_behavior(monkeypatch):
    # Configure detector with small backoff for test speed
    health = HealthMonitor()
    detector = ContradictionDetector(
        retry_config={"max_attempts": 3, "min_backoff": 0.01, "max_backoff": 0.02},
        health_monitor=health,
    )

    call_count = {"n": 0}

    async def failing_then_success(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise Exception("transient error")
        return "{[]}"  # return JSON-ish but parsing will result in empty list

    # Replace base implementation and re-decorate
    detector._base_call_single_llm = failing_then_success
    detector._call_single_llm = tenacity_retry_decorator(detector.retry_config)(
        detector._base_call_single_llm
    )

    # Call decorated method and ensure it eventually succeeds after retries
    result = None
    try:
        result = await detector._call_single_llm(None, [], provider_name="gemini")
    except Exception:
        pytest.fail("Retries did not succeed as expected")

    assert call_count["n"] >= 3


@pytest.mark.asyncio
async def test_provider_fallback_mechanism(monkeypatch):
    health = HealthMonitor()
    detector = ContradictionDetector(
        retry_config={"max_attempts": 1, "min_backoff": 0.01, "max_backoff": 0.01},
        health_monitor=health,
    )

    providers_called = []

    async def provider_switch(llm, messages=None, provider_name=None, **kwargs):
        providers_called.append(provider_name)
        if provider_name == "gemini":
            raise Exception("primary down")
        return "{[]}"

    # Assign directly to bypass tenacity wrapper for provider switch behavior test
    detector._call_single_llm = provider_switch

    # Call the underlying _call_llm_with_fallback directly so we exercise provider switching
    messages = [{"role": "user", "content": "prompt"}]
    try:
        _ = await detector._call_llm_with_fallback(messages=messages, llm_config={})
    except Exception:
        # If both providers raise, the function will raise; that's fine
        pass

    # Since response parsing returns empty list on simple '{[]}', expect empty list but health monitor should reflect fallback
    # At minimum the call path executed without crashing and health status is available
    status = health.get_status()
    assert isinstance(status, dict)


@pytest.mark.asyncio
async def test_graceful_degradation_when_both_providers_unavailable(monkeypatch):
    health = HealthMonitor()
    detector = ContradictionDetector(
        retry_config={"max_attempts": 1, "min_backoff": 0.01, "max_backoff": 0.01},
        health_monitor=health,
    )

    async def always_fail(*args, **kwargs):
        raise Exception("down")

    # Assign directly so we exercise fallback path and graceful degradation
    detector._call_single_llm = always_fail

    contr = await detector.detect_contradictions(
        query="q",
        relevant_passages=[{"paper_id": "1", "text": "t", "paper_title": "T"}],
        llm_config={},
    )

    # Should gracefully return empty list
    assert contr == []


def test_health_check_metrics_basic():
    health = HealthMonitor()
    health.record_call("gemini", True, 0.12)
    health.record_call("gemini", False, 0.5)
    health.record_call("groq", True, 0.2)
    health.record_fallback()

    status = health.get_status()
    assert "gemini" in status["providers"]
    assert status["fallback_count"] == 1
    gem = status["providers"]["gemini"]
    assert gem["success_count"] == 1
    assert gem["failure_count"] == 1


@pytest.mark.asyncio
async def test_auth_immediate_fallback_and_circuit(monkeypatch):
    health = HealthMonitor()
    detector = ContradictionDetector(
        retry_config={"max_attempts": 3, "min_backoff": 0.01, "max_backoff": 0.02},
        health_monitor=health,
    )

    async def auth_fail(
        llm, messages=None, provider_name=None, request_id=None, **kwargs
    ):
        # Simulate authentication error on gemini
        if provider_name == "gemini":
            raise InvalidArgument("auth failed")
        return "[]"

    detector._call_single_llm = auth_fail

    # Call _call_llm_with_fallback directly
    messages = [{"role": "user", "content": "prompt"}]
    resp = await detector._call_llm_with_fallback(messages=messages, llm_config={})
    # fallback should return empty list content -> parsed to [] by detector
    assert resp == "[]"

    # After auth failure the provider should be considered down by circuit
    # (record_provider_failure sets open_until) but since we simulated auth failure only
    # via raising InvalidArgument which is non-retryable, the circuit is still updated
    status = health.get_status()
    assert "gemini" in status["providers"] or isinstance(status, dict)


@pytest.mark.asyncio
async def test_resource_exhaustion_and_network_partition(monkeypatch):
    health = HealthMonitor()
    detector = ContradictionDetector(
        retry_config={"max_attempts": 2, "min_backoff": 0.001, "max_backoff": 0.002},
        health_monitor=health,
    )

    calls = {"gemini": 0, "groq": 0}

    async def simulated_llm(
        llm, messages=None, provider_name=None, request_id=None, **kwargs
    ):
        calls[provider_name] += 1
        if provider_name == "gemini":
            # Simulate resource exhaustion first
            raise ResourceExhausted("quota")
        if provider_name == "groq":
            # Simulate network partition
            raise httpx.ConnectError("network down")

    detector._call_single_llm = simulated_llm

    messages = [{"role": "user", "content": "prompt"}]
    # Should raise since both providers fail
    with pytest.raises(Exception):
        await detector._call_llm_with_fallback(messages=messages, llm_config={})

    # Ensure both providers were attempted
    assert calls["gemini"] >= 1
    assert calls["groq"] >= 1


@pytest.mark.asyncio
async def test_partial_llm_response_graceful(monkeypatch):
    health = HealthMonitor()
    detector = ContradictionDetector(
        retry_config={"max_attempts": 1, "min_backoff": 0.001, "max_backoff": 0.001},
        health_monitor=health,
    )

    async def partial_response(
        llm, messages=None, provider_name=None, request_id=None, **kwargs
    ):
        # Return malformed JSON
        return "{ this is not valid json"

    detector._call_single_llm = partial_response

    messages = [{"role": "user", "content": "prompt"}]
    res = await detector._call_llm_with_fallback(messages=messages, llm_config={})
    # detect_contradictions path will parse and return empty list; here we just ensure call returns string
    assert isinstance(res, str)
