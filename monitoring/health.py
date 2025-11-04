from typing import Dict, Any
import time
import statistics
from loguru import logger


class HealthMonitor:
    """Tracks operational metrics for the Analysis Agent and LLM providers."""

    def __init__(self):
        # provider -> counts
        self._provider_success: Dict[str, int] = {}
        self._provider_failures: Dict[str, int] = {}
        self._provider_latencies: Dict[str, list] = {}
        self._fallback_count: int = 0
        self._total_calls: int = 0
        self._last_updated = time.time()
        # Circuit breaker state: provider -> {failure_count, open_until}
        self._provider_state = {}
        # token usage tracking
        self._provider_tokens = {}

    def record_call(self, provider: str, success: bool, latency: float) -> None:
        self._total_calls += 1
        self._last_updated = time.time()
        if provider not in self._provider_success:
            self._provider_success[provider] = 0
            self._provider_failures[provider] = 0
            self._provider_latencies[provider] = []

        if success:
            self._provider_success[provider] += 1
        else:
            self._provider_failures[provider] += 1

        # keep latencies for simple stats
        try:
            self._provider_latencies[provider].append(latency)
            # keep last 1000 entries to bound memory
            if len(self._provider_latencies[provider]) > 1000:
                self._provider_latencies[provider] = self._provider_latencies[provider][
                    -1000:
                ]
        except Exception:
            pass

        # update provider state success/failure counts
        state = self._provider_state.setdefault(
            provider, {"failure_count": 0, "open_until": 0}
        )
        if success:
            state["failure_count"] = 0
            state["open_until"] = 0
        else:
            state["failure_count"] = state.get("failure_count", 0) + 1

    def record_fallback(self) -> None:
        self._fallback_count += 1
        self._last_updated = time.time()

    def record_provider_failure(
        self, provider: str, cooldown_base: float = 5.0
    ) -> None:
        """Mark a provider failure and open circuit for exponential cooldown."""
        st = self._provider_state.setdefault(
            provider, {"failure_count": 0, "open_until": 0}
        )
        st["failure_count"] = st.get("failure_count", 0) + 1
        # exponential backoff for circuit open time
        cooldown = min(cooldown_base * (2 ** (st["failure_count"] - 1)), 300)
        st["open_until"] = time.time() + cooldown
        self._last_updated = time.time()
        logger.warning(
            {
                "event": "provider_marked_down",
                "provider": provider,
                "cooldown": cooldown,
            }
        )

    def record_provider_success(self, provider: str) -> None:
        st = self._provider_state.setdefault(
            provider, {"failure_count": 0, "open_until": 0}
        )
        st["failure_count"] = 0
        st["open_until"] = 0
        self._last_updated = time.time()

    def is_provider_available(self, provider: str) -> bool:
        st = self._provider_state.get(provider)
        if not st:
            return True
        open_until = st.get("open_until", 0)
        if open_until and time.time() < open_until:
            return False
        return True

    def record_tokens(self, provider: str, tokens: int) -> None:
        lst = self._provider_tokens.setdefault(provider, [])
        lst.append(tokens)
        if len(lst) > 1000:
            self._provider_tokens[provider] = lst[-1000:]
        self._last_updated = time.time()

    def get_status(self) -> Dict[str, Any]:
        """Return a health status snapshot suitable for a health endpoint."""
        status = {
            "uptime_seconds": round(time.time() - self._last_updated, 2),
            "total_calls": self._total_calls,
            "fallback_count": self._fallback_count,
            "providers": {},
        }

        for provider in set(
            list(self._provider_success.keys()) + list(self._provider_failures.keys())
        ):
            succ = self._provider_success.get(provider, 0)
            fail = self._provider_failures.get(provider, 0)
            lat_list = self._provider_latencies.get(provider, [])
            avg_latency = round(statistics.mean(lat_list), 3) if lat_list else None
            tokens = self._provider_tokens.get(provider, [])
            avg_tokens = int(statistics.mean(tokens)) if tokens else None
            state = self._provider_state.get(provider, {})
            provider_state = {
                "circuit_open": bool(
                    state.get("open_until", 0)
                    and time.time() < state.get("open_until", 0)
                ),
                "failure_count": state.get("failure_count", 0),
            }
            status["providers"][provider] = {
                "success_count": succ,
                "failure_count": fail,
                "success_rate": (
                    round((succ / (succ + fail)) * 100, 2)
                    if (succ + fail) > 0
                    else None
                ),
                "avg_latency": avg_latency,
                "avg_tokens": avg_tokens,
                "state": provider_state,
            }

        return status


def quick_health_check(monitor: HealthMonitor) -> Dict[str, Any]:
    """Small wrapper to return health info quickly (<100ms expected)."""
    start = time.time()
    status = monitor.get_status()
    status["response_time_ms"] = round((time.time() - start) * 1000, 2)
    # Add an overall state
    status["operational"] = True
    return status
