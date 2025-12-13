"""Centralized LiteLLM router for model selection and fallback.

This module provides LLMRouter which wraps litellm.acompletion calls, picks
appropriate primary and fallback models from the project's `config` module and
retries automatically on failure.

Usage example:
    router = LLMRouter()
    content = await router.generate("Summarize this paper.", "synthesis")
"""

from typing import Optional
from loguru import logger
import asyncio
import litellm

import config.config as config


class LLMRouter:
    """A small router for LiteLLM calls that handles primary+fallback selection.

    It intentionally keeps logic minimal and async to add low overhead to
    actual model calls.
    """

    def __init__(self) -> None:
        # Store API keys for use in generate method
        self._primary_api_key = config.PRIMARY_LLM_API_KEY
        self._secondary_api_key = config.SECONDARY_LLM_API_KEY

        # Read model names from config on instantiation. Keep values simple so
        # tests can patch the module attributes.
        self._models = {
            "optimization": (
                config.QUERY_OPTIMIZATION_MODEL,
                config.QUERY_OPTIMIZATION_FALLBACK,
            ),
            "synthesis": (config.SYNTHESIS_MODEL, config.SYNTHESIS_FALLBACK),
        }

    async def _call_model(
        self, model: str, prompt: str, api_key: str, timeout: int = 30
    ):
        """Private helper to make an LLM API call with the given model and credentials.

        Args:
            model: The model identifier to use.
            prompt: The prompt text to send to the LLM.
            api_key: The API key for the provider.
            timeout: Request timeout in seconds (default 30).

        Returns:
            The response object from litellm.acompletion.

        Raises:
            Any exception from the LLM provider (APIError, TimeoutError, etc.)
        """
        return await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
            api_key=api_key,
            # max_tokens=4096,
            num_retries=0,
        )

    async def generate(self, prompt: str, model_type: str) -> str:
        """Generate a response for the given `prompt` using models associated
        with `model_type`.

        This method will attempt a primary model call first and fallback once on
        failure. It returns the inner content of the first choice:

            response.choices[0].message.content

        Raises the original exception if both primary and fallback fail.

        Args:
            prompt: The prompt text to send to the LLM.
            model_type: One of the keys present in the router ("optimization",
                "synthesis").

        Returns:
            The text content of the first model choice.
        """

        logger.debug(f"LLMRouter.generate entry: model_type={model_type}")

        mapping = self._models.get(model_type)
        if mapping is None:
            raise ValueError(f"Unknown model_type: {model_type}")

        primary, fallback = mapping
        timeout = 30

        # Track which model was ultimately used so we can log it at exit
        used_label: Optional[str] = None
        used_model_name: Optional[str] = None

        # Try primary model
        try:
            response = await self._call_model(
                primary, prompt, self._primary_api_key, timeout
            )
            used_label = "primary"
            used_model_name = primary
        except Exception as e:
            logger.warning(f"Primary LLM model {primary} failed: {e}")
            logger.info(f"Retrying LLM call with fallback model {fallback}")

            # Try fallback model
            try:
                response = await self._call_model(
                    fallback, prompt, self._secondary_api_key, timeout
                )
                used_label = "fallback"
                used_model_name = fallback
            except Exception as e_fallback:
                logger.exception(
                    f"Fallback LLM model {fallback} also failed: {e_fallback}"
                )
                raise

        # Extract the chat content like OpenAI style responses
        try:
            content = response.choices[0].message.content
            logger.debug(
                f"LLMRouter.generate exit: model_type={model_type} used={used_label} model={used_model_name}"
            )
            return content
        except Exception as e:
            logger.exception(f"Unexpected response structure from LLM: {e}")
            raise
