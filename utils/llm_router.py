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
        self._google_api_key = config.GOOGLE_API_KEY
        self._groq_api_key = config.GROQ_API_KEY

        # Read model names from config on instantiation. Keep values simple so
        # tests can patch the module attributes.
        self._models = {
            "optimization": (
                config.QUERY_OPTIMIZATION_MODEL,
                config.QUERY_OPTIMIZATION_FALLBACK,
            ),
            "synthesis": (config.SYNTHESIS_MODEL, config.SYNTHESIS_FALLBACK),
        }

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
        # Choose a default timeout per high level plan
        timeout = 30

        # Track which model was ultimately used so we can log it at exit
        used_label: Optional[str] = None
        used_model_name: Optional[str] = None

        try:
            response = await litellm.acompletion(
                model=primary,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
                api_key=self._google_api_key,
            )
            used_label = "primary"
            used_model_name = primary
        except litellm.exceptions.APIError as e:
            # API errors from providers
            logger.warning(f"Primary LLM APIError for model {primary}: {e}")
            logger.info(f"Retrying LLM call with fallback model {fallback}")
            try:
                response = await litellm.acompletion(
                    model=fallback,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=timeout,
                    api_key=self._groq_api_key,
                )
                used_label = "fallback"
                used_model_name = fallback
            except (litellm.exceptions.APIError, asyncio.TimeoutError) as e2:
                logger.exception(
                    f"Fallback LLM model {fallback} also failed with API/Timeout: {e2}"
                )
                raise
            except Exception:
                logger.exception(f"Fallback LLM model {fallback} also failed")
                raise
        except asyncio.TimeoutError as e:
            # Request timed out
            logger.warning(f"Primary LLM Timeout for model {primary}: {e}")
            logger.info(f"Retrying LLM call with fallback model {fallback}")
            try:
                response = await litellm.acompletion(
                    model=fallback,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=timeout,
                    api_key=self._groq_api_key,
                )
                used_label = "fallback"
                used_model_name = fallback
            except (litellm.exceptions.APIError, asyncio.TimeoutError) as e2:
                logger.exception(
                    f"Fallback LLM model {fallback} also failed with API/Timeout: {e2}"
                )
                raise
            except Exception:
                logger.exception(f"Fallback LLM model {fallback} also failed")
                raise
        except Exception as e:
            # Unknown/other exceptions
            logger.error(f"Primary LLM unexpected error for model {primary}: {e}")
            logger.info(f"Retrying LLM call with fallback model {fallback}")
            try:
                response = await litellm.acompletion(
                    model=fallback,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=timeout,
                    api_key=self._groq_api_key,
                )
                used_label = "fallback"
                used_model_name = fallback
            except Exception:
                logger.exception(f"Fallback LLM model {fallback} also failed")
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
