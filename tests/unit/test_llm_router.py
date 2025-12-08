import pytest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

from loguru import logger

from utils.llm_router import LLMRouter


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_happy_path():
    """generate should call litellm.acompletion with the primary model and return content"""
    primary_model = "primary_model"
    fallback_model = "fallback_model"

    # Mock configuration
    with (
        patch("config.config.QUERY_OPTIMIZATION_MODEL", primary_model),
        patch("config.config.QUERY_OPTIMIZATION_FALLBACK", fallback_model),
        patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
    ):

        # Mock response object
        response_obj = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        mock_acompletion.return_value = response_obj

        router = LLMRouter()
        result = await router.generate("test prompt", "optimization")

        assert result == "ok"
        # assert called once with primary model
        assert mock_acompletion.call_count == 1
        called_kwargs = mock_acompletion.call_args.kwargs
        assert called_kwargs.get("model") == primary_model
        assert called_kwargs.get("timeout") == 30


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_fallback_path():
    """If primary fails, router should retry with fallback and return content"""
    primary_model = "primary_model"
    fallback_model = "fallback_model"

    with (
        patch("config.config.QUERY_OPTIMIZATION_MODEL", primary_model),
        patch("config.config.QUERY_OPTIMIZATION_FALLBACK", fallback_model),
        patch("config.config.PRIMARY_LLM_API_KEY", "fake_google_key"),
        patch("config.config.SECONDARY_LLM_API_KEY", "fake_groq_key"),
        patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
    ):

        # Track call count to alternate between raising exception and returning
        call_count = 0

        # Configure side effects: raise for primary, return content for fallback
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call (primary) should fail
                raise Exception("primary failed")
            else:
                # Second call (fallback) should succeed
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content="fallback-ok"))
                    ]
                )

        mock_acompletion.side_effect = side_effect

        router = LLMRouter()
        result = await router.generate("test prompt", "optimization")

        assert result == "fallback-ok"
        assert mock_acompletion.call_count == 2
        assert mock_acompletion.call_args_list[0].kwargs.get("model") == primary_model
        assert mock_acompletion.call_args_list[1].kwargs.get("model") == fallback_model


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_both_fail():
    """If both primary and fallback fail, router should raise the exception"""
    primary_model = "primary_model"
    fallback_model = "fallback_model"

    with (
        patch("config.config.QUERY_OPTIMIZATION_MODEL", primary_model),
        patch("config.config.QUERY_OPTIMIZATION_FALLBACK", fallback_model),
        patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
    ):

        mock_acompletion.side_effect = Exception("both fail")

        router = LLMRouter()

        with pytest.raises(Exception):
            await router.generate("test prompt", "optimization")

        assert mock_acompletion.call_count == 2
