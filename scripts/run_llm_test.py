import asyncio
import sys
from pathlib import Path

from utils.llm_router import LLMRouter


async def test():
    router = LLMRouter()
    try:
        result = await router.generate(
            prompt="What is intermittent fasting?", model_type="optimization"
        )
        print("Response:", repr(result)[:500])
    except Exception as e:
        print("LLM call failed:", type(e).__name__, str(e))


if __name__ == "__main__":
    asyncio.run(test())
