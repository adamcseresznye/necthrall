import logging
from typing import Optional, List
from deepeval.models.base_model import DeepEvalBaseLLM
import litellm
from config import config

logger = logging.getLogger(__name__)


class LLMJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.model_name = config.SYNTHESIS_MODEL
        self.fallback_model_name = config.SYNTHESIS_FALLBACK
        self.primary_api_key = config.PRIMARY_LLM_API_KEY
        self.secondary_api_key = config.SECONDARY_LLM_API_KEY

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.primary_api_key,
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(
                f"Primary model {self.model_name} failed: {e}. Trying fallback."
            )
            if self.fallback_model_name:
                try:
                    response = litellm.completion(
                        model=self.fallback_model_name,
                        messages=[{"role": "user", "content": prompt}],
                        api_key=self.secondary_api_key,
                        temperature=0.1,
                    )
                    return response.choices[0].message.content
                except Exception as fallback_e:
                    logger.error(
                        f"Fallback model {self.fallback_model_name} also failed: {fallback_e}"
                    )
                    raise fallback_e
            else:
                raise e

    async def a_generate(self, prompt: str) -> str:
        try:
            response = await litellm.acompletion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.primary_api_key,
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(
                f"Primary model {self.model_name} failed: {e}. Trying fallback."
            )
            if self.fallback_model_name:
                try:
                    response = await litellm.acompletion(
                        model=self.fallback_model_name,
                        messages=[{"role": "user", "content": prompt}],
                        api_key=self.secondary_api_key,
                        temperature=0.1,
                    )
                    return response.choices[0].message.content
                except Exception as fallback_e:
                    logger.error(
                        f"Fallback model {self.fallback_model_name} also failed: {fallback_e}"
                    )
                    raise fallback_e
            else:
                raise e

    def get_model_name(self):
        return self.model_name
