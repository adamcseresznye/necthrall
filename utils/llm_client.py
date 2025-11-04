from loguru import logger
import json
import time
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

# CRITICAL: Import torch BEFORE langchain providers to avoid DLL initialization errors on Windows
# See: https://github.com/pytorch/pytorch/issues/91966
import torch

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from google.api_core.exceptions import ResourceExhausted

# Load environment variables from .env file
load_dotenv()

# Configure structured JSON logging
logger.add("file.log", format="{message}", serialize=True)


class LLMClient:
    """
    A client for interacting with LLMs, with automatic failover.

    This client uses ChatGoogleGenerativeAI (Gemini) as the primary provider
    and ChatGroq (Llama) as a fallback. If the primary provider fails,
    it automatically retries with the fallback provider.

    Usage example:
        client = LLMClient()
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        response = client.generate(messages)
        print(response)
    """

    def __init__(self, temperature: float = 0.3, max_tokens: int = 2000):
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.primary_llm = ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL_PRIMARY"),
            max_retries=2,
            timeout=30,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        self.fallback_llm = ChatGroq(
            model=os.getenv("LLM_MODEL_FALLBACK"),
            max_retries=2,
            timeout=30,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def _parse_messages(self, messages: List[Dict[str, str]]) -> List:
        """Converts a list of dictionaries to a list of LangChain messages."""
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
        return langchain_messages

    def _format_response(self, response: AIMessage, model_name: str, llm) -> Dict:
        """Formats the LLM response into the specified dictionary format."""
        content = response.content
        try:
            tokens_used = llm.get_num_tokens(content)
        except Exception:
            tokens_used = -1
        return {
            "content": content,
            "model_used": model_name,
            "tokens_used": tokens_used,
        }

    def _invoke_llm(
        self, llm, messages, provider_name, temperature, max_tokens
    ) -> Dict:
        """Invokes the LLM with logging and error handling."""
        try:
            logger.info(
                json.dumps({"event": "llm_call_start", "provider": provider_name})
            )
            start_time = time.time()
            response = llm.invoke(
                messages, temperature=temperature, max_tokens=max_tokens
            )
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(
                json.dumps(
                    {
                        "event": "llm_call_success",
                        "provider": provider_name,
                        "execution_time": execution_time,
                    }
                )
            )
            model_name = getattr(llm, "model", getattr(llm, "model_name", "unknown"))
            return self._format_response(response, f"{provider_name}/{model_name}", llm)
        except ResourceExhausted as e:
            logger.error(
                json.dumps(
                    {
                        "event": "rate_limit_error",
                        "provider": provider_name,
                        "error": str(e),
                    }
                )
            )
            logger.warning(
                f"Rate limit exceeded for {provider_name}. Suggest switching API keys or waiting."
            )
            raise e
        except Exception as e:
            logger.warning(
                json.dumps(
                    {
                        "event": "llm_call_failure",
                        "provider": provider_name,
                        "error": str(e),
                    }
                )
            )
            raise e

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict:
        """
        Generates a response from the LLM with automatic failover.
        """
        temp = temperature if temperature is not None else self.default_temperature
        max_tok = max_tokens if max_tokens is not None else self.default_max_tokens
        langchain_messages = self._parse_messages(messages)

        try:
            return self._invoke_llm(
                self.primary_llm, langchain_messages, "gemini", temp, max_tok
            )
        except Exception:
            logger.info(
                json.dumps(
                    {"event": "llm_fallback_triggered", "fallback_provider": "groq"}
                )
            )
            try:
                return self._invoke_llm(
                    self.fallback_llm, langchain_messages, "groq", temp, max_tok
                )
            except Exception as e_fallback:
                raise e_fallback

    def test_groq(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict:
        """
        Directly tests the fallback LLM (Groq).
        """
        temp = temperature if temperature is not None else self.default_temperature
        max_tok = max_tokens if max_tokens is not None else self.default_max_tokens
        langchain_messages = self._parse_messages(messages)
        return self._invoke_llm(
            self.fallback_llm, langchain_messages, "groq", temp, max_tok
        )
