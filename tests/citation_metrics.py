import asyncio
import json
import re
from typing import Any, Dict, List, Optional
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from utils.citation_verifier import CitationVerifier
from tests.eval_config import LLMJudge


class CitationValidityMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.verifier = CitationVerifier()
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase):
        # Extract actual output and retrieval context
        actual_output = test_case.actual_output
        retrieval_context = test_case.retrieval_context

        # Verify citations
        result = self.verifier.verify(actual_output, retrieval_context)

        # Set score and reason
        self.score = 1.0 if result["valid"] else 0.0
        self.reason = result["reason"]

        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Citation Validity"


class CitationAccuracyMetric(BaseMetric):
    def __init__(self, model: LLMJudge, threshold: float = 0.7):
        self.threshold = threshold
        self.model = model
        self.score = 0.0
        self.reason = ""

    async def a_measure(self, test_case: LLMTestCase):
        actual_output = test_case.actual_output
        retrieval_context = test_case.retrieval_context

        # Format context for the prompt
        context_str = ""
        for i, ctx in enumerate(retrieval_context):
            context_str += f"Passage {i+1}:\n{ctx}\n\n"

        prompt = f"""
        You are a strict citation evaluator. Your task is to verify if the citations in the provided text are factually supported by the corresponding passages.

        Text to Evaluate:
        {actual_output}

        Reference Passages:
        {context_str}

        Instructions:
        1. Identify every statement in the text that has a citation (e.g., [1], [2]).
        2. For each citation, check if the referenced Passage (Passage 1 for [1], etc.) explicitly supports the statement.
        3. Calculate a score: (Number of Supported Citations / Total Number of Citations). If there are no citations, return 1.0 (N/A).
        4. Provide a reason explaining which citations were correct and which were incorrect.

        Output must be valid JSON with the following format:
        {{
            "score": <float between 0.0 and 1.0>,
            "reason": "<explanation>"
        }}
        """

        response = await self.model.a_generate(prompt)

        # Parse JSON response
        try:
            # Clean up potential markdown code blocks
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            data = json.loads(cleaned_response.strip())
            self.score = float(data.get("score", 0.0))
            self.reason = data.get("reason", "No reason provided.")
        except Exception as e:
            self.score = 0.0
            self.reason = f"Failed to parse LLM response: {e}. Response was: {response}"

        self.success = self.score >= self.threshold
        return self.score

    def measure(self, test_case: LLMTestCase):
        return asyncio.run(self.a_measure(test_case))

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Citation Accuracy"
