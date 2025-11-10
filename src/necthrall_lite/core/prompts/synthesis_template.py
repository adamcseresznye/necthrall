"""
Synthesis prompt template for zero-hallucination citation verification.

This module provides a LangChain PromptTemplate for synthesizing scientific information
with strict citation grounding and contradiction handling.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
import re


class Citation(BaseModel):
    """Citation metadata for synthesized answers."""

    index: int = Field(..., ge=1, description="Citation number in the answer")
    paper_id: str = Field(..., description="ID of the cited paper")
    text: str = Field(..., max_length=500, description="Cited text excerpt")
    credibility_score: Optional[int] = Field(
        None, ge=0, le=100, description="Paper credibility score"
    )


class SynthesisOutput(BaseModel):
    """Output from the synthesis agent."""

    answer: str = Field(
        ...,
        min_length=50,
        max_length=2500,
        description="Synthesized answer with citations",
    )
    citations: List[Citation] = Field(..., description="Citation metadata")
    consensus_estimate: Optional[str] = Field(
        None, description="Consensus level among sources"
    )


def validate_passage_numbering(formatted_passages: str) -> None:
    """
    Validate that passages are numbered consecutively from 1 without duplicates.

    Args:
        formatted_passages: String containing passages with [N] numbering

    Raises:
        ValueError: If numbering is invalid
    """
    numbers = re.findall(r"\[(\d+)\]", formatted_passages)
    if not numbers:
        raise ValueError("No passage numbering found in formatted_passages")

    unique_nums = sorted(set(int(n) for n in numbers))
    if len(numbers) != len(unique_nums):
        raise ValueError("Passage numbering contains duplicates")

    expected = list(range(1, len(unique_nums) + 1))
    if unique_nums != expected:
        raise ValueError(
            f"Passage numbering must be consecutive from 1 to {len(unique_nums)} without duplicates. "
            f"Found: {unique_nums}"
        )


def create_synthesis_prompt(
    query: str, formatted_passages: str, contradiction_context: str
) -> PromptTemplate:
    """
    Create a LangChain PromptTemplate for scientific synthesis with citation verification.

    Args:
        query: The scientific question to answer
        formatted_passages: Numbered passages string with [1], [2], etc.
        contradiction_context: Description of detected contradictions

    Returns:
        PromptTemplate configured for synthesis with output parsing

    Raises:
        ValueError: If passage numbering is invalid
    """
    # Handle missing passages
    if not formatted_passages.strip():
        formatted_passages = (
            "[1] No passages provided. Please provide evidence for analysis."
        )

    # Validate passage numbering
    try:
        validate_passage_numbering(formatted_passages)
    except ValueError as e:
        # For now, raise; could fallback to simplified prompt in future
        raise e

    # Create output parser
    parser = PydanticOutputParser(pydantic_object=SynthesisOutput)

    # Get format instructions
    format_instructions = parser.get_format_instructions()

    # System prompt with role, rules, and few-shot examples
    # Prompt engineering: Clear rules prevent hallucination, few-shot shows expected format
    system_prompt = """You are a scientific research assistant specializing in citation verification and evidence synthesis.

CRITICAL RULES - FOLLOW THESE STRICTLY:
1. Every factual claim MUST have a citation [N] referencing exactly passage N from the provided passages.
2. NEVER generate citations that don't exist in the provided passages - this creates hallucinations.
3. If evidence contradicts, present BOTH viewpoints with credibility context from the passages.
4. Generate consensus estimates based on agreement patterns: "High consensus", "Moderate consensus", "Low consensus", or "Insufficient evidence".
5. Refuse speculation - only use information from provided passages.
6. Structure your answer with clear sections: introduction, body, contradictions (if any), conclusion.

FEW-SHOT EXAMPLE:

Query: What is the effect of exercise on blood pressure?

Passages:
[1] High-credibility study: Regular aerobic exercise reduces systolic blood pressure by 5-10 mmHg in hypertensive patients.
[2] Medium-credibility study: Resistance training shows no significant effect on blood pressure.

Contradictions: Studies disagree on exercise type effectiveness.

Expected JSON Output:
{{
  "answer": "Exercise can influence blood pressure through different mechanisms. Regular aerobic exercise has been shown to reduce systolic blood pressure by 5-10 mmHg in hypertensive patients [1]. However, resistance training appears to have no significant effect on blood pressure [2].\\n\\nWhile aerobic exercise demonstrates clear benefits, the evidence for resistance training is inconclusive. Consider consulting healthcare providers for personalized recommendations.",
  "citations": [
    {{"index": 1, "paper_id": "study_a", "text": "Regular aerobic exercise reduces systolic blood pressure by 5-10 mmHg", "credibility_score": 85}},
    {{"index": 2, "paper_id": "study_b", "text": "Resistance training shows no significant effect", "credibility_score": 60}}
  ],
  "consensus_estimate": "Moderate consensus"
}}

REMEMBER: Only cite passages that exist. If no relevant information, state "Insufficient evidence"."""

    # Human prompt template
    # Uses f-string style but with LangChain variables for flexibility
    human_template = f"""
Scientific Question: {{query}}

Provided Passages:
{{formatted_passages}}

Contradiction Context: {{contradiction_context}}

{{format_instructions}}

Answer the question using ONLY the provided passages. Every claim must be cited [N]."""

    # Combine prompts (system context + human template)
    full_template = system_prompt + "\n\n" + human_template

    # Create PromptTemplate with partial variables for format instructions
    prompt = PromptTemplate(
        template=full_template,
        input_variables=["query", "formatted_passages", "contradiction_context"],
        partial_variables={"format_instructions": format_instructions},
    )

    return prompt
