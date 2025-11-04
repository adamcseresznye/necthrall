from pydantic import BaseModel, Field
from typing import List, Optional


class Citation(BaseModel):
    """Citation metadata for synthesized answers.

    Fields:
        index: int - The citation number [N] in the answer
        paper_id: str - ID of the cited paper
        text: str - The specific text excerpt being cited
        credibility_score: Optional[int] - Credibility score of the paper (0-100)
    """

    index: int = Field(..., ge=1, description="Citation number in the answer")
    paper_id: str = Field(..., description="ID of the cited paper")
    text: str = Field(..., max_length=500, description="Cited text excerpt")
    credibility_score: Optional[int] = Field(
        None, ge=0, le=100, description="Paper credibility score"
    )


class SynthesisOutput(BaseModel):
    """Output from the synthesis agent.

    Fields:
        answer: str - Synthesized answer with inline [N] citations
        citations: List[Citation] - Metadata for each citation
        consensus_estimate: Optional[str] - Agreement level among sources
    """

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


class CitationValidation(BaseModel):
    """Validation result for citations in synthesized answers.

    Fields:
        total_citations: int - Total unique citations found in the answer
        valid_citations: int - Number of citations within valid range
        invalid_citations: List[int] - List of out-of-range citation numbers
        validation_passed: bool - Whether all citations are valid
        error_details: List[str] - Specific error messages for invalid citations
    """

    total_citations: int = Field(..., ge=0, description="Total unique citations found")
    valid_citations: int = Field(..., ge=0, description="Number of valid citations")
    invalid_citations: List[int] = Field(
        ..., description="Out-of-range citation numbers"
    )
    validation_passed: bool = Field(..., description="Overall validation result")
    error_details: List[str] = Field(..., description="Specific error messages")
