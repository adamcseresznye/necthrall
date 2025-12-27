from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Paper(BaseModel):
    paperId: str
    title: str
    abstract: Optional[str] = None
    year: Optional[int] = None
    citationCount: int = 0
    influentialCitationCount: int = 0
    openAccessPdf: Optional[Dict[str, Any]] = None
    externalIds: Optional[Dict[str, Any]] = None
    url: Optional[str] = None
    venue: Optional[str] = None
    embedding: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class Passage(BaseModel):
    paper_id: str
    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None


class State(BaseModel):
    """Progressive-enrichment State used by the Necthrall pipeline.

    The State is intentionally lean and uses Optional fields so agents can
    populate it step-by-step (progressive enrichment). Field-by-field
    assignment is validated (see model_config) and helper methods provide
    ergonomic updates and error tracking.

    Fields are small and serializable (dicts/lists/primitives) to keep the
    model lightweight for tests and local runs.
    """

    # Core identification
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Query optimization (Day 4-5)
    optimized_query: Optional[str] = None
    ss_variants: Optional[Dict[str, str]] = None  # {primary, broad, alternative}

    # Paper retrieval (Day 3)
    papers: List[Paper] = Field(default_factory=list)  # Raw Semantic Scholar hits

    # Quality and ranking (Day 4-5)
    quality_gate: Optional[Dict[str, Any]] = None
    ranked_papers: List[Paper] = Field(default_factory=list)
    finalists: List[Paper] = Field(default_factory=list)  # Top 5-10

    passages: List[Passage] = Field(default_factory=list)

    # Chunks produced by processing agents (LlamaIndex Documents / node objects)
    chunks: Optional[List[Any]] = None

    answer: Optional[str] = None
    citations: List[int] = Field(default_factory=list)

    # Error tracking
    errors: List[str] = Field(default_factory=list)

    # Pydantic v2 config style
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    def __init__(self, **data: Any):
        # Keep simple: run BaseModel init then log creation
        super().__init__(**data)
        logger.debug(
            "State initialized: {request_id} query={query}",
            request_id=self.request_id,
            query=self.query,
        )

    def update_fields(self, **kwargs: Any) -> None:
        """Update fields in-place with validation.

        Example: state.update_fields(optimized_query="...")
        This performs per-field validation thanks to validate_assignment.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"State has no field '{k}'")
            setattr(self, k, v)
            logger.debug(
                "State field updated: {field} -> {value}", field=k, value=repr(v)
            )

    def append_error(self, message: str) -> None:
        """Append an error message to the errors list and log it."""
        self.errors.append(message)
        logger.debug("State error appended: {err}", err=message)

    def to_public(self) -> Dict[str, Any]:
        """Return a public-appropriate dict (shallow) useful for APIs/logging."""
        # Keep potentially large fields; callers can prune if needed
        return self.model_dump()


__all__ = ["State", "Paper", "Passage"]
