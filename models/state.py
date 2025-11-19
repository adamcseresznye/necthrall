from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from uuid import uuid4
from loguru import logger


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
    papers: Optional[List[Dict[str, Any]]] = None  # Raw Semantic Scholar hits

    # Quality and ranking (Day 4-5)
    quality_gate: Optional[Dict[str, Any]] = None
    ranked_papers: Optional[List[Dict[str, Any]]] = None
    finalists: Optional[List[Dict[str, Any]]] = None  # Top 5-10

    # Passages (Week 2)
    passages: Optional[List[Dict[str, Any]]] = None

    # Chunks produced by processing agents (LlamaIndex Documents / node objects)
    chunks: Optional[List[Any]] = None

    # Answer and citations (Week 3)
    answer: Optional[str] = None
    citations: Optional[List[Dict[str, Any]]] = None

    # Error tracking
    errors: List[str] = Field(default_factory=list)

    # Pydantic v2 config style
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
    }

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


__all__ = ["State"]
