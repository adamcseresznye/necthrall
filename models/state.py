from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ConfigDict,
    model_validator,
    ValidationError,
)
from typing import List, Dict, Any, Optional, Union, Iterator, Iterable
import uuid
from datetime import datetime, timedelta
from enum import Enum
import re
import ormsgpack
import weakref
from functools import cached_property
from loguru import logger
import json
from typing import Callable, Any


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


class LazyList:
    """
    A lazy-loading list that only loads items when accessed.
    Useful for large collections that don't need to be fully loaded into memory.
    """

    def __init__(
        self, items: Optional[Iterable[Any]] = None, loader: Optional[callable] = None
    ):
        self._items = list(items) if items else []
        self._loader = loader
        self._loaded = items is not None

    def _ensure_loaded(self) -> None:
        """Load items if not already loaded and loader is available."""
        if not self._loaded and self._loader:
            self._items = list(self._loader())
            self._loaded = True

    def __getitem__(self, key):
        self._ensure_loaded()
        return self._items[key]

    def __setitem__(self, key, value):
        self._ensure_loaded()
        self._items[key] = value

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._items)

    def __iter__(self) -> Iterator[Any]:
        self._ensure_loaded()
        return iter(self._items)

    def __contains__(self, item) -> bool:
        self._ensure_loaded()
        return item in self._items

    def append(self, item):
        self._ensure_loaded()
        self._items.append(item)

    def extend(self, items):
        self._ensure_loaded()
        self._items.extend(items)

    def clear(self):
        self._items.clear()
        self._loaded = True

    def to_list(self) -> List[Any]:
        """Convert to a regular Python list."""
        self._ensure_loaded()
        return self._items.copy()


class ObjectPool:
    """
    Memory pool for frequently allocated objects to reduce GC pressure.
    """

    def __init__(self, factory: callable, max_size: int = 1000):
        self._factory = factory
        self._pool = []
        self._max_size = max_size
        self._weak_refs = weakref.WeakSet()

    def acquire(self, *args, **kwargs):
        """Get an object from the pool or create new one."""
        if self._pool:
            obj = self._pool.pop()
            # Reset object state if it has a reset method
            if hasattr(obj, "reset"):
                obj.reset()
            return obj
        return self._factory(*args, **kwargs)

    def release(self, obj):
        """Return object to pool if space available."""
        if len(self._pool) < self._max_size:
            self._pool.append(obj)

    def clear(self):
        """Clear the pool."""
        self._pool.clear()


# Global object pools for common Pydantic models
_chunk_pool = ObjectPool(lambda: Chunk(paper_id="", content=""))
_passage_pool = ObjectPool(
    lambda: Passage(content="", paper_id="", retrieval_score=0.0)
)
_paper_pool = ObjectPool(
    lambda: Paper(paper_id="", title="", authors=[], type="article")
)


class ProcessingStatus(str, Enum):
    """Enum for tracking processing pipeline status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ProcessingConfig(BaseModel):
    """Configuration for the ProcessingAgent pipeline."""

    chunk_size: int = Field(default=500, ge=100, le=1000)
    chunk_overlap: int = Field(default=50, ge=0, le=200)
    embedding_model: str = "all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = Field(default=20, ge=5, le=50)
    final_k: int = Field(default=10, ge=1, le=20)
    max_chunk_tokens: int = Field(default=800, ge=100, le=1500)
    enable_reranking: bool = True
    preserve_section_boundaries: bool = True
    batch_size: int = Field(default=32, ge=1, le=64)


class RetrievalScores(BaseModel):
    """Enhanced scoring structure for hybrid retrieval with BM25, semantic, RRF, and reranking scores."""

    bm25_score: Optional[float] = Field(None, description="BM25 keyword matching score")
    semantic_score: Optional[float] = Field(
        None, description="Semantic similarity score"
    )
    rrf_score: Optional[float] = Field(None, description="Reciprocal rank fusion score")
    reranking_score: Optional[float] = Field(None, description="Final reranking score")
    final_rank: int = Field(default=0, ge=0, description="Final ranking position")

    @field_validator(
        "bm25_score",
        "semantic_score",
        "rrf_score",
        "reranking_score",
    )
    @classmethod
    def validate_scores(cls, v):
        """Validate scores are in valid ranges with bounds checking."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("All scores must be between 0.0 and 1.0")
        return v


class Chunk(BaseModel):
    """Represents a processed document chunk with backward compatibility and enhanced features."""

    # Core identification - make paper_id optional to support direct field creation
    paper_id: str = Field(..., description="ID of the paper this chunk belongs to")
    content: Optional[str] = Field(None, description="Text content of the chunk")

    # Section information
    section: str = Field(
        default="unknown", description="Section of paper this chunk belongs to"
    )

    # Legacy fields (backward compatibility with existing codebase)
    paper_title: Optional[str] = Field(None, description="Paper title (legacy field)")
    start_position: Optional[int] = Field(
        None, description="Starting character position (legacy field)"
    )
    end_position: Optional[int] = Field(
        None, description="Ending character position (legacy field)"
    )
    use_fallback: bool = Field(
        default=False, description="Whether fallback chunking was used"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )

    # Enhanced fields (for new ProcessingAgent interface)
    chunk_id: Optional[str] = Field(None, description="Unique identifier for the chunk")
    text: Optional[str] = Field(
        None, description="The actual text content (alias for content)"
    )
    chunk_index: Optional[int] = Field(
        None, ge=0, description="Ordinal position of chunk within paper"
    )
    char_start: Optional[int] = Field(
        None, ge=0, description="Starting character position in original text"
    )
    char_end: Optional[int] = Field(
        None, ge=0, description="Ending character position in original text"
    )
    token_count: int = Field(
        default=0, ge=0, description="Number of tokens in the chunk"
    )

    # Enable aliases to support both old and new field names
    model_config = ConfigDict(populate_by_name=True)

    @field_validator("text")
    @classmethod
    def sync_text_content(cls, v: Optional[str], values) -> Optional[str]:
        """Synchronize text field with content field for backward compatibility."""
        # If text is being set, also set content for backward compatibility
        if v is not None:
            values.data["content"] = v
        return v

    @field_validator("content")
    @classmethod
    def sync_content_text(cls, v: Optional[str], values) -> Optional[str]:
        """Synchronize content field with text field for backward compatibility."""
        # If content is set but text is not, set text also
        if v is not None and values.data.get("text") is None:
            values.data["text"] = v
        return v

    @field_validator("char_end")
    @classmethod
    def validate_char_end(cls, v: Optional[int], values) -> Optional[int]:
        """Ensure char_end is greater than or equal to char_start if both are provided."""
        if v is not None:
            char_start = values.data.get("char_start")
            if char_start is not None and v < char_start:
                raise ValueError("char_end must be >= char_start")
        return v

    @field_validator("end_position")
    @classmethod
    def validate_end_position(cls, v: Optional[int], values) -> Optional[int]:
        """Ensure end_position is greater than or equal to start_position if both are provided."""
        if v is not None:
            start_position = values.data.get("start_position")
            if start_position is not None and v < start_position:
                raise ValueError("end_position must be >= start_position")
        return v

    @property
    def text_content(self) -> str:
        """Get text content, preferring text field but falling back to content."""
        return self.text or self.content or ""


class ProcessingMetadata(BaseModel):
    """Enhanced processing metadata with detailed performance tracking."""

    total_papers: int = 0
    processed_papers: int = 0
    skipped_papers: int = 0
    total_sections: int = 0
    total_chunks: int = 0
    chunks_embedded: int = 0
    retrieval_candidates: int = 0
    reranked_passages: int = 0
    fallback_used_count: int = 0

    stage_times: Dict[str, float] = Field(default_factory=dict)
    total_time: float = 0.0

    paper_errors: List[Dict[str, Any]] = Field(default_factory=list)
    processing_errors: List[str] = Field(default_factory=list)

    memory_usage_mb: Optional[float] = None
    throughput_chunks_per_second: Optional[float] = None


class Paper(BaseModel):
    """Scientific paper metadata from OpenAlex"""

    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    journal: Optional[str]
    citation_count: int = 0
    doi: Optional[str] = None
    abstract: Optional[str] = None  # May be None if not available
    pdf_url: Optional[str]
    type: str = Field(..., description="Paper type: 'review' or 'article'")

    @field_validator("type")
    @classmethod
    def validate_paper_type(cls, v: str) -> str:
        """Validate paper type is either 'review' or 'article'"""
        if v not in ["review", "article"]:
            raise ValueError("Paper type must be either 'review' or 'article'")
        return v


class Passage(BaseModel):
    """Represents a scored passage retrieved from a paper with enhanced scoring and provenance metadata."""

    content: str = Field(..., description="The actual text content of the passage")
    section: str = Field(
        default="other", description="Section of paper this passage belongs to"
    )
    paper_id: str = Field(..., description="ID of the paper this passage belongs to")
    chunk_id: Optional[str] = Field(
        None, description="ID of the chunk this passage comes from"
    )
    scores: RetrievalScores = Field(
        default_factory=RetrievalScores,
        description="Comprehensive retrieval scoring information",
    )
    retrieval_score: float = Field(
        ..., ge=0.0, le=1.0, description="Legacy field - primary retrieval score"
    )
    cross_encoder_score: Optional[float] = Field(
        None,
        description="Cross-encoder reranking score (raw model outputs, can be negative)",
    )
    final_score: Optional[float] = Field(
        None,
        description="Final combined relevance score (can be denormalized for ranking)",
    )
    char_start: Optional[int] = Field(
        None, ge=0, description="Character start position in original document"
    )
    char_end: Optional[int] = Field(
        None, ge=0, description="Character end position in original document"
    )

    @field_validator("section")
    @classmethod
    def validate_section(cls, v: str) -> str:
        """Normalize section to allowed values, mapping unknowns to 'other' with warning."""
        allowed_sections = {"introduction", "methods", "results", "discussion", "other"}
        if v not in allowed_sections:
            logger.warning(f"Unknown section '{v}' mapped to 'other'")
            return "other"
        return v

    @field_validator("char_end")
    @classmethod
    def validate_char_positions(cls, v: Optional[int], values) -> Optional[int]:
        """Ensure char_end >= char_start if both are provided."""
        if v is not None:
            char_start = values.data.get("char_start")
            if char_start is not None and v < char_start:
                raise ValueError("char_end must be >= char_start")
        return v

    @property
    def bm25_score(self) -> Optional[float]:
        """Convenience property for BM25 score from scores object."""
        return self.scores.bm25_score

    @property
    def semantic_score(self) -> Optional[float]:
        """Convenience property for semantic score from scores object."""
        return self.scores.semantic_score

    @property
    def rrf_score(self) -> Optional[float]:
        """Convenience property for RRF score from scores object."""
        return self.scores.rrf_score

    @property
    def reranking_score(self) -> Optional[float]:
        """Convenience property for reranking score from scores object."""
        return self.scores.reranking_score


class Score(BaseModel):
    """Represents a scoring metric for papers or passages."""

    score_type: str  # e.g., "relevance", "novelty", "quality"
    value: float
    justification: Optional[str] = None
    confidence: Optional[float] = None


class CredibilityScore(BaseModel):
    """Credibility score for a paper.

    Fields:
        paper_id: str
        score: int (0-100)
        tier: str ("high", "medium", "low")
        rationale: str (concise explanation, max 100 chars, <=15 words)

    Example:
        CredibilityScore(paper_id="p1", score=85, tier="high", rationale="high citations (120), recent (2023), top-tier journal")
    """

    paper_id: str
    score: int = Field(..., ge=0, le=100)
    tier: str = Field(..., pattern="^(high|medium|low)$")
    rationale: str = Field(..., max_length=100)

    @field_validator("rationale")
    @classmethod
    def validate_rationale_word_count(cls, v: str) -> str:
        """Ensure rationale is short (<=15 words)."""
        words = [w for w in v.strip().split() if w]
        if len(words) > 15:
            # Truncate to 15 words preserving basic meaning
            return " ".join(words[:15])
        return v


class ContradictionClaim(BaseModel):
    """A claim in a contradiction detection result.

    Fields:
        paper_id: str - ID of the paper containing the claim
        text: str - The claim text (max 150 characters)
    """

    paper_id: str
    text: str = Field(..., max_length=150)


class DetectedContradiction(BaseModel):
    """A detected contradiction between two claims.

    Fields:
        topic: str - Brief description of the disagreement area (max 50 chars)
        claim_1: ContradictionClaim - First conflicting claim
        claim_2: ContradictionClaim - Second conflicting claim
        severity: str - "major" for direct opposition, "minor" for nuanced disagreement
    """

    topic: str = Field(..., max_length=50)
    claim_1: ContradictionClaim
    claim_2: ContradictionClaim
    severity: str = Field(..., pattern="^(major|minor)$")


class DownloadResult(BaseModel):
    paper_id: str
    success: bool
    content: Optional[bytes] = None
    error: Optional[str] = None
    download_time: float = 0.0
    file_size: Optional[int] = None


class PDFContent(BaseModel):
    paper_id: str
    raw_text: str
    page_count: int
    char_count: int
    extraction_time: float


class ErrorReport(BaseModel):
    """Detailed report for a single failure."""

    paper_id: str
    url: str
    error_type: str
    message: str
    timestamp: float
    recoverable: bool


class AcquisitionMetrics(BaseModel):
    """Metrics for the entire acquisition process."""

    total_papers: int
    successful_downloads: int
    failed_downloads: int
    extraction_failures: int
    total_time: float
    avg_download_time: float
    failure_breakdown: Dict[str, int]


class StateLogger:
    """
    Enhanced logging system for State changes with structured JSON output.
    Tracks field modifications, processing stage transitions, and performance metrics.
    """

    def __init__(self, state: "State", log_file: Optional[str] = None):
        self.state = state
        self.log_file = log_file
        self._original_values = {}
        self._transition_history = []

        # Set up logger with structured output
        self.logger = logger.bind(state_id=state.request_id)

        # Configure file sink if specified
        if log_file:
            logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[state_id]} | {message}",
                level="INFO",
                serialize=True,  # JSON serialization
            )

        # Take initial snapshot
        self._snapshot_initial_state()

    def _snapshot_initial_state(self):
        """Take initial snapshot of all important state fields for change tracking."""
        self._original_values = {
            "status": self.state.processing_status.value,
            "total_papers": len(self.state.papers_metadata),
            "filtered_papers": len(self.state.filtered_papers),
            "chunks": len(self.state.chunks),
            "passages": len(self.state.relevant_passages),
        }

    def log_field_change(
        self,
        field_name: str,
        old_value: Any,
        new_value: Any,
        reason: Optional[str] = None,
    ):
        """Log individual field modifications with context."""
        change_data = {
            "field": field_name,
            "old_value": self._serialize_value(old_value),
            "new_value": self._serialize_value(new_value),
            "timestamp": datetime.utcnow().isoformat(),
            "reason": reason,
            "current_status": self.state.processing_status.value,
        }

        if self._is_significant_change(field_name, old_value, new_value):
            self.logger.info("Field modified", **change_data)
            self._log_structured("field_change", change_data)

    def log_processing_transition(
        self,
        from_status: ProcessingStatus,
        to_status: ProcessingStatus,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log processing stage transitions with performance context."""
        transition_data = {
            "from_status": from_status.value,
            "to_status": to_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "duration_seconds": metadata.get("duration", 0.0) if metadata else 0.0,
            "performance_metrics": self._get_current_performance_metrics(),
            "error_count": (
                len(self.state.processing_metadata.processing_errors)
                if self.state.processing_metadata
                else 0
            ),
        }

        self._transition_history.append(transition_data)
        self.logger.info("Processing stage transition", **transition_data)
        self._log_structured("stage_transition", transition_data)

        # Log significant performance metrics
        perf_metrics = self._get_current_performance_metrics()
        if perf_metrics.get("memory_usage_mb", 0) > 500:  # Log if over 500MB
            self.logger.warning("High memory usage detected", **perf_metrics)

    def log_performance_metrics(
        self, metrics: Dict[str, Any], context: str = "periodic"
    ):
        """Log comprehensive performance metrics."""
        perf_data = {
            "context": context,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "current_status": self.state.processing_status.value,
        }

        if metrics.get("throughput_chunks_per_second", 0) > 500:  # High throughput
            self.logger.info("High processing throughput", **perf_data)
        elif metrics.get("memory_usage_mb", 0) > 1000:  # High memory usage
            self.logger.warning("High memory consumption", **perf_data)
        else:
            self.logger.info("Performance metrics", **perf_data)

        self._log_structured("performance", perf_data)

    def log_error(
        self,
        error_type: str,
        message: str,
        recoverable: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Log errors with recovery information."""
        error_data = {
            "error_type": error_type,
            "message": message,
            "recoverable": recoverable,
            "timestamp": datetime.utcnow().isoformat(),
            "current_status": self.state.processing_status.value,
            "context": context or {},
        }

        if recoverable:
            self.logger.warning("Recoverable error", **error_data)
        else:
            self.logger.error("Critical error", **error_data)

        self._log_structured("error", error_data)

    def get_change_summary(self) -> Dict[str, Any]:
        """Generate summary of all changes made to the state."""
        current_values = {
            "status": self.state.processing_status.value,
            "total_papers": len(self.state.papers_metadata),
            "filtered_papers": len(self.state.filtered_papers),
            "chunks": len(self.state.chunks),
            "passages": len(self.state.relevant_passages),
        }

        changes = {}
        for key in current_values:
            if current_values[key] != self._original_values.get(key, 0):
                changes[key] = {
                    "from": self._original_values.get(key, 0),
                    "to": current_values[key],
                    "delta": current_values[key] - self._original_values.get(key, 0),
                }

        return {
            "total_transitions": len(self._transition_history),
            "field_changes": changes,
            "final_status": self.state.processing_status.value,
            "processing_errors": (
                len(self.state.processing_metadata.processing_errors)
                if self.state.processing_metadata
                else 0
            ),
            "validation_errors": len(self.state.validation_errors),
        }

    def _is_significant_change(
        self, field_name: str, old_value: Any, new_value: Any
    ) -> bool:
        """Determine if a field change is significant enough to log."""
        # Always log status changes
        if field_name == "processing_status":
            return True

        # Log collection size changes over certain thresholds
        if field_name in ["chunks", "passages", "papers_metadata", "filtered_papers"]:
            return abs(len(new_value) - len(old_value)) > 10  # Changed by more than 10

        # Log performance metrics changes
        if field_name in ["memory_usage_mb", "throughput_chunks_per_second"]:
            return abs(new_value - old_value) > 50  # Significant change

        return False

    def _get_current_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics from state."""
        if not self.state.processing_metadata:
            return {}

        return {
            "total_time": self.state.processing_metadata.total_time,
            "memory_usage_mb": self.state.processing_metadata.memory_usage_mb,
            "throughput_chunks_per_second": self.state.processing_metadata.throughput_chunks_per_second,
            "chunks_embedded": self.state.processing_metadata.chunks_embedded,
            "retrieval_candidates": self.state.processing_metadata.retrieval_candidates,
            "stage_times": dict(self.state.processing_metadata.stage_times),
        }

    def _serialize_value(self, value: Any) -> Any:
        """Safely serialize values for logging."""
        try:
            if isinstance(value, (int, float, str, bool)):
                return value
            elif isinstance(value, list):
                return f"list[{len(value)}]"
            elif isinstance(value, dict):
                return f"dict[{len(value)}]"
            elif hasattr(value, "value"):  # Enum
                return value.value
            else:
                return str(type(value).__name__)
        except Exception:
            return "<unserializable>"

    def _log_structured(self, event_type: str, data: Dict[str, Any]):
        """Log structured data to file if configured."""
        if self.log_file:
            try:
                log_entry = {
                    "timestamp": data["timestamp"],
                    "event_type": event_type,
                    "state_id": self.state.request_id,
                    "data": data,
                }

                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(log_entry, ensure_ascii=False, default=str) + "\n"
                    )
            except Exception as e:
                self.logger.error(f"Failed to write structured log: {e}")


class State(BaseModel):
    """
    LangGraph State schema for Necthrall Lite MVP.
    Tracks query optimization (Week 1), paper retrieval/filtering, processing outputs (Week 2), and prepares for analysis/synthesis (Week 3).

    Example usage:
        state = State(original_query="What is quantum computing?")
        state.top_passages = [
            Passage(content="Quantum computing uses qubits...", section="introduction", paper_id="123", retrieval_score=0.95)
        ]
    """

    # Core identification
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Query Management - NEW optimized fields
    original_query: str = Field(
        ..., alias="query", description="User's original input query"
    )
    optimized_query: Optional[str] = Field(
        None, description="LLM-optimized query for OpenAlex search"
    )
    # Dual-query support: boolean search for OpenAlex, natural-language for retrieval
    search_query: Optional[str] = Field(
        None, description="Boolean search query to send to OpenAlex (OR-based)"
    )
    retrieval_query: Optional[str] = Field(
        None,
        description="Natural-language retrieval query used for BM25/embedding retrieval",
    )
    refinement_count: int = Field(
        0, ge=0, le=2, description="Number of refinement iterations (max 2)"
    )

    # Paper Retrieval & Filtering - UPDATED and NEW
    papers_metadata: List[Paper] = Field(
        default_factory=list,
        description="Raw papers from SearchAgent (up to 100 before filtering, may contain duplicates from multiple search attempts)",
    )
    filtered_papers: List[Paper] = Field(
        default_factory=list,
        description="Final 50 papers after deduplication and BM25+embedding filtering",
    )

    # Filtering & Deduplication Metrics - NEW
    filtering_scores: Optional[Dict[str, Any]] = Field(
        None,
        description="Filtering stage metrics: BM25 scores, semantic scores, ranking details",
    )
    dedup_stats: Optional[Dict[str, Any]] = Field(
        None,
        description="Deduplication metrics: raw_count, unique_count, duplicates_removed, deduplication_rate",
    )
    search_quality: Optional[Dict[str, Any]] = Field(
        None,
        description="Search quality assessment: passed, reason, paper_count, avg_relevance",
    )

    # Enhanced processing fields
    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Current status of processing pipeline",
    )
    config: ProcessingConfig = Field(
        default_factory=ProcessingConfig,
        description="Processing configuration settings",
    )
    validation_errors: List[str] = Field(default_factory=list)

    # Processing pipeline fields
    passages: List[Passage] = Field(default_factory=list)
    scores: Optional[Dict[str, Any]] = None
    answer: Optional[str] = None
    citations: List[int] = Field(default_factory=list)
    metrics: Optional[Dict[str, float]] = None
    retry_count: int = Field(0, ge=0, le=2)

    # New ProcessingAgent output (enhanced interface)
    chunks: List[Chunk] = Field(
        default_factory=list,
        description="Processed document chunks with metadata from chunking pipeline",
    )
    relevant_passages: List[Passage] = Field(
        default_factory=list,
        description="Top passages with enhanced metadata and comprehensive scoring",
    )
    processing_metadata: Optional[ProcessingMetadata] = Field(
        None, description="Detailed processing metadata with performance metrics"
    )

    # Legacy ProcessingAgent output (for backward compatibility)
    top_passages: List[Passage] = Field(
        default_factory=list,
        description="Top 10 passages from processing pipeline with content, section, paper_id, scores",
    )
    processing_stats: Optional[Dict[str, Any]] = Field(
        None, description="Processing stage timings and counts"
    )

    # Week 3 placeholders
    analysis_results: Optional[Dict[str, Any]] = Field(
        None, description="Placeholder for Week 3 analysis results"
    )
    synthesized_answer: Optional[str] = Field(
        None, description="Synthesized answer with inline citations from analysis phase"
    )
    citations: List[Citation] = Field(
        default_factory=list, description="Citation metadata for the synthesized answer"
    )
    consensus_estimate: Optional[str] = Field(
        None,
        description="Agreement level among sources (High/Moderate/Low consensus or Conflicting evidence)",
    )

    # Analysis results
    credibility_scores: List[CredibilityScore] = Field(
        default_factory=list, description="Credibility scores for filtered papers"
    )
    contradictions: List[DetectedContradiction] = Field(
        default_factory=list, description="Detected contradictions between passages"
    )

    # Analysis error handling and recovery
    analysis_errors: List[str] = Field(
        default_factory=list, description="Errors encountered during analysis phase"
    )
    recovery_actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recovery actions taken during analysis failures",
    )

    # Analysis performance data
    performance_data: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Performance metrics and monitoring data from analysis phase",
    )

    # Execution timing
    execution_times: Dict[str, float] = Field(
        default_factory=dict, description="Execution times for each pipeline stage"
    )

    # AcquisitionAgent output
    pdf_contents: List[PDFContent] = Field(default_factory=list)
    download_failures: List[ErrorReport] = Field(default_factory=list)
    acquisition_metrics: Optional[AcquisitionMetrics] = None
    citation_validation: Optional[Dict[str, Any]] = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        populate_by_name=True,  # Allows both alias and field name assignment
        by_alias=True,  # Serialize with aliases by default for backward compatibility
    )

    @field_validator(
        "original_query", "optimized_query", "search_query", "retrieval_query"
    )
    @classmethod
    def validate_query_length(cls, v: Optional[str]) -> Optional[str]:
        """Query should be 3-256 characters and not just whitespace"""
        if v is not None:
            v = v.strip()
            if len(v) < 3:
                raise ValueError("Query must be at least 3 characters")
            if len(v) > 256:
                raise ValueError("Query must be less than 256 characters")
            if not v:
                raise ValueError("Query cannot be empty or just whitespace")
        return v

    @field_validator("optimized_query")
    @classmethod
    def validate_optimized_query_length(cls, v: Optional[str]) -> Optional[str]:
        """Optimized query should be 20-200 characters (8-15 words)"""
        if v and (len(v) < 20 or len(v) > 200):
            raise ValueError("Optimized query must be 20-200 characters")
        return v

    @field_validator("filtered_papers")
    @classmethod
    def validate_filtered_papers_count(cls, v: List[Paper]) -> List[Paper]:
        """Filtered papers should be ≤50 (or less if insufficient results)"""
        if len(v) > 50:
            raise ValueError(f"Filtered papers should be ≤50, got {len(v)}")
        return v

    @field_validator("filtering_scores")
    @classmethod
    def validate_filtering_scores_structure(
        cls, v: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Validate filtering_scores has expected structure"""
        if v is not None:
            expected_keys = {
                "bm25_top_50",
                "semantic_top_50",
                "avg_bm25_score",
                "avg_semantic_score",
            }
            if not any(key in v for key in expected_keys):
                raise ValueError(
                    "filtering_scores must contain at least one of: bm25_top_50, semantic_top_25, avg_bm25_score, avg_semantic_score"
                )
        return v

    @field_validator("dedup_stats")
    @classmethod
    def validate_dedup_stats_structure(
        cls, v: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Validate dedup_stats has expected structure"""
        if v is not None:
            expected_keys = {"raw_count", "unique_count", "duplicates_removed"}
            if not all(key in v for key in expected_keys):
                raise ValueError(
                    "dedup_stats must contain: raw_count, unique_count, duplicates_removed"
                )
            if v.get("unique_count", 0) > v.get("raw_count", 0):
                raise ValueError("unique_count cannot be greater than raw_count")
        return v

    @field_validator("search_quality")
    @classmethod
    def validate_search_quality_structure(
        cls, v: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Validate search_quality has expected structure"""
        if v is not None:
            required_keys = {"passed", "reason", "paper_count", "avg_relevance"}
            if not all(key in v for key in required_keys):
                raise ValueError(
                    "search_quality must contain: passed, reason, paper_count, avg_relevance"
                )
            if not isinstance(v.get("passed"), bool):
                raise ValueError("search_quality.passed must be boolean")
            if not (0 <= v.get("avg_relevance", 0) <= 1):
                raise ValueError("search_quality.avg_relevance must be between 0 and 1")
        return v

    @field_validator("top_passages")
    @classmethod
    def validate_top_passages_length(cls, v: List[Passage]) -> List[Passage]:
        """Top passages should be ≤10"""
        if len(v) > 10:
            raise ValueError(f"Top passages should be ≤10, got {len(v)}")
        return v

    @field_validator("created_at")
    @classmethod
    def validate_timestamp_reasonable(cls, v: datetime) -> datetime:
        """Validate created_at timestamp is reasonable (not in future, not too old)."""
        now = datetime.utcnow()
        max_age = timedelta(days=365 * 2)  # 2 years max age
        min_time = now - max_age
        max_time = now + timedelta(hours=1)  # Allow 1 hour in future for clock skew

        if v < min_time:
            raise ValueError(f"created_at timestamp too old: {v} (minimum: {min_time})")
        if v > max_time:
            raise ValueError(
                f"created_at timestamp in future: {v} (maximum: {max_time})"
            )
        return v

    @field_validator("processing_metadata")
    @classmethod
    def validate_processing_metadata_bounds(
        cls, v: Optional[ProcessingMetadata]
    ) -> Optional[ProcessingMetadata]:
        """Validate processing metadata has reasonable bounds and consistency."""
        if v is None:
            return v

        # Check count consistency
        if v.processed_papers > v.total_papers:
            raise ValueError(
                f"processed_papers ({v.processed_papers}) > total_papers ({v.total_papers})"
            )

        if v.chunks_embedded > v.total_chunks:
            raise ValueError(
                f"chunks_embedded ({v.chunks_embedded}) > total_chunks ({v.total_chunks})"
            )

        if v.reranked_passages > v.retrieval_candidates:
            raise ValueError(
                f"reranked_passages ({v.reranked_passages}) > retrieval_candidates ({v.retrieval_candidates})"
            )

        # Check reasonableness bounds
        max_reasonable_chunks = 10000  # Max chunks per batch
        if v.total_chunks > max_reasonable_chunks:
            raise ValueError(
                f"total_chunks ({v.total_chunks}) exceeds reasonable limit ({max_reasonable_chunks})"
            )

        if v.memory_usage_mb is not None and v.memory_usage_mb > 10000:  # 10GB max
            raise ValueError(
                f"memory_usage_mb ({v.memory_usage_mb}) exceeds reasonable limit (10000 MB)"
            )

        if (
            v.throughput_chunks_per_second is not None
            and v.throughput_chunks_per_second > 1000
        ):  # 1000 chunks/sec max
            raise ValueError(
                f"throughput_chunks_per_second ({v.throughput_chunks_per_second}) exceeds reasonable limit (1000)"
            )

        # Validate stage times are positive
        for stage, time_val in v.stage_times.items():
            if time_val < 0:
                raise ValueError(
                    f"stage_times['{stage}'] cannot be negative: {time_val}"
                )

        return v

    @model_validator(mode="after")
    def validate_papers_for_acquisition(self):
        """
        Validates that there are papers with PDF URLs before the acquisition step.
        """
        if not self.papers_metadata:
            self.validation_errors.append("SearchAgent returned no papers.")
        elif not any(p.pdf_url for p in self.papers_metadata):
            self.validation_errors.append("No papers with PDF URLs found.")
        return self

    @model_validator(mode="after")
    def validate_state_integrity(self):
        """
        Validate overall state integrity and detect corrupted data gracefully.
        This handles corrupted state data by attempting recovery or marking as recoverable.
        """
        corruption_issues = []

        # Check for data consistency issues
        if self.filtered_papers and self.papers_metadata:
            filtered_ids = {p.paper_id for p in self.filtered_papers}
            metadata_ids = {p.paper_id for p in self.papers_metadata}
            if not filtered_ids.issubset(metadata_ids):
                corruption_issues.append(
                    "filtered_papers contains papers not in papers_metadata"
                )

        # Check chunk-paper relationships
        if self.chunks and self.filtered_papers:
            chunk_paper_ids = {c.paper_id for c in self.chunks}
            paper_ids = {p.paper_id for p in self.filtered_papers}
            if not chunk_paper_ids.issubset(paper_ids):
                corruption_issues.append(
                    "chunks reference papers not in filtered_papers"
                )

        # Check passage-chunk relationships
        if self.relevant_passages and self.chunks:
            chunk_ids = {c.chunk_id for c in self.chunks if c.chunk_id}
            passage_chunk_ids = {
                p.chunk_id for p in self.relevant_passages if p.chunk_id
            }
            if not passage_chunk_ids.issubset(chunk_ids):
                corruption_issues.append(
                    "relevant_passages reference chunks not in chunks"
                )

        # Check for empty required fields in collections
        if self.chunks and any(
            c.paper_id is None or c.paper_id == "" for c in self.chunks
        ):
            corruption_issues.append("chunks contain entries with empty paper_id")

        if self.relevant_passages and any(
            p.paper_id is None or p.paper_id == "" for p in self.relevant_passages
        ):
            corruption_issues.append(
                "relevant_passages contain entries with empty paper_id"
            )

        # If corruption detected, add to validation errors but don't fail validation
        if corruption_issues:
            for issue in corruption_issues:
                self.validation_errors.append(f"Data integrity issue: {issue}")

        return self

    def repair_corrupted_state(self) -> "State":
        """
        Attempt to repair corrupted state data by removing inconsistent entries.
        Returns a new repaired State instance.
        """
        repaired_state = self.model_copy(deep=True)

        # Remove chunks that don't have corresponding papers
        if repaired_state.chunks and repaired_state.filtered_papers:
            paper_ids = {p.paper_id for p in repaired_state.filtered_papers}
            repaired_state.chunks = [
                c for c in repaired_state.chunks if c.paper_id in paper_ids
            ]

        # Remove passages that don't have corresponding chunks (if chunk_id specified)
        if repaired_state.relevant_passages and repaired_state.chunks:
            chunk_ids = {c.chunk_id for c in repaired_state.chunks if c.chunk_id}
            if chunk_ids:  # Only filter if we have chunk_ids to reference
                repaired_state.relevant_passages = [
                    p
                    for p in repaired_state.relevant_passages
                    if p.chunk_id is None or p.chunk_id in chunk_ids
                ]

        # Remove empty/invalid entries
        repaired_state.chunks = [c for c in repaired_state.chunks if c.paper_id]
        repaired_state.relevant_passages = [
            p for p in repaired_state.relevant_passages if p.paper_id
        ]

        # Clear validation errors since we've attempted repair
        repaired_state.validation_errors = []

        return repaired_state

    def to_msgpack(self) -> bytes:
        """
        Serialize State to msgpack format for efficient storage.
        Uses ormsgpack for fast binary serialization.
        """
        # Convert to dict with aliases for backward compatibility
        state_dict = self.model_dump(by_alias=True)
        return ormsgpack.packb(state_dict)

    @classmethod
    def from_msgpack(cls, data: bytes) -> "State":
        """
        Deserialize State from msgpack format.
        Uses ormsgpack for fast binary deserialization.
        """
        state_dict = ormsgpack.unpackb(data)
        return cls(**state_dict)

    def to_msgpack_file(self, filepath: str) -> None:
        """
        Serialize State to msgpack file for efficient storage and fast loading.
        """
        data = self.to_msgpack()
        with open(filepath, "wb") as f:
            f.write(data)

    @classmethod
    def from_msgpack_file(cls, filepath: str) -> "State":
        """
        Load State from msgpack file.
        """
        with open(filepath, "rb") as f:
            data = f.read()
        return cls.from_msgpack(data)

    def enable_lazy_loading(self) -> None:
        """
        Convert large collections to LazyList for memory efficiency.
        Useful when dealing with very large state objects.
        """
        # Convert chunks to lazy loading if over threshold
        if len(self.chunks) > 100:  # Threshold for lazy loading
            chunk_data = self.chunks.copy()
            self.chunks = LazyList(items=chunk_data)

        # Convert passages to lazy loading if over threshold
        if len(self.relevant_passages) > 50:  # Threshold for lazy loading
            passage_data = self.relevant_passages.copy()
            self.relevant_passages = LazyList(items=passage_data)

    @cached_property
    def estimated_memory_usage(self) -> int:
        """
        Estimate memory usage of the state in bytes.
        Cached property for performance.
        """
        # Rough estimation based on collection sizes
        base_size = 1024  # Base object overhead
        chunk_size = len(self.chunks) * 512  # Rough chunk size estimate
        passage_size = len(self.relevant_passages) * 256  # Rough passage size estimate
        paper_size = (
            len(self.papers_metadata) + len(self.filtered_papers)
        ) * 128  # Rough paper size

        return base_size + chunk_size + passage_size + paper_size

    @property
    def query(self) -> str:
        """Alias property for backward compatibility with existing code."""
        return self.original_query

    @query.setter
    def query(self, value: str) -> None:
        """Alias setter for backward compatibility."""
        self.original_query = value

    # State transition methods
    def start_processing(self) -> None:
        """Mark processing as in progress."""
        if self.processing_status == ProcessingStatus.PENDING:
            self.processing_status = ProcessingStatus.IN_PROGRESS
        else:
            logger.warning(
                f"Cannot start processing from status: {self.processing_status}"
            )

    def complete_processing(self) -> None:
        """Mark processing as completed."""
        if self.processing_status in [
            ProcessingStatus.IN_PROGRESS,
            ProcessingStatus.PARTIAL,
        ]:
            self.processing_status = ProcessingStatus.COMPLETED
        else:
            logger.warning(
                f"Cannot complete processing from status: {self.processing_status}"
            )

    def fail_processing(self, error_message: Optional[str] = None) -> None:
        """Mark processing as failed and record error."""
        if self.processing_status in [
            ProcessingStatus.IN_PROGRESS,
            ProcessingStatus.PENDING,
        ]:
            self.processing_status = ProcessingStatus.FAILED
            if error_message:
                if self.processing_metadata:
                    self.processing_metadata.processing_errors.append(error_message)
                else:
                    logger.error(error_message)
        else:
            logger.warning(
                f"Cannot fail processing from status: {self.processing_status}"
            )

    def mark_partial(self) -> None:
        """Mark processing as partially completed."""
        if self.processing_status in [
            ProcessingStatus.IN_PROGRESS,
            ProcessingStatus.COMPLETED,
        ]:
            self.processing_status = ProcessingStatus.PARTIAL

    # Summary and calculation methods
    @property
    def total_execution_time(self) -> float:
        """Calculate total execution time across all phases."""
        total_time = 0.0

        # Add execution_time from legacy field if exists
        if "execution_time" in self.__dict__ and self.execution_time:
            total_time += self.execution_time

        # Add stage times from processing metadata
        if self.processing_metadata:
            total_time += self.processing_metadata.total_time

        # Add stage times from processing stats for backward compatibility
        if self.processing_stats and "stage_times" in self.processing_stats:
            stage_times = self.processing_stats["stage_times"]
            total_time += sum(stage_times.values()) if stage_times else 0

        return total_time

    def get_processing_summary(self) -> Dict[str, Any]:
        """Generate a summary of processing results."""
        summary = {
            "status": self.processing_status.value,
            "total_time": self.total_execution_time,
            "papers_processed": len(self.filtered_papers),
            "chunks_created": len(self.chunks),
            "passages_retrieved": len(self.relevant_passages),
            "config": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "embedding_model": self.config.embedding_model,
                "enable_reranking": self.config.enable_reranking,
            },
            "validation_errors": len(self.validation_errors),
        }

        # Add processing metadata if available
        if self.processing_metadata:
            summary.update(
                {
                    "total_chunks_embedded": self.processing_metadata.chunks_embedded,
                    "retrieval_candidates": self.processing_metadata.retrieval_candidates,
                    "reranked_passages": self.processing_metadata.reranked_passages,
                    "processing_errors": len(
                        self.processing_metadata.processing_errors
                    ),
                    "memory_usage_mb": self.processing_metadata.memory_usage_mb,
                    "throughput_chunks_per_second": self.processing_metadata.throughput_chunks_per_second,
                }
            )

        # Add section distribution
        section_counts = {}
        for chunk in self.chunks:
            section = chunk.section or "unknown"
            section_counts[section] = section_counts.get(section, 0) + 1
        summary["section_distribution"] = section_counts

        return summary

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Extract performance metrics for monitoring."""
        metrics = {
            "status": self.processing_status.value,
            "total_time": self.total_execution_time,
            "validation_errors": len(self.validation_errors),
        }

        # Add processing metadata metrics
        if self.processing_metadata:
            metrics.update(
                {
                    "papers_processed": self.processing_metadata.processed_papers,
                    "total_chunks": self.processing_metadata.total_chunks,
                    "chunks_embedded": self.processing_metadata.chunks_embedded,
                    "retrieval_candidates": self.processing_metadata.retrieval_candidates,
                    "stage_times": dict(self.processing_metadata.stage_times),
                    "memory_usage_mb": self.processing_metadata.memory_usage_mb,
                    "throughput_chunks_per_second": self.processing_metadata.throughput_chunks_per_second,
                    "processing_errors_count": len(
                        self.processing_metadata.processing_errors
                    ),
                }
            )

        # Add legacy processing stats for backward compatibility
        if self.processing_stats:
            # Extract timing information
            legacy_timing = {}
            if "total_time" in self.processing_stats:
                legacy_timing["total_time"] = self.processing_stats["total_time"]
            if "stage_times" in self.processing_stats:
                legacy_timing.update(self.processing_stats["stage_times"])

            if legacy_timing and not metrics.get("stage_times"):
                metrics["stage_times"] = legacy_timing

            # Extract error information
            if (
                "error" in self.processing_stats
                and "processing_errors_count" not in metrics
            ):
                metrics["processing_errors_count"] = 1

        return metrics

    @classmethod
    def from_legacy_dict(cls, data: Dict[str, Any]) -> "State":
        """
        Migrate legacy State dictionaries to the new schema.
        Safely upgrades older State dicts with List[Dict[str, Any]] top_passages to List[Passage].
        """
        data_copy = data.copy()

        # Migrate top_passages from List[Dict] to List[Passage]
        if "top_passages" in data_copy and isinstance(data_copy["top_passages"], list):
            migrated_passages = []
            for passage_dict in data_copy["top_passages"]:
                if isinstance(passage_dict, dict):
                    try:
                        # Map legacy field names if needed (e.g., 'text' -> 'content')
                        passage_data = passage_dict.copy()
                        if "text" in passage_data and "content" not in passage_data:
                            passage_data["content"] = passage_data.pop("text")
                        # Create Passage, with defaults for missing optional fields
                        passage = Passage(
                            content=passage_data.get("content", ""),
                            section=passage_data.get("section", "other"),
                            paper_id=passage_data.get("paper_id", ""),
                            retrieval_score=passage_data.get("retrieval_score", 0.0),
                            cross_encoder_score=passage_data.get("cross_encoder_score"),
                            final_score=passage_data.get("final_score"),
                        )
                        migrated_passages.append(passage)
                    except Exception as e:
                        logger.warning(f"Failed to migrate passage: {e}, skipping")
                else:
                    # If already a Passage object, keep as is
                    migrated_passages.append(passage_dict)
            data_copy["top_passages"] = migrated_passages

        return cls(**data_copy)
