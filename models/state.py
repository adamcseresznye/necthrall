from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import logging


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
    """Represents a scored passage retrieved from a paper for Week 2 processing.
    Fields capture content, metadata, and relevance scores from retrieval pipeline.
    """

    content: str
    section: str
    paper_id: str
    retrieval_score: float
    cross_encoder_score: Optional[float] = None
    final_score: Optional[float] = None

    @field_validator("section")
    @classmethod
    def validate_section(cls, v: str) -> str:
        """Normalize section to allowed values, mapping unknowns to 'other' with warning."""
        allowed_sections = {"introduction", "methods", "results", "discussion", "other"}
        if v not in allowed_sections:
            logging.warning(f"Unknown section '{v}' mapped to 'other'")
            return "other"
        return v


class Score(BaseModel):
    """Represents a scoring metric for papers or passages."""

    score_type: str  # e.g., "relevance", "novelty", "quality"
    value: float
    justification: Optional[str] = None
    confidence: Optional[float] = None


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
        description="Final 25 papers after deduplication and BM25+embedding filtering",
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

    # Configuration and workflow control
    config: Dict[str, Any] = Field(default_factory=dict)
    validation_errors: List[str] = Field(default_factory=list)

    # Processing pipeline fields
    passages: List[Passage] = Field(default_factory=list)
    scores: Optional[Dict[str, Any]] = None
    answer: Optional[str] = None
    citations: List[int] = Field(default_factory=list)
    metrics: Optional[Dict[str, float]] = None
    retry_count: int = Field(0, ge=0, le=2)

    # Processing agent output
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
        None, description="Placeholder for synthesized answer from analysis phase"
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

    @field_validator("original_query", "optimized_query")
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
        """Filtered papers should be exactly 25 (or less if insufficient results)"""
        if len(v) > 25:
            raise ValueError(f"Filtered papers should be ≤25, got {len(v)}")
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
                "semantic_top_25",
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

    @property
    def query(self) -> str:
        """Alias property for backward compatibility with existing code."""
        return self.original_query

    @query.setter
    def query(self, value: str) -> None:
        """Alias setter for backward compatibility."""
        self.original_query = value

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
                        logging.warning(f"Failed to migrate passage: {e}, skipping")
                else:
                    # If already a Passage object, keep as is
                    migrated_passages.append(passage_dict)
            data_copy["top_passages"] = migrated_passages

        return cls(**data_copy)
