# Necthrall State Schema Documentation

## Overview

The `State` model represents the complete application state for the Necthrall Lite MVP system. It tracks progression through query optimization (Week 1), paper retrieval/filtering, processing outputs (Week 2), and prepares for analysis/synthesis (Week 3).

## Core Components

### Passage Model

Represents a scored text passage retrieved from scientific papers.

**Fields:**
- `content: str` - The actual text content of the passage
- `section: str` - Paper section (must be one of: introduction, methods, results, discussion, other)
- `paper_id: str` - Unique identifier of the source paper
- `retrieval_score: float` - BM25/semantic retrieval score
- `cross_encoder_score: Optional[float]` - Cross-encoder reranking score (optional)
- `final_score: Optional[float]` - Final combined score (optional)

**Validation:**
- Unknown sections are automatically mapped to "other" with a warning log
- All required fields must be provided

### State Model

#### Identification
- `request_id: str` - Auto-generated UUID for request tracking
- `original_query: str` (alias: `query`) - User's original input query
- `created_at: datetime` - Timestamp when state was created

#### Query Management (Week 1)
- `optimized_query: Optional[str]` - LLM-optimized query for OpenAlex search
- `refinement_count: int` - Number of refinement iterations (max 2)

#### Paper Retrieval & Filtering
- `papers_metadata: List[Paper]` - Raw papers from SearchAgent (up to 100 before filtering)
- `filtered_papers: List[Paper]` - Final 25 papers after deduplication and BM25+embedding filtering

#### Processing Pipeline (Week 2)
- `passages: List[Passage]` - Extracted text passages from papers
- `top_passages: List[Passage]` - Top 10 passages with scores (max 10)
- `processing_stats: Optional[Dict[str, Any]]` - Processing stage timings and counts
- `scores: Optional[Dict[str, Any]]` - Passage scoring results
- `answer: Optional[str]` - Intermediate answer from processing
- `citations: List[int]` - Citation indices
- `metrics: Optional[Dict[str, float]]` - Performance metrics
- `retry_count: int` - Processing retry attempts (max 2)

#### Acquisition & Metrics
- `pdf_contents: List[PDFContent]` - Downloaded PDF content
- `download_failures: List[ErrorReport]` - Failed download reports
- `acquisition_metrics: Optional[AcquisitionMetrics]` - Download statistics
- `citation_validation: Optional[Dict[str, Any]]` - Citation validation results

#### Filtering & Quality Metrics
- `filtering_scores: Optional[Dict[str, Any]]` - BM25 and semantic filtering scores
- `dedup_stats: Optional[Dict[str, Any]]` - Deduplication metrics
- `search_quality: Optional[Dict[str, Any]]` - Overall search quality assessment

#### Week 3 Placeholders
- `analysis_results: Optional[Dict[str, Any]]` - Analysis results (future use)
- `synthesized_answer: Optional[str]` - Final synthesized answer (future use)

#### Configuration
- `config: Dict[str, Any]` - Application configuration
- `validation_errors: List[str]` - Accumulated validation errors

## Validation Rules

- Query length: 3-256 characters
- Filtered papers: ≤25
- Top passages: ≤10
- Refinement/retry counts: ≤2
- Section values: normalized to allowed set
- Search quality relevance: 0.0-1.0 range

## Usage Examples

### Basic State Creation
```python
from models.state import State, Passage

state = State(original_query="What is quantum computing?")
```

### Adding Top Passages
```python
passage = Passage(
    content="Quantum computing uses qubits instead of bits...",
    section="introduction",
    paper_id="10.1038/nature12345",
    retrieval_score=0.95,
    cross_encoder_score=0.87,
    final_score=0.91
)

state.top_passages = [passage]
```

### Legacy Migration
```python
legacy_data = {
    "query": "quantum effects",
    "top_passages": [
        {
            "text": "Quantum mechanics explains...",
            "section": "introduction",
            "paper_id": "123",
            "score": 0.8
        }
    ]
}

# Migrate to new schema
state = State.from_legacy_dict(legacy_data)
```

## Schema Evolution

### Week 1 → Week 2
- Added `top_passages: List[Passage]` (changed from `List[Dict[str, Any]]`)
- Added `processing_stats` and filtering metrics
- Enhanced Passage model with scoring fields

### Week 2 → Week 3
- Added `analysis_results` and `synthesized_answer` placeholders
- Schema prepared for future analysis and synthesis phases

## Migration Guide

Use `State.from_legacy_dict(data)` to upgrade older state dictionaries:

1. Maps legacy `passages[].text` to `content`
2. Converts `List[Dict]` `top_passages` to `List[Passage]`
3. Provides defaults for missing optional fields
4. Logs warnings for incompatible data and skips problematic items

## Serialization

The State model serializes cleanly to JSON with:
- Automatic UUID generation for `request_id`
- Alias handling for backward compatibility (`query` ↔ `original_query`)
- Proper datetime formatting
- Default empty lists and None values for optional fields
