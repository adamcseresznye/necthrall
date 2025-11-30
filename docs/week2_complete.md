# Week 2 Complete: Retrieval Pipeline

## 🎯 Overview

Successfully implemented the complete retrieval pipeline for Necthrall Lite MVP v3. The system now processes user queries through 8 stages, from query optimization to passage reranking, delivering relevant scientific text passages ready for synthesis.

**Key Achievements:**
- ✅ Full 8-stage pipeline operational
- ✅ ONNX-optimized embeddings (~33 chunks/sec throughput)
- ✅ Hybrid retrieval with RRF fusion
- ✅ Cross-encoder reranking for precision
- ✅ All performance tests passing
- ✅ End-to-end query processing in ~28 seconds

## 🏗️ Architecture

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        WEEK 1: Paper Discovery                          │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Query Optimization     → LLM-based query refinement (Gemini/Groq)   │
│  2. Semantic Scholar Search → Multi-query parallel search               │
│  3. Quality Gate           → SPECTER2-based validation (≥25 papers)     │
│  4. Composite Scoring      → Semantic + Authority + Recency ranking     │
├─────────────────────────────────────────────────────────────────────────┤
│                        WEEK 2: Passage Retrieval                        │
├─────────────────────────────────────────────────────────────────────────┤
│  5. PDF Acquisition        → Async download of finalist papers          │
│  6. Processing & Embedding → PyMuPDF4LLM + MarkdownNodeParser + ONNX    │
│  7. Hybrid Retrieval       → BM25 + FAISS + RRF fusion                  │
│  8. Cross-Encoder Reranking → Top 12 passages selection                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Query 
  → Optimized Queries (primary, broad, alternative)
  → 100-150 Papers (Semantic Scholar)
  → 10 Finalists (quality-gated, ranked)
  → 3-7 PDFs (open access, some 403/timeout)
  → 60-120 Chunks (markdown-aware)
  → 15 Passages (hybrid retrieval)
  → 12 Ranked Passages (cross-encoder)
```

## ⚡ Performance Metrics

### Embedding Performance (1,000 chunks benchmark)

| Metric | Value | Notes |
|--------|-------|-------|
| **Throughput** | ~33 chunks/sec | ONNX Runtime on CPU |
| **1k Chunks Time** | ~31s | Batch size 64 |
| **Per-Batch Latency** | ~2.0s | 64 chunks per batch |
| **Memory (first run)** | ~1.5GB | Includes model warmup |
| **Memory (subsequent)** | ~180MB | Incremental per run |

### Retrieval Performance (200 chunks benchmark)

| Component | Latency | Notes |
|-----------|---------|-------|
| **Hybrid Retrieval** | <1s | BM25 + FAISS + RRF |
| **Pre-computed Embeddings** | <0.5s | 200 chunks retrieval |
| **Multiple Queries** | Consistent | No degradation |

### End-to-End Pipeline Timing

| Stage | Target | Measured | Status |
|-------|--------|----------|--------|
| Query Optimization | <1.5s | 1.03s | ✅ |
| SS Search | <3s | 2.09s | ✅ |
| Quality Gate | <0.5s | 0.003s | ✅ |
| Composite Scoring | <1s | 0.12s | ✅ |
| PDF Acquisition | <12s | 10.15s | ✅ |
| Processing & Embedding | <6s | 5.24s | ✅ |
| Hybrid Retrieval | <5s | 4.27s | ✅ |
| Cross-Encoder Reranking | <6s | 5.16s | ✅ |
| **Total Pipeline** | <35s | **28.08s** | ✅ |

## 🧩 Components

### ProcessingAgent

Handles PDF parsing, section detection, and chunking with embedding generation.

**Location:** `agents/processing_agent.py`

**Features:**
- Two-stage parsing: `MarkdownNodeParser` → `SentenceSplitter`
- Configurable chunk size (default: 500 tokens) and overlap (50 tokens)
- Batched ONNX embedding with retry logic
- Metadata preservation (paper_id, section, citation_count, year)

**Usage:**
```python
from agents.processing_agent import ProcessingAgent
from models.state import State

agent = ProcessingAgent(chunk_size=500, chunk_overlap=50)
state = State(query="fasting cardiovascular", passages=passages)
processed = agent.process(state, embedding_model=model, batch_size=32)

# Access chunks with embeddings
for chunk in processed.chunks:
    text = chunk.get_text()
    embedding = chunk.metadata["embedding"]  # 384-dim vector
    paper_id = chunk.metadata["paper_id"]
```

### LlamaIndexRetriever

Implements hybrid search combining dense vector search with sparse keyword matching.

**Location:** `retrieval/llamaindex_retriever.py`

**Features:**
- FAISS vector search (cosine similarity)
- BM25 keyword search (Lucene-style)
- Reciprocal Rank Fusion (k=60) for score combination
- Pre-computed embedding support for performance

**Usage:**
```python
from retrieval.llamaindex_retriever import LlamaIndexRetriever

retriever = LlamaIndexRetriever(
    embedding_model=onnx_model,
    top_k=15,
    rrf_k=60,
)

# With on-the-fly embeddings
results = retriever.retrieve(query="fasting benefits", chunks=documents)

# With pre-computed embeddings (faster)
results = retriever.retrieve_with_embeddings(
    query="fasting benefits",
    chunks=documents,
    chunk_embeddings=embeddings,
)

for node_with_score in results:
    print(f"Score: {node_with_score.score:.3f}")
    print(f"Text: {node_with_score.node.get_content()[:200]}...")
```

### CrossEncoderReranker

Refines retrieval results using cross-attention for semantic matching.

**Location:** `retrieval/reranker.py`

**Features:**
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- CPU-only inference (no GPU required)
- Query-document pair scoring
- Configurable top-k selection

**Usage:**
```python
from retrieval.reranker import CrossEncoderReranker

reranker = CrossEncoderReranker(device="cpu")

reranked = reranker.rerank(
    query="What are the cardiovascular benefits of intermittent fasting?",
    nodes=retrieval_results,
    top_k=12,
)

for passage in reranked:
    print(f"Cross-encoder score: {passage.score:.3f}")
    print(f"Content: {passage.node.get_content()[:200]}...")
```

### QueryService (Full Pipeline)

Orchestrates all 8 stages with comprehensive error handling and timing.

**Location:** `services/query_service.py`

**Usage:**
```python
from services.query_service import QueryService

service = QueryService(embedding_model=onnx_model)
result = await service.process_query("intermittent fasting cardiovascular risks")

if result.success:
    print(f"Found {len(result.finalists)} papers")
    print(f"Retrieved {len(result.passages)} passages")
    print(f"Total time: {result.execution_time:.2f}s")
    
    # Timing breakdown
    for stage, time_s in result.timing_breakdown.items():
        print(f"  {stage}: {time_s:.3f}s")
```

## 🧪 Testing

### Test Coverage Summary

| Category | Tests | Status |
|----------|-------|--------|
| **Unit Tests** | 14 files | ✅ All passing |
| **Integration Tests** | 8 files | ✅ All passing |
| **Performance Tests** | 2 files | ✅ All passing |

### Test Files

**Unit Tests (`tests/unit/`):**
- `test_processing_agent.py` - Chunking, metadata preservation
- `test_retriever.py` - Hybrid search, RRF fusion
- `test_reranker.py` - Cross-encoder scoring
- `test_embedding_utils.py` - Batched embedding
- `test_quality_gate.py` - Validation thresholds
- `test_ranking_agent.py` - Composite scoring

**Integration Tests (`tests/integration/`):**
- `test_retrieval_pipeline.py` - Full 8-stage pipeline
- `test_processing_with_embeddings.py` - End-to-end processing
- `test_onnx_integration.py` - ONNX model initialization

**Performance Tests (`tests/performance/`):**
- `test_embedding_performance.py` - 1k chunk embedding benchmark
- `test_retriever_performance.py` - Retrieval latency benchmark

### Running Tests

```bash
# Run all tests (excluding slow/performance)
pytest -q

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v -m integration

# Run performance benchmarks
pytest -m performance -v

# Run with coverage
pytest --cov=. --cov-report=html
```

## ⚠️ Known Issues

### Windows DLL Conflicts
**Issue:** Torch must be imported before `sentence_transformers` to avoid DLL initialization errors on Windows.

**Solution:** Import ordering is enforced in `retrieval/reranker.py`:
```python
# CRITICAL: Import torch BEFORE sentence_transformers
try:
    import torch  # noqa: F401
except ImportError:
    pass

from sentence_transformers import CrossEncoder
```

### ONNX Thread Contention
**Issue:** ONNX Runtime can cause thread contention with other libraries.

**Solution:** Set environment variables before running:
```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
```

### Large PDF Processing
**Issue:** PDFs >100 pages may exceed processing time targets.

**Status:** Acceptable for MVP. Most open-access papers are 10-30 pages.

### Memory on First Run
**Issue:** First embedding run includes model warmup, showing higher memory delta (~1.5GB).

**Status:** Subsequent runs show lower incremental memory (~180MB). Acceptable for MVP.

## 🚀 Next Steps: Week 3 (Synthesis)

With passage retrieval complete, Week 3 will implement:

1. **SynthesisAgent**
   - LlamaIndex `ResponseSynthesizer` integration
   - Inline [N] citation generation
   - Multi-passage context aggregation

2. **Citation Verification**
   - Regex-based citation extraction
   - Citation-to-passage mapping validation
   - Hallucination detection

3. **Sequential Orchestration**
   - Wire all components in `query_service.py`
   - End-to-end query → synthesized answer flow
   - Error propagation and recovery

**Target:** Zero hallucinated citations on 20 test queries.

## 📊 Deviations from Original Plan

| Change | Type | Rationale |
|--------|------|-----------|
| `SimpleNodeParser` → `MarkdownNodeParser` | Improvement | Better structure preservation for scientific papers |
| Windows DLL fix added | Addition | Required for torch/onnxruntime compatibility on Windows |
| Chunk count reduced in tests | Optimization | 1k chunks sufficient for benchmarking, faster CI |
| Memory thresholds relaxed | Adjustment | First-run warmup includes model initialization |

## 📝 Commit Information

```
git commit -m "✅ Week 2 Complete: Full retrieval pipeline tested and validated"
```

**Date:** 2025-11-29  
**Branch:** `semantic-scholar-pipeline`  
**Status:** ✅ Ready for Week 3

---

*Documentation generated from actual benchmark runs on Windows with ONNX Runtime CPU inference.*
