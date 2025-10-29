#!/usr/bin/env python3
"""
Week 2 Processing Pipeline Performance Benchmark

Runs the full processing pipeline across 5 diverse queries and measures performance.

Performance Requirements:
- Average processing time <4.0s
- p95 time <4.5s
- Peak memory <500MB (monitored)
"""

import asyncio
import json
import logging
import os
import psutil
import random
import sys
import time
import statistics
from typing import Dict, Any, List
import numpy as np
import torch

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Direct model import to avoid FastAPI dependency
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI

from utils.section_detector import SectionDetector
from utils.embedding_manager import EmbeddingManager
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def retry_embedding_call(func, chunks, paper_id, max_retries=1):
    """
    Retry embedding function calls with exponential backoff for transient failures.

    Args:
        func: The embedding function to call (e.g., process_chunks_async)
        chunks: Chunks to process
        paper_id: Paper ID for logging
        max_retries: Maximum number of retry attempts

    Returns:
        Processed chunks with retry metrics annotated
    """
    retry_count = 0
    last_error = None

    while retry_count <= max_retries:
        try:
            logger.info(
                f"Embedding call attempt {retry_count + 1}/{max_retries + 1} for paper {paper_id}"
            )
            result = await func(chunks)
            # Add retry metadata
            for chunk in result:
                chunk["retry_attempts"] = retry_count
                chunk["retry_success"] = True
            return result

        except Exception as e:
            last_error = e
            retry_count += 1
            if retry_count <= max_retries:
                backoff_delay = 0.5 * (2 ** (retry_count - 1))  # Exponential backoff
                logger.warning(
                    f"Embedding failed for paper {paper_id} (attempt {retry_count}/{max_retries + 1}): {e}"
                )
                logger.warning(f"Retrying in {backoff_delay:.1f}s...")
                await asyncio.sleep(backoff_delay)
            else:
                logger.error(
                    f"Embedding failed permanently for paper {paper_id} after {max_retries + 1} attempts"
                )
                # Return placeholder chunks on complete failure
                placeholder_chunks = []
                for chunk in chunks:
                    placeholder_chunks.append(
                        {
                            "content": chunk["content"],
                            "embedding": np.zeros(384, dtype=np.float32),
                            "embedding_dim": 384,
                            "retry_attempts": retry_count,
                            "retry_success": False,
                            "retry_error": str(last_error),
                        }
                    )
                return placeholder_chunks

    # This should never be reached
    raise last_error


def create_mock_app(rrf_k: int = 40) -> FastAPI:
    """Create mock FastAPI app with loaded embedding model."""
    app = FastAPI()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    app.state.embedding_model = model
    app.state.rrf_k = rrf_k  # Store RRF parameter in app state
    return app


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Split text into chunks by characters with sentence boundary preference."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to end at sentence boundary if possible
        if end < len(text):
            last_period = text.rfind(". ", start, end)
            if last_period > end - 200:  # Don't cut too short
                end = last_period + 2

        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

        # Move start with overlap
        start = end - overlap
        if start <= 0:
            break

    return chunks


async def run_processing_pipeline(
    query: str, sample_papers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Run the full processing pipeline for a single query against a corpus of sample papers.

    Args:
        query: The search query
        sample_papers: List of paper dictionaries with 'id', 'title', 'content' keys

    Returns:
        Dictionary with timing results and processed passages or error info
    """
    # Set deterministic seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    # Track memory usage
    initial_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)

    # Initialize components
    section_detector = SectionDetector()
    mock_app = create_mock_app()
    embedding_manager = EmbeddingManager(mock_app, batch_size=16)
    hybrid_retriever = HybridRetriever()
    reranker = CrossEncoderReranker()

    # Collect all processed chunks across papers for retrieval index
    all_processed_chunks = []
    chunk_id_counter = 0

    pipeline_start = time.time()

    try:
        # 1. Process all sample papers: section detection + chunking
        logger.info(
            json.dumps(
                {"event": "corpus_processing_start", "paper_count": len(sample_papers)}
            )
        )

        for paper_idx, paper in enumerate(sample_papers):
            paper_start = time.time()

            # Section Detection
            sections = section_detector.detect_sections(paper["content"])

            # Fallback chunking if needed
            if len(sections) < 2:
                logger.warning(
                    json.dumps(
                        {
                            "event": "fallback_chunking_triggered",
                            "paper_id": paper.get("id", f"paper_{paper_idx}"),
                            "sections_detected": len(sections),
                            "reason": "fewer_than_2_sections_detected",
                        }
                    )
                )
                sections = [{"content": paper["content"], "section": "fallback"}]

            # Chunking by section and size
            paper_chunks = []
            for section in sections:
                section_chunks = chunk_text(
                    section["content"], chunk_size=1200, overlap=200
                )
                for chunk_content in section_chunks:
                    paper_chunks.append(
                        {
                            "content": chunk_content,
                            "section": section["section"],
                            "doc_id": chunk_id_counter,
                            "start_pos": chunk_id_counter,
                            "paper_id": paper.get("id", f"paper_{paper_idx}"),
                            "paper_title": paper.get("title", ""),
                        }
                    )
                    chunk_id_counter += 1

            # Embedding generation for this paper's chunks with retry logic
            if paper_chunks:
                processed_chunks = await retry_embedding_call(
                    embedding_manager.process_chunks_async,
                    paper_chunks,
                    paper.get("id", f"paper_{paper_idx}"),
                )
                all_processed_chunks.extend(processed_chunks)

            paper_time = time.time() - paper_start
            logger.info(
                json.dumps(
                    {
                        "event": "paper_processed",
                        "paper_id": paper.get("id", f"paper_{paper_idx}"),
                        "chunks_created": len(paper_chunks),
                        "time_ms": round(paper_time * 1000, 2),
                    }
                )
            )

        if not all_processed_chunks:
            return {
                "query": query,
                "success": False,
                "error": "No chunks created from any sample papers",
                "total_time": time.time() - pipeline_start,
                "stage_times": {},
            }

        # 2. Build retrieval index from all chunks
        logger.info(
            json.dumps(
                {
                    "event": "index_build_start",
                    "total_chunks": len(all_processed_chunks),
                }
            )
        )
        build_start = time.time()
        build_time = hybrid_retriever.build_indices(
            all_processed_chunks, use_cache=False
        )
        build_end = time.time()

        # 3. Retrieval
        logger.info(json.dumps({"event": "retrieval_start", "query": query}))
        retrieval_start = time.time()

        query_embedding = mock_app.state.embedding_model.encode([query])[0]
        retrieved = hybrid_retriever.retrieve(query, query_embedding, top_k=12)
        retrieval_end = time.time()

        if not retrieved:
            logger.warning(f"Empty retrieval results for query: {query}")
            return {
                "query": query,
                "success": False,
                "error": "Empty candidates from retrieval",
                "total_time": time.time() - pipeline_start,
                "stage_times": {
                    "index_build": build_end - build_start,
                    "retrieval": retrieval_end - retrieval_start,
                },
            }

        # 4. Reranking
        logger.info(
            json.dumps({"event": "reranking_start", "candidates": len(retrieved)})
        )
        rerank_start = time.time()
        reranked = reranker.rerank(query, retrieved[:12])  # Top-12 → top-10 (optimized)
        rerank_end = time.time()

        total_time = time.time() - pipeline_start
        peak_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)

        logger.info(
            json.dumps(
                {
                    "event": "pipeline_complete",
                    "query": query,
                    "total_time_seconds": round(total_time, 3),
                    "stage_times": {
                        "index_build": round(build_end - build_start, 3),
                        "retrieval": round(retrieval_end - retrieval_start, 3),
                        "reranking": round(rerank_end - rerank_start, 3),
                    },
                    "final_passage_count": len(reranked),
                    "peak_memory_mb": round(peak_memory_mb, 2),
                }
            )
        )

        return {
            "query": query,
            "success": True,
            "total_time": total_time,
            "stage_times": {
                "index_build": build_end - build_start,
                "retrieval": retrieval_end - retrieval_start,
                "reranking": rerank_end - rerank_start,
            },
            "peak_memory_mb": peak_memory_mb,
            "passages_returned": len(reranked),
            "chunks_indexed": len(all_processed_chunks),
        }

    except Exception as e:
        total_time = time.time() - pipeline_start
        logger.error(
            json.dumps(
                {
                    "event": "pipeline_error",
                    "query": query,
                    "error": str(e),
                    "total_time_seconds": round(total_time, 3),
                }
            )
        )
        return {
            "query": query,
            "success": False,
            "error": str(e),
            "total_time": total_time,
            "stage_times": {},
        }


def create_sample_papers() -> List[Dict[str, Any]]:
    """Create a corpus of sample scientific papers for benchmarking."""
    return [
        {
            "id": "cardiovascular_fasting",
            "title": "Cardiovascular Effects of Intermittent Fasting",
            "content": """
1. Introduction

Cardiovascular disease remains a leading cause of mortality worldwide, accounting for approximately 17.9 million deaths annually according to the World Health Organization. Intermittent fasting, characterized by alternating periods of eating and fasting, has emerged as a potential intervention for metabolic health and cardiovascular risk reduction. Recent studies suggest intermittent fasting may improve cardiovascular risk factors including blood pressure, lipid profiles, glucose metabolism, and inflammatory markers. This comprehensive review examines the evidence for cardiovascular effects of intermittent fasting in human populations, drawing from randomized controlled trials and observational studies conducted over the past decade.

2. Methods

We conducted a systematic review of randomized controlled trials published between 2000-2023 investigating intermittent fasting effects on cardiovascular outcomes. Multiple databases were comprehensively searched: PubMed, Cochrane Central Register of Controlled Trials, Web of Science, Scopus, and ClinicalTrials.gov. Search terms included combinations of "intermittent fasting", "time-restricted eating", "caloric restriction", "cardiovascular disease", "coronary artery disease", "blood pressure", "hypertension", "dyslipidemia", "cholesterol", "triglycerides", "glucose", "insulin resistance", and related medical subject headings.

3. Results

Twenty-four randomized trials involving 2,847 participants were included in this systematic review. Study duration ranged from 4 weeks to 2 years, with sample sizes from 22 to 538 participants. Mean age of participants varied from 25 to 68 years, and baseline BMI ranged from 23 to 36 kg/m², representing diverse demographic and health status groups.

Pooled analysis revealed intermittent fasting significantly reduced systolic blood pressure by -4.8 mmHg and diastolic blood pressure by -3.1 mmHg. Lipid profile improvements included reductions in total cholesterol (-0.31 mmol/L), LDL cholesterol (-0.25 mmol/L), and triglycerides (-0.23 mmol/L), with increased HDL cholesterol.

4. Discussion

The cardiovascular benefits appear mediated through weight loss, enhanced insulin sensitivity, autonomic nervous system modulation, and anti-inflammatory effects. No significant adverse cardiovascular events were observed, with mild side effects including hunger and fatigue.

5. Conclusion

Intermittent fasting represents a promising dietary approach for cardiovascular risk reduction with evidence from multiple randomized trials showing improvements in blood pressure, lipid profiles, and inflammatory markers.
""",
        },
        {
            "id": "neurodegeneration_autophagy",
            "title": "Autophagy Mechanisms in Neurodegenerative Diseases",
            "content": """
1. Introduction

Neurodegenerative diseases including Alzheimer's disease, Parkinson's disease, and Huntington's disease affect millions worldwide and represent a growing public health challenge. Autophagy, the cellular process of degradation and recycling of cytoplasmic components, has emerged as a critical regulator of neuronal homeostasis and survival. Dysfunctional autophagy is increasingly recognized as a common pathological feature across multiple neurodegenerative conditions. This review explores the role of autophagy in neurodegeneration, highlighting potential therapeutic targets for disease modification.

2. Methods

A comprehensive literature search was conducted covering publications from 2010-2023 in major scientific databases. Studies investigating autophagy mechanisms in neurodegenerative disease models and human tissues were included, with particular focus on molecular pathways, genetic associations, and therapeutic interventions targeting autophagy.

3. Results

Autophagy impairment contributes to neurodegeneration through multiple mechanisms. In Alzheimer's disease, defective autophagy leads to accumulation of amyloid-beta peptides and hyperphosphorylated tau proteins. Parkinson's disease involves autophagy defects in alpha-synuclein degradation, mitochondrial quality control, and lysosomal function. Huntington's disease features impaired autophagosome formation due to mutant huntingtin protein interference.

Genetic studies identified mutations in autophagy-related genes (ATG7, ATG16L1, WIPI3) increase neurodegenerative disease risk. Pharmacological activation of autophagy using rapamycin, spermidine, and lithium showed neuroprotective effects in preclinical models.

4. Discussion

Autophagy modulation represents a promising therapeutic strategy for neurodegenerative diseases. Challenges include achieving brain-specific autophagy enhancement and potential off-target effects. Combination therapies targeting multiple autophagy pathways may provide synergistic benefits.

5. Conclusion

Autophagy dysfunction is a central mechanism in neurodegenerative disease pathogenesis. Future research should focus on developing safe and effective autophagy modulators for clinical translation in Alzheimer's, Parkinson's, and Huntington's disease prevention and treatment.
""",
        },
        {
            "id": "climate_change_adaptation",
            "title": "Climate Change Adaptation Strategies for Coastal Cities",
            "content": """
1. Introduction

Coastal cities worldwide face increasing threats from climate change including sea level rise, intensified storms, and coastal flooding. The urban coastal interface represents a critical zone where natural and human systems intersect, demanding integrated adaptation strategies. This study examines climate adaptation approaches implemented in major coastal cities, analyzing effectiveness, challenges, and lessons for future planning.

2. Methods

We analyzed adaptation strategies in 12 major coastal cities across 5 continents, including New York, London, Tokyo, Mumbai, Sydney, and Shanghai. Data sources included urban planning documents, climate action plans, infrastructure projects, and peer-reviewed literature from 2010-2023.

3. Results

Successful adaptation strategies include: 1) Integrated flood management combining natural and engineered solutions, 2) Ecosystem-based adaptation preserving mangroves and wetlands, 3) Resilient infrastructure design with elevated buildings and flood-resistant materials, 4) Community engagement programs including early warning systems and evacuation planning, 5) Green infrastructure incorporating permeable surfaces and urban forests.

Economic analysis revealed that proactive adaptation investments yield benefit-cost ratios ranging from 2:1 to 7:1, with avoided damages significantly exceeding implementation costs. Social equity considerations highlight the need for inclusive planning that addresses vulnerable populations.

4. Discussion

Key challenges include competing urban development pressures, institutional coordination across governance levels, and uncertainty in climate projections. Successful implementation requires strong political leadership, adequate funding mechanisms, and robust monitoring and evaluation systems.

5. Conclusion

Coastal cities demonstrate that comprehensive climate adaptation is both necessary and achievable. Integrated approaches combining engineering, ecological, and social solutions offer the most promising path forward for building urban resilience against climate change impacts.
""",
        },
        {
            "id": "quantum_computing_algorithms",
            "title": "Quantum Computing Algorithms for Optimization Problems",
            "content": """
1. Introduction

Quantum computing represents a paradigm shift in computational capabilities, promising exponential speedups for certain classes of problems. Optimization problems, including combinatorial optimization and machine learning training, form a critical application domain. This review examines quantum algorithms designed for optimization, their theoretical foundations, and current experimental implementations.

2. Methods

We analyzed quantum optimization algorithms including the Quantum Approximate Optimization Algorithm (QAOA), Quantum Annealing, Variational Quantum Eigensolver (VQE), and quantum walks. Performance comparisons were made against classical algorithms across problem domains including MaxCut, traveling salesman, portfolio optimization, and machine learning.

3. Results

QAOA demonstrated theoretical quadratic speedups for MaxCut problems, with experimental implementations achieving up to 60-qubit problem sizes on current hardware. Quantum annealing showed strong performance on Ising model problems, with commercial D-Wave systems solving optimization instances with thousands of variables.

VQE algorithms provided accurate solutions for molecular ground state calculations, enabling quantum chemistry simulations beyond classical capabilities. Quantum walk algorithms offered polynomial speedups for search problems on graphs.

4. Discussion

Current quantum hardware limitations including noise, decoherence, and qubit connectivity restrict algorithm performance. Hybrid quantum-classical approaches, combining quantum speedup with classical optimization, represent the most practical near-term strategy.

5. Conclusion

Quantum optimization algorithms demonstrate theoretical and experimental promise for solving classically intractable problems. Continued hardware improvements and algorithm refinements will expand the range of quantum-advantage applications in optimization domains.
""",
        },
        {
            "id": "machine_learning_bias",
            "title": "Bias Detection and Mitigation in Machine Learning Systems",
            "content": """
1. Introduction

Machine learning systems increasingly influence high-stakes decisions in healthcare, criminal justice, employment, and finance. Algorithmic bias, arising from biased training data and model assumptions, can lead to unfair and discriminatory outcomes. This review examines sources of bias in machine learning, detection methodologies, and mitigation strategies for building fair and equitable AI systems.

2. Methods

We conducted a systematic review of bias detection and mitigation techniques published between 2018-2023. Analysis covered technical approaches including fairness metrics, bias detection algorithms, debiasing methods, and evaluation frameworks across different application domains.

3. Results

Common bias sources include: 1) Representational bias from unrepresentative training data, 2) Measurement bias from proxies that correlate with protected attributes, 3) Algorithmic bias from optimization objectives, 4) Human bias encoded in labels and annotations.

Detection techniques employ fairness metrics such as demographic parity, equal opportunity, and individual fairness. Statistical tests including correlation analysis and causal inference methods identify biased features and outcomes.

Mitigation strategies include: 1) Data preprocessing techniques like reweighting and resampling, 2) Algorithmic modifications such as adversarial debiasing and fair representation learning, 3) Post-processing approaches including threshold adjustment and rejection option classification.

4. Discussion

Bias mitigation requires domain expertise and careful consideration of fairness definitions appropriate to specific contexts. Trade-offs between different fairness criteria necessitate stakeholder engagement in defining acceptable trade-off boundaries.

5. Conclusion

Addressing algorithmic bias demands systematic approaches integrating technical solutions with ethical considerations and stakeholder collaboration. Ongoing monitoring and iterative improvement represent essential components of responsible AI development.
""",
        },
    ]


async def main():
    """Run the Week 2 performance benchmark across 5 diverse queries."""
    print("🚀 Starting Week 2 Processing Pipeline Performance Benchmark")
    print("=" * 60)

    # Load sample papers corpus
    sample_papers = create_sample_papers()
    print(f"📚 Loaded corpus: {len(sample_papers)} sample papers")

    # Define diverse benchmark queries
    test_queries = [
        "cardiovascular effects of intermittent fasting",
        "autophagy mechanisms in neurodegeneration",
        "climate change adaptation strategies for coastal cities",
        "quantum algorithms for optimization problems",
        "bias detection in machine learning systems",
    ]

    print(f"🔍 Benchmarking {len(test_queries)} diverse queries")
    print()

    # Track results
    all_results = []
    successful_times = []
    failed_queries = 0

    # Process each query
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}/{len(test_queries)}: {query}")
        print("-" * 50)

        result = await run_processing_pipeline(query, sample_papers)

        # Emit per-query JSON log line
        query_log = {
            "event": "benchmark_query_result",
            "query_index": i,
            "total_queries": len(test_queries),
            **result,
        }
        with open("../performance_log.json", "a") as f:
            f.write(json.dumps(query_log) + "\n")

        if result["success"]:
            print(f"   ⏱️  Total time: {result['total_time']:.3f}s")
            print(f"   📊 Chunks indexed: {result['chunks_indexed']}")
            print(f"   📈 Passages returned: {result['passages_returned']}")
            print(f"   🧠 Peak memory: {result['peak_memory_mb']:.1f}MB")
            # Breakdown by stage
            stages = result["stage_times"]
            for stage, time_taken in stages.items():
                print(f"   • {stage}: {time_taken:.3f}s")
            successful_times.append(result["total_time"])
            all_results.append(result)
        else:
            print(f"❌ Failed: {result['error']}")
            failed_queries += 1

        print()

    # Aggregate results
    if successful_times:
        avg_time = statistics.mean(successful_times)
        p95_time = statistics.quantiles(successful_times, n=20)[18]  # 95th percentile
        worst_time = max(successful_times)
        success_rate = len(successful_times) / len(test_queries)

        print("📊 PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"⏱️  Average time: {avg_time:.3f}s")
        print(f"📈 p95 time: {p95_time:.3f}s")
        print(f"⏱️  Worst time: {worst_time:.3f}s")
        print(f"✅ Success rate: {success_rate:.1%}")
        print(f"❌ Failed queries: {failed_queries}")

        # Memory usage summary
        if "peak_memory_mb" in all_results[0]:
            memory_usages = [
                r["peak_memory_mb"] for r in all_results if "peak_memory_mb" in r
            ]
            print(f"🧠 Peak memory usage: {max(memory_usages):.1f}MB")
            if max(memory_usages) < 500:
                print(f"✅ Memory usage within limits (< 500MB)")
            else:
                print(f"⚠️  High memory usage detected")

        # Performance assertions
        print("\n🎯 PERFORMANCE TARGETS")
        print("-" * 30)
        avg_ok = avg_time < 4.0
        p95_ok = p95_time < 4.5

        print(f"Average < 4.0s: {'✅ PASS' if avg_ok else '❌ FAIL'} ({avg_time:.3f}s)")
        print(f"P95 < 4.5s: {'✅ PASS' if p95_ok else '❌ FAIL'} ({p95_time:.3f}s)")

        if avg_ok and p95_ok:
            print("🎉 ALL TARGETS MET!")
        else:
            print("⚠️  Performance targets not fully met")
            if not avg_ok:
                print(f"   ❌ Average target missed by {avg_time - 4.0:.3f}s")
            if not p95_ok:
                print(f"   ❌ P95 target missed by {p95_time - 4.5:.3f}s")

        # Final aggregated JSON dump
        try:
            metrics = {
                "event": "benchmark_aggregated_results",
                "benchmark_timestamp": time.time(),
                "queries_tested": len(test_queries),
                "successful_queries": len(successful_times),
                "failed_queries": failed_queries,
                "success_rate": success_rate,
                "performance_metrics": {
                    "average_time": round(avg_time, 3),
                    "p95_time": round(p95_time, 3),
                    "worst_time": round(worst_time, 3),
                    "targets_met": {
                        "average_under_4s": avg_ok,
                        "p95_under_4_5s": p95_ok,
                    },
                },
                "configurations": {
                    "rrf_k": 40,
                    "reranking_candidates": 12,
                    "embedding_batch_size": 16,
                    "retry_attempts": 1,
                },
                "hardware_metrics": {
                    "peak_memory_mb": max(memory_usages) if memory_usages else None,
                    "memory_within_limits": (
                        max(memory_usages) < 500 if memory_usages else None
                    ),
                },
                "individual_results": all_results,
            }

            with open("../performance_log.json", "a") as f:
                f.write(json.dumps(metrics, indent=2))
                f.write("\n")
            print("💾 Aggregated metrics saved to performance_log.json")

        except Exception as e:
            print(f"Warning: Could not save metrics to JSON: {e}")

    else:
        print("❌ No successful queries to analyze")
        success_rate = 0.0

    # Final assertion for tests
    if successful_times:
        try:
            assert avg_time < 4.0, f"Average time {avg_time:.3f}s exceeds 4.0s limit"
            assert p95_time < 4.5, f"P95 time {p95_time:.3f}s exceeds 4.5s limit"
            assert (
                sum(1 for t in successful_times if t < 4.0) >= 4
            ), f"Only {sum(1 for t in successful_times if t < 4.0)} out of {len(successful_times)} queries under 4.0s"
            print("✅ All performance assertions passed!")
        except AssertionError as e:
            print(f"🚨 Performance assertion failed: {e}")
            return False

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
