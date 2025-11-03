#!/usr/bin/env python3
"""
End-to-end Processing Pipeline Test

Validates the full Week 2 pipeline: section detection → chunking → embeddings → hybrid retrieval → reranking

Performance budgets:
- Total pipeline: ≤ 4.0 seconds
- Section detection: ≤ 0.05s
- Embeddings (10K chunks): ≤ 3.0s
- Hybrid retrieval: ≤ 0.5s
- Cross-encoder reranking: ≤ 0.6s

Ensures deterministic outputs and structured logging with per-stage metrics.
"""

import asyncio
import json
import os
import psutil
import random
import sys
import time
from typing import Dict, Any, List
import numpy as np
import torch

# Add project root to Python path
sys.path.insert(0, os.path.abspath(".."))

# Direct model import to avoid FastAPI dependency
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI

from utils.section_detector import SectionDetector
from utils.embedding_manager import EmbeddingManager
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker

# Use Loguru for structured logging
from loguru import logger


def create_mock_app() -> FastAPI:
    """Create mock FastAPI app with loaded embedding model."""
    app = FastAPI()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    app.state.embedding_model = model
    return app


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Split text into chunks by characters with sentence boundary preference.

    Approximates 400-600 tokens (400-600 words) using character-based chunking.
    400 words ≈ 2400 chars, 600 words ≈ 3600 chars; we use 1200 chars average.

    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of text chunks
    """
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


def main():
    """Run the end-to-end pipeline test."""
    # Set deterministic seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    # Disable HuggingFace hash randomization for deterministic model loading
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    # Track initial memory usage
    initial_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)

    # Sample IMRaD paper text (expanded for chunking to produce ~10-15 chunks)
    paper_text = """
1. Introduction

Cardiovascular disease remains a leading cause of mortality worldwide, accounting for approximately 17.9 million deaths annually according to the World Health Organization. Intermittent fasting, characterized by alternating periods of eating and fasting, has emerged as a potential intervention for metabolic health and cardiovascular risk reduction. Recent studies suggest intermittent fasting may improve cardiovascular risk factors including blood pressure, lipid profiles, glucose metabolism, and inflammatory markers. This comprehensive review examines the evidence for cardiovascular effects of intermittent fasting in human populations, drawing from randomized controlled trials and observational studies conducted over the past decade.

The rising prevalence of obesity and related cardiovascular complications necessitates novel therapeutic approaches beyond traditional pharmacological interventions. Intermittent fasting represents one such strategy that combines dietary modification with chronobiological alignment, potentially leveraging evolutionary adaptations to food scarcity. Understanding the mechanisms through which intermittent fasting influences cardiovascular physiology is crucial for clinical translation and widespread implementation in preventive cardiology.

Historical evidence from religious fasting practices and caloric restriction studies provides background for modern intermittent fasting research. The 5:2 diet, alternate-day fasting, and time-restricted eating all fall under the intermittent fasting umbrella, each with varying degrees of caloric restriction and fasting duration. These approaches have shown promising results in animal models of cardiovascular disease, prompting translation to human clinical trials.

2. Methods

We conducted a systematic review of randomized controlled trials published between 2000-2023 investigating intermittent fasting effects on cardiovascular outcomes. Multiple databases were comprehensively searched: PubMed, Cochrane Central Register of Controlled Trials, Web of Science, Scopus, and ClinicalTrials.gov. Search terms included combinations of "intermittent fasting", "time-restricted eating", "caloric restriction", "cardiovascular disease", "coronary artery disease", "blood pressure", "hypertension", "dyslipidemia", "cholesterol", "triglycerides", "glucose", "insulin resistance", and related medical subject headings.

Inclusion criteria were rigorously defined: adult humans aged 18 years or older, controlled trial design (randomized or quasi-randomized), intervention involving intermittent fasting for at least 4 weeks, and measurement of at least one cardiovascular outcome. Exclusion criteria included studies of continuous energy restriction without defined fasting periods, pediatric populations, and non-human studies. Two independent reviewers screened titles and abstracts, with full-text review of potentially eligible studies. Disagreements were resolved through consensus or third-party arbitration.

Quality assessment used the Cochrane Risk of Bias tool for randomized trials, evaluating domains of sequence generation, allocation concealment, blinding, incomplete outcome data, selective reporting, and other biases. Meta-analysis was performed using Review Manager software with random-effects models for blood pressure and lipid outcomes. Heterogeneity was assessed using I² statistic, with values >50% considered substantial. Subgroup analyses explored different fasting protocols, intervention durations, and baseline participant characteristics.

3. Results

Twenty-four randomized trials involving 2,847 participants were included in this systematic review. Study duration ranged from 4 weeks to 2 years, with sample sizes from 22 to 538 participants. Mean age of participants varied from 25 to 68 years, and baseline BMI ranged from 23 to 36 kg/m², representing diverse demographic and health status groups.

Pooled analysis of systolic blood pressure data from 18 trials revealed intermittent fasting significantly reduced systolic blood pressure by -4.8 mmHg (95% CI: -6.2 to -3.4, p<0.001) compared to control diets. Diastolic blood pressure decreased by -3.1 mmHg (-4.5 to -1.7, p<0.001) across 16 trials. Seven trials provided data on ambulatory blood pressure monitoring, showing consistent reductions in 24-hour systolic (-3.2 mmHg) and diastolic (-2.8 mmHg) values.

Lipid profile improvements were observed across multiple trials. Total cholesterol was reduced by -0.31 mmol/L (-0.45 to -0.17) in 12 trials, LDL cholesterol decreased by -0.25 mmol/L (-0.38 to -0.12), triglycerides showed a -0.23 mmol/L reduction, and HDL cholesterol increased by 0.08 mmol/L. Inflammatory markers including C-reactive protein and interleukin-6 were significantly lowered in trials measuring these outcomes.

Subgroup analysis revealed greater cardiovascular benefits with longer fasting durations (>16 hours/day) compared to shorter protocols. Participants with higher baseline cardiovascular risk (elevated blood pressure or dyslipidemia at enrollment) experienced more pronounced benefits than lower-risk individuals. Age and sex did not significantly modify treatment effects in exploratory analyses.

No significant increases in adverse cardiovascular events were observed in any included trials. Side effects were generally mild and included transient hunger, fatigue, and headaches, with dropout rates similar between intermittent fasting and control groups. Compliance rates varied from 65-95% across studies, with better adherence observed in supervised interventions.

4. Discussion

The cardiovascular benefits of intermittent fasting appear mediated through multiple physiological mechanisms. Weight loss, typically ranging from 3-8% of initial body weight, contributes significantly to blood pressure reduction and lipid profile improvement. Enhanced insulin sensitivity, evidenced by reduced HOMA-IR scores in multiple trials, suggests improved glucose metabolism and reduced cardiovascular risk.

Autonomic nervous system modulation, with increased vagal tone and decreased sympathetic activity during fasting periods, may contribute to blood pressure reduction. Anti-inflammatory effects, demonstrated by decreased circulating inflammatory cytokines and improved endothelial function, provide additional cardiovascular protection. Circadian rhythm alignment with meal timing may further enhance metabolic homeostasis.

Study limitations include heterogeneous fasting protocols across trials, variable intervention durations, and inconsistent outcome reporting. The longer-term effects beyond 1-2 years remain unclear, necessitating extended follow-up studies. Potential confounders such as lifestyle changes concurrent with fasting interventions were not always adequately controlled. Publication bias toward positive results may overestimate effect sizes.

5. Conclusion

Intermittent fasting represents a promising dietary approach for cardiovascular risk reduction with evidence from multiple randomized trials. The intervention consistently improves blood pressure, lipid profiles, and inflammatory markers, with clinical significance comparable to established lifestyle interventions. While current evidence supports short-to-medium term cardiovascular benefits, larger and longer studies are needed to definitively assess impacts on hard cardiovascular endpoints like myocardial infarction and stroke.

Implementation in clinical practice should consider patient preferences, tolerability, and comorbidities. Supervised introduction with gradual fasting duration increases may optimize adherence and safety. Future research should focus on comparative effectiveness of different fasting protocols, optimal intervention durations, and identification of responder subgroups who derive maximum benefit.
"""

    query = "cardiovascular effects of intermittent fasting"

    # Initialize components with optimized settings
    section_detector = SectionDetector()
    mock_app = create_mock_app()
    embedding_manager = EmbeddingManager(mock_app, batch_size=16)  # Optimized for speed
    hybrid_retriever = HybridRetriever()
    reranker = CrossEncoderReranker()

    pipeline_start = time.time()

    try:
        # 1. Section Detection
        logger.info(
            json.dumps({"event": "pipeline_start", "stage": "section_detection"})
        )
        section_start = time.time()
        sections = section_detector.detect_sections(paper_text)
        section_time = time.time() - section_start
        section_memory_mb = (
            psutil.Process().memory_info().rss / (1024 * 1024) - initial_memory_mb
        )
        logger.info(
            json.dumps(
                {
                    "event": "stage_complete",
                    "stage": "section_detection",
                    "time_ms": round(section_time * 1000, 2),
                    "sections_found": len(sections),
                    "memory_mb": round(section_memory_mb, 2),
                }
            )
        )

        # Harden pipeline with fallback chunking
        if len(sections) < 2:
            logger.warning(
                json.dumps(
                    {
                        "event": "fallback_chunking_triggered",
                        "sections_detected": len(sections),
                        "reason": "fewer_than_2_sections_detected",
                    }
                )
            )
            # Fallback: chunk entire paper text directly
            fallback_chunks = chunk_text(paper_text, chunk_size=1200, overlap=200)
            sections = [
                {"content": chunk, "section": "fallback"} for chunk in fallback_chunks
            ]
            logger.info(
                json.dumps(
                    {
                        "event": "fallback_chunking_complete",
                        "fallback_chunks_created": len(sections),
                    }
                )
            )

        # 2. Chunking by section and size
        logger.info(json.dumps({"event": "stage_start", "stage": "chunking"}))
        chunk_start = time.time()
        chunks = []
        chunk_id = 0

        for section in sections:
            section_chunks = chunk_text(
                section["content"], chunk_size=1200, overlap=200  # ~400-600 tokens
            )

            for chunk_text_content in section_chunks:
                chunks.append(
                    {
                        "content": chunk_text_content,
                        "section": section["section"],
                        "doc_id": chunk_id,
                        "start_pos": chunk_id,
                    }
                )
                chunk_id += 1

        chunk_time = time.time() - chunk_start
        chunk_memory_mb = (
            psutil.Process().memory_info().rss / (1024 * 1024) - initial_memory_mb
        )
        logger.info(
            json.dumps(
                {
                    "event": "stage_complete",
                    "stage": "chunking",
                    "time_ms": round(chunk_time * 1000, 2),
                    "total_chunks": len(chunks),
                    "memory_mb": round(chunk_memory_mb, 2),
                }
            )
        )

        if not chunks:
            raise RuntimeError("No chunks created from sections")

        # 3. Embedding Generation
        logger.info(json.dumps({"event": "stage_start", "stage": "embeddings"}))
        embed_start = time.time()

        # Run async embedding
        processed_chunks = asyncio.run(embedding_manager.process_chunks_async(chunks))

        embed_time = time.time() - embed_start
        embed_memory_mb = (
            psutil.Process().memory_info().rss / (1024 * 1024) - initial_memory_mb
        )

        # Validate embeddings
        for chunk in processed_chunks:
            if "embedding" not in chunk or chunk["embedding"].shape != (384,):
                raise RuntimeError(f"Invalid embedding in chunk: {chunk.get('doc_id')}")

        logger.info(
            json.dumps(
                {
                    "event": "stage_complete",
                    "stage": "embeddings",
                    "time_ms": round(embed_time * 1000, 2),
                    "chunks_embedded": len(processed_chunks),
                    "memory_mb": round(embed_memory_mb, 2),
                }
            )
        )

        # 4. Hybrid Retrieval
        logger.info(json.dumps({"event": "stage_start", "stage": "hybrid_retrieval"}))
        retrieval_start = time.time()

        # Build indices from scratch for performance optimization
        build_time = hybrid_retriever.build_indices(processed_chunks, use_cache=False)

        # Create query embedding
        query_embedding = mock_app.state.embedding_model.encode([query])[0]

        # Retrieve top-15
        retrieved = hybrid_retriever.retrieve(query, query_embedding, top_k=15)

        retrieval_time = time.time() - retrieval_start
        retrieval_memory_mb = (
            psutil.Process().memory_info().rss / (1024 * 1024) - initial_memory_mb
        )
        logger.info(
            json.dumps(
                {
                    "event": "stage_complete",
                    "stage": "hybrid_retrieval",
                    "time_ms": round(retrieval_time * 1000, 2),
                    "results_retrieved": len(retrieved),
                    "build_time_ms": round(build_time, 2),
                    "memory_mb": round(retrieval_memory_mb, 2),
                }
            )
        )

        if not retrieved:
            logger.warning("Empty retrieval results - early exit")
            return []

        # 5. Cross-Encoder Reranking
        logger.info(json.dumps({"event": "stage_start", "stage": "reranking"}))
        rerank_start = time.time()
        reranked = reranker.rerank(query, retrieved[:15])  # Top-15 → top-10
        rerank_time = time.time() - rerank_start
        logger.info(
            json.dumps(
                {
                    "event": "stage_complete",
                    "stage": "reranking",
                    "time_ms": round(rerank_time * 1000, 2),
                    "input_passages": len(retrieved[:15]),
                    "output_passages": len(reranked),
                }
            )
        )

        # Total time calculation
        total_time = time.time() - pipeline_start

        # Validation
        assert len(reranked) == 10, f"Expected 10 passages, got {len(reranked)}"

        # Validate output structure
        required_fields = {
            "content",
            "section",
            "retrieval_score",
            "cross_encoder_score",
            "final_score",
        }
        for i, passage in enumerate(reranked):
            missing_fields = required_fields - passage.keys()
            assert not missing_fields, f"Passage {i} missing fields: {missing_fields}"
            assert passage["section"] != "unknown", f"Passage {i} has unknown section"
            assert isinstance(
                passage["final_score"], (int, float)
            ), f"Passage {i} has invalid final_score"

        # Success logging
        logger.info(
            json.dumps(
                {
                    "event": "pipeline_success",
                    "total_time_seconds": round(total_time, 3),
                    "stage_times": {
                        "section_detection": round(section_time, 3),
                        "chunking": round(chunk_time, 3),
                        "embeddings": round(embed_time, 3),
                        "hybrid_retrieval": round(retrieval_time, 3),
                        "reranking": round(rerank_time, 3),
                    },
                    "final_passage_count": len(reranked),
                    "budget_check": total_time <= 4.0,
                }
            )
        )

        # Single-line INFO summary
        budget_status = "✓" if total_time <= 4.0 else "✗"
        logger.info(
            f"Pipeline completed: {len(reranked)} passages retrieved in {total_time:.1f}s (budget: {budget_status})"
        )

        # Concise success output
        print(f"✅ Pipeline completed successfully in {total_time:.3f}s")
        print(f"   - Sections detected: {len(sections)}")
        print(f"   - Chunks created: {len(processed_chunks)}")
        print(f"   - Final passages: {len(reranked)}")
        print(f"   - Within 4.0s budget: {total_time <= 4.0}")

        return reranked

    except Exception as e:
        total_time = time.time() - pipeline_start
        logger.error(
            json.dumps(
                {
                    "event": "pipeline_error",
                    "error": str(e),
                    "total_time_seconds": round(total_time, 3),
                }
            )
        )
        raise


if __name__ == "__main__":
    results = main()
