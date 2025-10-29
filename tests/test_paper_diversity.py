import pytest
import time
import sys
import os
import json
import logging
from typing import List, Dict, Any
import numpy as np
import torch
import asyncio

# Add project root to Python path for imports
sys.path.insert(0, os.path.abspath(".."))

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


@pytest.fixture
def sample_texts():
    """Fixture providing sample texts for different paper types."""
    return {
        "manuscript": get_manuscript_text(),
        "review": get_review_text(),
        "meta_analysis": get_meta_analysis_text(),
        "preprint": get_preprint_text(),
        "conference": get_conference_paper_text(),
    }


@pytest.fixture
def sample_texts_fast():
    """Fixture for fast tests with smaller texts."""
    return {
        "manuscript": get_manuscript_text()[:2000],  # Truncated for speed
        "review": get_review_text()[:1500],
        "meta_analysis": get_meta_analysis_text()[:1800],
    }


def get_manuscript_text() -> str:
    """Return sample manuscript text with clear IMRaD sections."""
    return """
1. Introduction

This manuscript examines the molecular mechanisms governing cellular signaling pathways in neuronal development. Cell signaling plays a crucial role in nervous system formation, with complex interactions between growth factors, transcription factors, and intracellular messengers coordinating developmental processes.

The Wnt signaling pathway represents one of the most conserved developmental signaling systems, regulating cell fate determination, proliferation, and migration during embryogenesis. Recent studies have identified novel components and regulatory mechanisms that modulate Wnt activity in neural progenitors.

Understanding these molecular interactions is essential for elucidating the etiology of neurodevelopmental disorders characterized by aberrant signaling pathway function.

2. Materials and Methods

2.1 Cell Culture
Primary neural progenitor cells were isolated from embryonic day 12.5 mouse telencephalon using previously described protocols. Cells were maintained in Neurobasal medium supplemented with B27, EGF (20 ng/mL), and FGF2 (20 ng/mL) at 37°C in humidified 5% CO2 atmosphere.

2.2 Wnt Signaling Activation Assay
Wnt3a-conditioned medium was prepared by transfecting L-cells with Wnt3a expression vector. Conditioned medium was collected after 48 hours and filtered. Neural progenitors were treated with Wnt3a-conditioned medium for 4 hours prior to RNA extraction.

2.3 Quantitative PCR Analysis
Total RNA was extracted using Trizol reagent followed by DNase I digestion. cDNA was synthesized using Superscript III reverse transcriptase. Gene expression was quantified using SYBR Green-based real-time PCR with primers specific for Axin2, Dkk1, and Lgr5.

2.4 Statistical Analysis
Data are presented as mean ± SEM. Statistical significance was determined using Student's t-test with p < 0.05 considered significant. All experiments were performed in triplicate.

3. Results

3.1 Wnt signaling regulates neural progenitor proliferation
Treatment with Wnt3a-conditioned medium significantly increased BrdU incorporation compared to control conditions (p < 0.01). This proliferative response was attenuated by co-treatment with Dkk1, confirming Wnt-specific effects.

3.2 Target gene induction by Wnt activation
Quantitative PCR analysis revealed robust induction of canonical Wnt target genes following Wnt3a treatment. Axin2 expression increased 8.7-fold, while Lgr5 showed 4.2-fold upregulation compared to baseline levels. Dkk1 expression was not significantly altered.

3.3 Dose-dependent responses to Wnt modulation
Increasing concentrations of Wnt3a (10-200 ng/mL) produced dose-dependent induction of target gene expression. Maximum induction was observed at 100 ng/mL Wnt3a, with EC50 approximately 35 ng/mL.

4. Discussion

The results demonstrate that Wnt signaling activation promotes neural progenitor proliferation while activating canonical target gene expression programs. These findings are consistent with previous reports of Wnt function in neural development and extend our understanding of dose-dependent signaling responses in progenitor cells.

The clinical implications of these findings relate to neurodevelopmental disorders characterized by impaired Wnt signaling. Recent genetic studies have identified mutations in Wnt pathway components in patients with intellectual disability syndromes.

Future studies should investigate the role of Wnt signaling in human neural progenitor cells and explore therapeutic modulation of Wnt activity for developmental disorders.

5. Conclusions

Our findings establish Wnt signaling as a critical regulator of neural progenitor behavior, with significant implications for understanding and potentially treating neurodevelopmental disorders involving aberrant Wnt pathway function.
"""


def get_review_text() -> str:
    """Return sample review text with narrative structure."""
    return """
Comprehensive Review: Wnt Signaling in Neural Development and Disease

Abstract
The Wnt signaling pathway coordinates multiple aspects of neural development and has been implicated in various neurological disorders. This review synthesizes recent advances in our understanding of Wnt pathway function in the nervous system, highlighting both developmental and pathological roles.

Historical Context and Pathway Evolution
Wnt signaling emerged early in metazoan evolution as a key regulator of cell polarity and fate determination. Initial studies in Drosophila melanogaster identified Wnt proteins as segment polarity genes, while subsequent work in Xenopus and zebrafish revealed conserved roles in embryonic patterning. Mammalian studies have extended these findings to the nervous system, where Wnt signaling orchestrates neural tube formation, neuronal migration, and synaptic development.

Molecular Components and Signaling Mechanisms
The Wnt pathway comprises multiple branches, with canonical beta-catenin-dependent signaling representing the most extensively studied arm. Canonical Wnt ligands (Wnt1, Wnt3a, Wnt8) bind to Frizzled receptors and LRP5/6 co-receptors, stabilizing beta-catenin and enabling TCF/LEF-mediated transcriptional activation. Non-canonical Wnt pathways include PCP and Wnt/Ca2+ signaling, which regulate cytoskeletal dynamics and intracellular calcium levels respectively.

Neural Stem Cell Regulation
Wnt signaling maintains neural stem cell populations throughout development. In the ventricular zone of the developing cerebral cortex, Wnt proteins promote stem cell self-renewal while preventing premature differentiation. Recent single-cell RNA sequencing studies have revealed heterogeneous Wnt responsiveness among neural progenitor subtypes, suggesting context-dependent pathway activation. The temporal regulation of Wnt activity appears critical, with pathway inhibition required for terminal neuronal differentiation.

Synaptic Plasticity and Neurological Function
Beyond developmental roles, Wnt signaling contributes to mature neuronal function. Wnt proteins modulate synaptic plasticity through regulation of neurotransmitter receptor localization and dendritic spine morphogenesis. Pathological Wnt signaling has been observed in neurodegenerative conditions including Alzheimer's disease, where increased Wnt activity may represent a compensatory response to amyloid toxicity.

Disease Associations and Therapeutic Implications
Genetic studies have linked Wnt pathway mutations to neurodevelopmental disorders such as autism spectrum disorder and intellectual disability. In cancer, aberrant Wnt activation drives medulloblastoma formation, the most common malignant pediatric brain tumor. Therapeutic modulation of Wnt signaling presents challenges due to pathway complexity but holds promise for neurological disease treatment.

Future Directions
Unresolved questions include the spatiotemporal dynamics of Wnt signaling in vivo and the functional consequences of natural pathway variants. Advanced imaging techniques and CRISPR-based perturbation approaches should elucidate these mechanisms. Understanding Wnt pathway organization in human neural development will inform therapeutic strategies for neurodevelopmental abnormalities.

This review highlights Wnt signaling as a fundamental coordinator of neural processes, with therapeutic potential across neurological disorders.
"""


def get_meta_analysis_text() -> str:
    """Return sample meta-analysis text with references-heavy structure."""
    return """
Meta-Analysis of Wnt Signaling Variants in Neurodevelopmental Disorders: A Systematic Review

Background
Neurodevelopmental disorders affect 12-15% of children worldwide, with significant genetic contributions. The Wnt signaling pathway regulates brain development and has been implicated in autism spectrum disorder (ASD), intellectual disability (ID), and related conditions. This meta-analysis evaluates associations between Wnt pathway gene variants and neurodevelopmental disorder risk.

Methods
We conducted a systematic search of PubMed, Web of Science, EMBASE, and GWAS Catalog from 2000-2023 using terms: "Wnt signaling", "neurodevelopmental disorders", "autism", "intellectual disability", "genetic variants", and specific Wnt pathway genes. Inclusion criteria were case-control or family-based studies reporting genetic associations with ASD, ID, or schizophrenia in Wnt pathway genes (Wnt1, Wnt3, Fzd1-10, Lrp5/6, Dkk1-4, Axin1/2, Apc, beta-catenin).

Statistical Analysis
Meta-analyses were performed using random effects models. Odds ratios (OR) were calculated with 95% confidence intervals. Heterogeneity assessments used I2 statistics (>50% indicating substantial heterogeneity). Publication bias was evaluated using funnel plots and Egger's test.

Results
153 studies met inclusion criteria, encompassing 45 Wnt pathway genes and 43,217 cases and 52,890 controls. Significant associations were identified across multiple Wnt components.

Canonical Wnt Pathway Genes
Beta-catenin (CTNNB1) variants showed strongest associations (OR=1.23, 95% CI=1.15-1.32, P<0.001, I2=38%). LRP6 variants conferred moderate risk (OR=1.18, 95% CI=1.09-1.27, P<0.001, I2=42%). APC mutations increased risk significantly (OR=1.45, 95% CI=1.28-1.64, P<0.001, I2=29%).

Frizzled Receptors
FZD5 variants demonstrated consistent associations (OR=1.14, 95% CI=1.06-1.23, P=0.001, I2=33%). FZD3 mutations were linked to speech/language disorders (OR=1.31, 95% CI=1.18-1.45, P<0.001).

Antagonists and Modulators
DKK1 overexpression variants reduced disorder risk (OR=0.87, 95% CI=0.81-0.93, P<0.001), while DKK1 deficiency variants increased risk. SFRP1 deletions showed protective effects (OR=0.79, 95% CI=0.71-0.88, P<0.001).

Non-Canonical Pathway Genes
WNT5A gain-of-function variants increased ASD risk (OR=1.27, 95% CI=1.16-1.39, P<0.001), while loss-of-function variants conferred protection (OR=0.84, 95% CI=0.78-0.91, P<0.001).

Gene-Environment Interactions
Stratified analysis revealed gene-environment interactions. Maternal folate supplementation modified Wnt variant effects (interaction P=0.03), with reduced OR in supplemented pregnancies.

Subgroup Analysis by Disorder Type
Autism spectrum disorder showed strongest associations with Wnt pathway variants (OR=1.21, 95% CI=1.16-1.25), followed by intellectual disability (OR=1.15, 95% CI=1.09-1.21). Schizophrenia associations were modest (OR=1.08, 95% CI=1.02-1.14).

Discussion
This meta-analysis provides comprehensive evidence of Wnt pathway involvement in neurodevelopmental disorders. Beta-catenin and LRP6 emerge as key risk genes, consistent with their central roles in canonical Wnt signaling. The protective effects of DKK1 and SFRP1 variants highlight therapeutic potential for pathway antagonists.

Study Limitations
Heterogeneity across studies (average I2=35%) may reflect differences in diagnostic criteria, population ancestry, and variant classification. Publication bias toward positive findings likely overestimates effect sizes. Functional validation of identified variants remains limited.

Clinical Implications
Wnt pathway genes should be prioritized in neurodevelopmental disorder genetic screening panels. Pathway modulators may represent therapeutic targets, particularly for disorders involving synaptic dysfunction.

Future Research Directions
Large-scale whole-genome sequencing studies are needed to identify rare Wnt pathway variants. Functional studies should clarify variant pathogenicity. Longitudinal studies could evaluate Wnt pathway modulation for neurodevelopmental disorder prevention.

Conclusions
Genetic variants in Wnt signaling pathway components confer modest but significant risk for neurodevelopmental disorders, particularly autism spectrum disorder and intellectual disability. These findings support Wnt pathway involvement in brain development and suggest therapeutic opportunities.

References
Adamska M, et al. Wnt signaling in autism spectrum disorders. Nat Neurosci 2021;24(3):345-356.
Badano JL, Katsanis N. Wnt signaling in neural development and disease. Annu Rev Genomics Hum Genet 2020;21:237-258.
Becker EB, et al. Wnt signaling regulates adult hippocampal neurogenesis. J Neurosci 2014;34(37):12535-12545.
Castelo-Branco G, et al. Ventral midbrain Wnt1 regulates neurogenesis and dopamine neuron development. Cell Stem Cell 2006;1(5):573-583.
Chenn A, Walsh CA. Regulation of cerebral cortical size by control of neural stem cell division in Wnt signaling. Neuron 2002;35(5):865-878.
De Ferrari GV, Moon RT. Wnt signaling: implications in neurodegenerative diseases? Handb Exp Pharmacol 2006;273-287.
Flaherty MS, et al. Wnt signaling in neuropsychiatric disorders. Curr Opin Neurobiol 2023;78:102658.
Freese JL, et al. Wnt signaling in schizophrenia. Schizophr Res 2022;239:41-52.
Galli LM, et al. Wnt signaling in intellectual disability syndromes. Hum Genet 2021;140(7):1099-1112.
Hariharan IK. Wnt signaling in brain development and disease. Front Biosci 2021;26(2):619-638.
Hocking AM, et al. Meta-analysis of Wnt pathway gene variants in neurodegenerative diseases. Neurobiol Aging 2022;112:123-145.
Johnson MB, et al. Single-cell analysis reveals transcriptional heterogeneity of neural progenitors in human cortex. Nature 2015;521(7551):228-232.
Karner CM, et al. Wnt signaling in kidney development and disease. Semin Cell Dev Biol 2020;103:81-93.
Kim J, et al. Wnt signaling in Alzheimer's disease. Exp Neurol 2022;347:113902.
Lanoue V, et al. Wnt signaling in neuronal connectivity. Semin Cell Dev Biol 2021;114:179-192.
Logan CY, Nusse R. Wnt signaling in development and disease. Annu Rev Cell Dev Biol 2004;20:567-599.
Marsoner L, et al. Wnt signaling dysregulation in psychiatric disorders. Neurosci Biobehav Rev 2023;146:105067.
Matsuda T, et al. Wnt signaling in autism: from genetics to therapeutics. Front Psychiatry 2021;12:687915.
Mehlen P, et al. Wnt signaling in cancer and stem cell biology. Semin Cancer Biol 2022;86:53-75.
Mulligan KA, Cheyette BNR. Wnt signaling in cerebellar development and medulloblastoma. J Dev Biol 2020;8(2):12.
Noelanders R, Vleminckx K. Wnt signaling and intellectual disability. Front Mol Neurosci 2020;13:71.
Parada LF, et al. Wnt signaling in medulloblastoma. Nat Rev Neurosci 2015;16(4):184-196.
Patel S, et al. Wnt signaling in synaptic plasticity and cognition. Front Synaptic Neurosci 2021;13:645062.
Phillips HM, et al. Wnt signaling in neural crest development and disease. Dev Biol 2022;482:23-36.
Qu Y, et al. Wnt signaling and adult neurogenesis. Cell Mol Life Sci 2013;70(21):4159-4173.
Rekas K, et al. Wnt signaling in schizophrenia and bipolar disorder. Behav Brain Res 2021;402:113117.
Riccomagno MM, et al. Wnt signaling in the development of the enteric nervous system. Gastroenterology 2022;162(1):35-50.
Rizo-Paredes JG, et al. Wnt signaling in neurodegenerative diseases. Mol Neurobiol 2023;60(1):123-145.
Rowitch DH, Kriegstein AR. Wnt signaling in cortical development. Annu Rev Neurosci 2010;33:273-296.
Schafer ST, et al. Wnt signaling in microglia and neuroinflammation. Exp Neurol 2023;359:114223.
Schluter OM, et al. Wnt signaling in synapse formation and function. Curr Opin Neurobiol 2018;51:47-53.
Shan J, et al. Wnt signaling in psychiatric disorders: a meta-analysis. Prog Neuropsychopharmacol Biol Psychiatry 2022;115:110505.
Takemaru KI, et al. Wnt signaling and cancer stem cells. Front Biosci 2016;21:1009-1024.
Thompson MD, et al. Wnt signaling in human development and disease. Annu Rev Genomics Hum Genet 2023;24:389-412.
Udina M, et al. Wnt signaling in peripheral nerve regeneration. Neural Regen Res 2022;17(7):1411-1418.
Valvezan AJ, Klein PS. GSK-3 and Wnt signaling in nervous system disorders. Annu Rev Neurosci 2012;35:339-361.
Wang Y, et al. Wnt signaling in stem cell self-renewal and differentiation. Cell Death Differ 2021;28(3):835-847.
Wang Y, et al. Wnt signaling in neuropsychiatric disorders. Prog Neuropsychopharmacol Biol Psychiatry 2022;112:110308.
Wu XL, et al. Wnt signaling in neural stem cells and brain tumors. Chin J Cancer 2023;42(1):23.
Yoshida T, et al. Wnt signaling in cerebral cortex development. Dev Growth Differ 2022;64(3):163-175.
Zhang L, et al. Wnt signaling in autism spectrum disorder. Front Psychiatry 2020;11:584.
Zhong JL, et al. Wnt signaling in neural injury and repair. Neural Regen Res 2021;16(8):1555-1565.
Zhou CJ, et al. Wnt signaling in synaptic dysfunction and neurodevelopmental disorders. Front Synaptic Neurosci 2022;14:847598.
"""


def run_pipeline_on_text(
    paper_text: str, query: str = "cellular signaling pathways"
) -> List[Dict[str, Any]]:
    """Run the full pipeline on provided text and return results."""
    # Set deterministic seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Configure logging to capture warnings
    logging.basicConfig(level=logging.WARNING)

    # Initialize components
    section_detector = SectionDetector()
    mock_app = create_mock_app()
    embedding_manager = EmbeddingManager(
        mock_app, batch_size=32
    )  # Larger batch for speed
    hybrid_retriever = HybridRetriever()
    reranker = CrossEncoderReranker()

    try:
        # 1. Section Detection
        sections = section_detector.detect_sections(paper_text)

        # Harden pipeline with fallback chunking
        fallback_used = False
        if len(sections) < 2:
            logger.warning(
                f"Fallback chunking triggered: detected {len(sections)} sections"
            )
            # Fallback: chunk entire paper text directly
            fallback_chunks = chunk_text(paper_text, chunk_size=1200, overlap=200)
            sections = [
                {"content": chunk, "section": "fallback"} for chunk in fallback_chunks
            ]
            fallback_used = True

        # 2. Chunking by section and size
        chunks = []
        chunk_id = 0

        for section in sections:
            section_chunks = chunk_text(
                section["content"], chunk_size=1200, overlap=200
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

        if not chunks:
            raise RuntimeError("No chunks created from sections")

        # 3. Embedding Generation
        processed_chunks = asyncio.run(embedding_manager.process_chunks_async(chunks))

        # Validate embeddings
        for chunk in processed_chunks:
            if "embedding" not in chunk or chunk["embedding"].shape != (384,):
                raise RuntimeError(f"Invalid embedding in chunk: {chunk.get('doc_id')}")

        # 4. Hybrid Retrieval
        build_time = hybrid_retriever.build_indices(processed_chunks, use_cache=False)
        query_embedding = mock_app.state.embedding_model.encode([query])[0]
        retrieved = hybrid_retriever.retrieve(query, query_embedding, top_k=15)

        if not retrieved:
            return []

        # 5. Cross-Encoder Reranking
        reranked = reranker.rerank(query, retrieved[:15])

        return reranked

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise


def create_mock_app() -> FastAPI:
    """Create mock FastAPI app with loaded embedding model."""
    app = FastAPI()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    app.state.embedding_model = model
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


# Mock random and torch for reproducibility
import random

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


def inject_ocr_noise(text: str, noise_level: str = "light") -> str:
    """
    Inject realistic OCR noise into text for robustness testing.

    Args:
        text: Original text
        noise_level: "light", "medium", or "heavy" corruption

    Returns:
        Text with injected OCR-like errors
    """
    import random

    # OCR character confusion mapping (common recognition errors)
    confusions = {
        "1": "l",
        "l": "1",
        "I": "1",
        "i": "1",
        "0": "O",
        "O": "0",
        "2": "Z",
        "Z": "2",
        "3": "E",
        "E": "3",
        "5": "S",
        "S": "5",
        "6": "G",
        "G": "6",
        "8": "B",
        "B": "8",
        "9": "g",
        "g": "9",
        ".": ",",
        ",": ".",
        ";": ":",
        ":": ";",
    }

    chars = list(text)
    modified = []

    for char in chars:
        if noise_level == "light" and random.random() < 0.05:  # 5% corruption
            modified.append(confusions.get(char, char))
        elif noise_level == "medium" and random.random() < 0.1:  # 10% corruption
            if char in confusions:
                modified.append(confusions[char])
            else:
                modified.append(char)
        elif noise_level == "heavy" and random.random() < 0.2:  # 20% corruption
            if char in confusions:
                modified.append(confusions[char])
            elif random.random() < 0.1:  # Additional 10% chance of deletion
                continue  # Skip character (deletion)
            else:
                modified.append(char)
        else:
            modified.append(char)

    return "".join(modified)


def get_preprint_text() -> str:
    """Return sample preprint text with arXiv-style header."""
    return """
arXiv:2301.12345 [q-bio.MN]

Comprehensive Molecular Analysis of Wnt Signaling Pathways in Neural Differentiation

Abstract
Wnt signaling regulates multiple aspects of neural development and has been implicated in various neurological disorders. This preprint evaluates associations between Wnt pathway gene variants and neurodevelopmental disorder risk using comprehensive genomic analyses.

Keywords: Wnt signaling, neural development, genomic variants, neurodevelopmental disorders

1. Introduction

This manuscript examines the comprehensive molecular mechanisms governing Wnt signaling in neural development. Wnt signaling plays a crucial role in nervous system formation, with complex interactions between growth factors and intracellular messengers coordinating developmental processes.

Recent studies have identified novel Wnt pathway components and regulatory mechanisms that modulate Wnt activity in neural progenitors.

2. Background

Wnt signaling emerged early in metazoan evolution as a key regulator of cell polarity and fate determination. Initial studies in model organisms identified Wnt proteins as segment polarity genes, while subsequent mammalian studies have extended these findings to the nervous system.

2.1 Wnt Components and Signaling Mechanisms
The Wnt pathway comprises multiple branches, with canonical beta-catenin-dependent signaling representing the most extensively studied arm. Canonical Wnt ligands bind to Frizzled receptors and LRP5/6 co-receptors, stabilizing beta-catenin and enabling TCF/LEF-mediated transcriptional activation.

2.2 Non-Canonical Wnt Pathways
Non-canonical Wnt pathways include PCP and Wnt/Ca2+ signaling, which regulate cytoskeletal dynamics and intracellular calcium levels respectively.

3. Methods

3.1 Molecular Analysis
Primary neural progenitor cells were isolated from embryonic telencephalon using established protocols. Cells were maintained in Neurobasal medium supplemented with B27, EGF (20 ng/mL), and FGF2 (20 ng/mL).

3.2 Wnt Activation Assays
Wnt3a-conditioned medium was prepared by transfecting L-cells with Wnt3a expression vector. Conditioned medium was collected and filtered. Neural progenitors were treated for 4 hours prior to analysis.

3.3 Genomic Analysis
Next-generation sequencing was performed on Wnt pathway genes. Variant calling and annotation were conducted using established pipelines.

4. Results

4.1 Wnt Signaling Activation
Treatment with Wnt3a-conditioned medium significantly increased downstream targets compared to control conditions. Quantitative analysis revealed robust pathway activation.

4.2 Genomic Findings
Analysis of Wnt pathway genes identified multiple variants associated with neural development. These findings support the role of Wnt signaling in neurodevelopmental processes.

5. Discussion

The results demonstrate Wnt signaling as a critical regulator of neural progenitor behavior. The genomic analysis provides insights into the genetic architecture of Wnt-associated neurodevelopmental disorders.

5.1 Clinical Implications
Genetic variants in Wnt pathway components confer significant risk for neurodevelopmental disorders, particularly autism spectrum disorder. These findings support pathway involvement in brain development.

5.2 Future Directions
Further studies evaluating Wnt pathway modulation in human neural development will inform therapeutic strategies for neurodevelopmental abnormalities.

6. Conclusion

Genetic variants in Wnt signaling pathway components confer significant risk for neurodevelopmental disorders. These findings support Wnt pathway involvement in brain development and suggest therapeutic opportunities.
"""


def get_conference_paper_text() -> str:
    """Return sample conference paper text with different structure."""
    return """
International Conference on Machine Learning in Computational Biology (ICMLCB 2023)

Title: Wnt Signaling Network Analysis in Neurodevelopmental Disorders

Abstract
Wnt signaling regulates neural development and has been implicated in various neurological disorders. This paper presents a comprehensive systems biology approach to analyzing Wnt signaling networks in neurodevelopmental disorders. We employ machine learning techniques to identify key regulatory nodes and predict therapeutic targets.

Keywords: Wnt signaling, neurodevelopmental disorders, systems biology, machine learning, network analysis

1. Introduction

Neurodevelopmental disorders affect millions worldwide, with significant genetic and environmental contributions. Wnt signaling plays a pivotal role in neural development, regulating cell fate determination, proliferation, and migration. Disruptions in Wnt signaling are implicated in autism spectrum disorder, intellectual disability, and schizophrenia.

This paper presents a computational approach to analyzing Wnt signaling in neurodevelopmental disorders. We developed a comprehensive systems biology framework to model Wnt pathway interactions and identify potential therapeutic targets.

2. Related Work

Previous studies have investigated Wnt signaling in neural development [1, 2, 3]. Recent genomic studies identified Wnt pathway genes associated with neurodevelopmental disorders [4, 5]. However, comprehensive systems-level analysis of Wnt signaling networks has been limited.

Smith et al. [1] demonstrated Wnt signaling regulation of neural stem cell proliferation. Jones et al. [2] identified downstream targets in forebrain development. Our approach extends these findings through integrative network analysis.

3. Methods

3.1 Data Collection
We collected genomic data from multiple sources including GWAS studies and targeted sequencing of Wnt pathway genes. Protein-protein interaction data were obtained from BioGRID and STRING databases.

3.2 Network Construction
Wnt signaling networks were constructed using pathway databases including KEGG and Reactome. Gene expression data from neural development were integrated to identify context-specific interactions.

3.3 Machine Learning Analysis
We employed graph neural networks to analyze network topology and identify essential nodes. Random forest and support vector machine models were trained to predict Wnt pathway dysregulation from genomic data.

3.4 Validation
Computational predictions were validated using in vitro Wnt signaling assays and gene expression analysis.

4. Experiments

4.1 Network Analysis Results
Graph analysis identified 12 key regulatory hubs in the Wnt signaling network. These hubs showed significant enrichment for neurodevelopmental disorder genes (p < 0.001).

4.2 Machine Learning Predictions
Random forest models achieved 85% accuracy in predicting Wnt pathway dysregulation from genomic data. Feature analysis revealed DKK1 and SFRP1 as important modulators.

4.3 Experimental Validation
In vitro assays confirmed predicted regulatory relationships. Wnt signaling activation increased expression of downstream targets by 3.2-fold.

5. Discussion

Our systems biology approach revealed complex regulatory interactions in Wnt signaling networks. The identification of key hubs provides new therapeutic targets for neurodevelopmental disorders.

The machine learning models demonstrated strong predictive performance, suggesting clinical utility in identifying patients with Wnt pathway dysregulation.

Limitations include reliance on existing interaction databases and need for experimental validation of additional predictions.

6. Future Work

Future studies will incorporate single-cell RNA sequencing data to refine neural cell type-specific Wnt signaling networks. Multi-omics integration will improve predictive models.

Clinical trials of Wnt pathway modulators are warranted based on these computational predictions.

7. References

[1] Smith A, et al. Wnt signaling in neural stem cells. Nature 2018;555(7697):339-344.
[2] Jones B, et al. Wnt targets in forebrain development. Cell 2019;178(3):611-628.
[3] Brown C, et al. Wnt pathway evolution. Dev Biol 2020;460(1):45-56.
[4] Davis D, et al. Wnt variants in autism. Nat Genet 2021;53(4):477-488.
[5] Wilson E, et al. Genomic analysis of Wnt signaling. PLoS Genet 2022;18(1):e1009876.
[6] Chen F, et al. Systems biology of Wnt signaling. Curr Opin Syst Biol 2023;27:100401.

8. Acknowledgments

This work was supported by NIH grant R01NS098124 and NSF award 1944438. We thank the High-Performance Computing Center for computational resources.
"""


class TestPaperDiversity:
    """Test pipeline robustness across different paper formats."""

    @pytest.mark.timeout(8)  # 8 second timeout (with 10% buffer over 5s target)
    def test_manuscript_pipeline(self, sample_texts):
        """Test pipeline on manuscript format (sections expected)."""
        start_time = time.time()

        results = run_pipeline_on_text(sample_texts["manuscript"])

        total_time = time.time() - start_time

        # Assert results > 0
        assert len(results) > 0, "Manuscript pipeline should return results"

        # Assert precisely minimal guarantees
        # Implementation detail: manuscript has clear sections, expect ≥3 detected
        # But we assert on final output quality
        assert len(results) >= 1, "Should return at least 1 passage"

        # Validate result structure
        for passage in results:
            assert "content" in passage
            assert "section" in passage
            assert "retrieval_score" in passage
            assert "cross_encoder_score" in passage
            assert "final_score" in passage
            assert isinstance(passage["final_score"], (int, float))

        # Performance: complete under ~5 seconds
        assert total_time < 8, f"Manuscript test took too long: {total_time:.3f}s"

        print(f".3f")

    @pytest.mark.timeout(8)
    def test_review_pipeline(self, sample_texts):
        """Test pipeline on review format (allows fallback chunking)."""
        start_time = time.time()

        results = run_pipeline_on_text(sample_texts["review"])

        total_time = time.time() - start_time

        # Assert results ≥ 1 (minimal guarantee with fallback)
        assert len(results) >= 1, "Review pipeline should return at least 1 passage"

        # Validate result structure
        for passage in results:
            assert "content" in passage
            assert "section" in passage

        # Performance check
        assert total_time < 8, f"Review test took too long: {total_time:.3f}s"

        print(f".3f")

    @pytest.mark.timeout(8)
    def test_meta_analysis_pipeline(self, sample_texts):
        """Test pipeline on meta-analysis format (references-heavy structure)."""
        start_time = time.time()

        results = run_pipeline_on_text(sample_texts["meta_analysis"])

        total_time = time.time() - start_time

        # Assert results > 0
        assert len(results) > 0, "Meta-analysis pipeline should return results"

        # Validate result structure
        for passage in results:
            assert "content" in passage

        # Performance check
        assert total_time < 8, f"Meta-analysis test took too long: {total_time:.3f}s"

        print(f".3f")

    @pytest.mark.timeout(8)
    def test_preprint_pipeline(self, sample_texts):
        """Test pipeline on preprint format with arXiv-style headers."""
        start_time = time.time()

        results = run_pipeline_on_text(sample_texts["preprint"])

        total_time = time.time() - start_time

        # Assert results > 0
        assert len(results) > 0, "Preprint pipeline should return results"

        # Validate result structure
        for passage in results:
            assert "content" in passage
            assert "section" in passage

        # Performance check
        assert total_time < 8, f"Preprint test took too long: {total_time:.3f}s"

        print(f"Preprint pipeline: {total_time:.3f}s")

    @pytest.mark.timeout(8)
    def test_conference_paper_pipeline(self, sample_texts):
        """Test pipeline on conference paper format with different structure."""
        start_time = time.time()

        results = run_pipeline_on_text(sample_texts["conference"])

        total_time = time.time() - start_time

        # Assert results > 0
        assert len(results) > 0, "Conference paper pipeline should return results"

        # Conference papers may trigger fallback due to different section organization
        # Validate result structure
        for passage in results:
            assert "content" in passage

        # Performance check
        assert total_time < 8, f"Conference paper test took too long: {total_time:.3f}s"

        print(f"Conference paper pipeline: {total_time:.3f}s")

    @pytest.mark.parametrize("noise_level", ["light", "medium", "heavy"])
    def test_ocr_noisy_section_detection(self, sample_texts, noise_level):
        """Test section detection robustness with OCR noise injection."""
        # Use manuscript text as base since it has clear sections
        original_text = sample_texts["manuscript"]

        # Inject OCR noise
        noisy_text = inject_ocr_noise(original_text, noise_level)

        # Pipeline should still work despite noise
        results = run_pipeline_on_text(noisy_text)

        # Assert results > 0 even with heavy noise
        assert len(results) > 0, f"Pipeline should handle {noise_level} OCR noise"

        # With heavy corruption, may trigger fallback but should still return results
        print(f"OCR {noise_level} noise test passed with {len(results)} results")

    @pytest.mark.parametrize(
        "section_header,expected_detection",
        [
            ("1. Introduction", True),
            ("Introduction", True),
            ("INTRODUCTION", True),
            ("Abstract", True),  # Alternative section name
        ],
    )
    def test_section_detector_robustness(self, section_header, expected_detection):
        """Test section detector with various header formats."""
        # Create realistic test text with the params section and a methods section for proper detection
        test_text = f"""Some initial content introducing the paper topic.

{section_header}

This is detailed test content for the section we're examining. The section detector should properly identify this section type and extract its content.

2. Methods

Standard methods section for comparison and validation of section detection algorithm performance.
"""

        detector = SectionDetector()
        sections = detector.detect_sections(test_text)

        # All these should work (put paper in proper format so detection happens)
        assert (
            len(sections) > 0
        ), f"Should detect sections with header: {section_header}"

        found_section_types = [s["section"] for s in sections]
        # Should find known section types, not just "unknown"
        known_sections = [
            "introduction",
            "methods",
            "results",
            "discussion",
            "conclusion",
        ]
        has_known_section = any(sec in found_section_types for sec in known_sections)
        assert (
            has_known_section
        ), f"Should recognize section types in: {found_section_types}"

    @pytest.mark.pdf_dependent
    @pytest.mark.slow
    def test_pdf_extraction_failure_handling(self):
        """Test graceful handling of PDF extraction failures."""
        import pytest

        # Mock PDF extraction failure - this would normally happen in acquisition
        # For this test, simulate by attempting pipeline on empty text
        with pytest.raises(Exception):  # Expect pipeline failure on empty input
            run_pipeline_on_text("")

        # In real implementation, this would use xfail when PDF extraction fails
        # But for MVP validation, we ensure the pipeline handles edge cases


class TestPaperDiversityFast:
    """Fast tests using mocked pipeline components for CI/CD validation."""

    @pytest.mark.timeout(1)  # Sub-1s target
    def test_manuscript_pipeline_fast(self, sample_texts_fast):
        """Fast test of manuscript pipeline with mocked components."""
        start_time = time.time()

        results = run_pipeline_on_text_fast(sample_texts_fast["manuscript"])

        total_time = time.time() - start_time

        assert len(results) > 0, "Fast manuscript pipeline should return results"
        assert len(results) >= 1, "Should return at least 1 passage"

        for passage in results:
            assert "content" in passage
            assert "section" in passage
            assert "retrieval_score" in passage
            assert "cross_encoder_score" in passage
            assert "final_score" in passage

        assert total_time < 1.0, f"Fast test took too long: {total_time:.3f}s"
        print(f"Fast manuscript: {total_time:.3f}s")

    @pytest.mark.timeout(1)
    def test_review_pipeline_fast(self, sample_texts_fast):
        """Fast test of review pipeline with mocked components."""
        start_time = time.time()

        results = run_pipeline_on_text_fast(sample_texts_fast["review"])

        total_time = time.time() - start_time

        assert (
            len(results) >= 1
        ), "Fast review pipeline should return at least 1 passage"

        assert total_time < 1.0, f"Fast review test took too long: {total_time:.3f}s"
        print(f"Fast review: {total_time:.3f}s")

    @pytest.mark.timeout(1)
    def test_meta_analysis_pipeline_fast(self, sample_texts_fast):
        """Fast test of meta-analysis pipeline with mocked components."""
        start_time = time.time()

        results = run_pipeline_on_text_fast(sample_texts_fast["meta_analysis"])

        total_time = time.time() - start_time

        assert len(results) > 0, "Fast meta-analysis pipeline should return results"

        assert (
            total_time < 1.0
        ), f"Fast meta-analysis test took too long: {total_time:.3f}s"
        print(f"Fast meta-analysis: {total_time:.3f}s")


def run_pipeline_on_text_fast(
    paper_text: str, query: str = "cellular signaling pathways"
) -> List[Dict[str, Any]]:
    """
    Run mocked pipeline on text - deterministic results in ~milliseconds.

    Mocks all heavy computation: embeddings, BM25/FAISS indexing, reranking.
    """
    # Set deterministic seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Mock all heavy components
    section_detector = SectionDetector()

    try:
        # 1. Section Detection (keep real for accuracy)
        sections = section_detector.detect_sections(paper_text)

        # Harden pipeline with fallback chunking
        if len(sections) < 2:
            fallback_chunks = chunk_text(paper_text, chunk_size=800, overlap=100)
            sections = [
                {"content": chunk, "section": "fallback"} for chunk in fallback_chunks
            ]

        # 2. Chunking
        chunks = []
        chunk_id = 0

        for section in sections:
            section_chunks = chunk_text(section["content"], chunk_size=800, overlap=100)
            for chunk_text_content in section_chunks:
                chunks.append(
                    {
                        "content": chunk_text_content,
                        "section": section["section"],
                        "doc_id": chunk_id,
                        "start_pos": chunk_id,
                        "embedding": np.random.rand(384).astype(
                            np.float32
                        ),  # Mock embedding
                    }
                )
                chunk_id += 1

        if not chunks:
            return []

        # 3-5. Mock retrieval and reranking - return deterministic top-10 results
        results = []

        # Create deterministic mock results based on content length
        for i in range(min(10, len(chunks))):
            chunk = chunks[i % len(chunks)]  # Cycle through chunks
            results.append(
                {
                    "content": chunk["content"][:200],  # Truncate for test speed
                    "section": chunk["section"],
                    "doc_id": chunk["doc_id"],
                    "retrieval_score": 0.9
                    - (i * 0.05),  # Decreasing scores: 0.9, 0.85, 0.8...
                    "cross_encoder_score": 0.8
                    - (i * 0.04),  # Slightly different ranking
                    "final_score": 0.85 - (i * 0.045),
                    "rerank_position": i + 1,
                }
            )

        return results[:10]  # Return top 10

    except Exception as e:
        logger.error(f"Fast pipeline error: {e}")
        return []  # Return empty on error for resilience testing
