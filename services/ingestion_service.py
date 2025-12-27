"""Ingestion service for stages 5-6 of the pipeline.

Responsibilities:
5. PDF Acquisition
6. Processing & Embedding
"""

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from llama_index.core.schema import TextNode
from loguru import logger

from services.exceptions import AcquisitionError, ProcessingError
from utils.embedding_utils import batched_embed

if TYPE_CHECKING:
    from agents.acquisition_agent import AcquisitionAgent
    from agents.processing_agent import ProcessingAgent
    from models.state import Paper, State


@dataclass
class IngestionResult:
    """Result of the ingestion phase."""

    passages: List[Any]
    chunks: List[Any]
    timing_breakdown: Dict[str, float]


class IngestionService:
    """Service for PDF acquisition and processing."""

    def __init__(self, embedding_model: Any = None):
        """Initialize the ingestion service.

        Args:
            embedding_model: Pre-loaded embedding model for chunk embedding.
        """
        self.embedding_model = embedding_model
        self._acquisition_agent = None
        self._processing_agent = None

    def _get_acquisition_agent(self) -> "AcquisitionAgent":
        """Lazy initialization of PDF acquisition agent."""
        if self._acquisition_agent is None:
            from agents.acquisition_agent import AcquisitionAgent

            self._acquisition_agent = AcquisitionAgent()
        return self._acquisition_agent

    def _get_processing_agent(self) -> "ProcessingAgent":
        """Lazy initialization of processing agent."""
        if self._processing_agent is None:
            from agents.processing_agent import ProcessingAgent

            self._processing_agent = ProcessingAgent(
                chunk_size=512,
                chunk_overlap=50,
            )
        return self._processing_agent

    async def ingest(self, finalists: List["Paper"], query: str) -> IngestionResult:
        """Execute the ingestion phase (Stages 5-6).

        Args:
            finalists: List of finalist papers to process.
            query: The original user query (for context).

        Returns:
            IngestionResult containing acquired passages and processed chunks.
        """
        timing_breakdown = {}
        passages = []
        chunks = []

        # Check if embedding model is available
        if self.embedding_model is None:
            logger.warning(
                "⚠️ Embedding model not available - skipping ingestion stages."
            )
            return IngestionResult(passages=[], chunks=[], timing_breakdown={})

        # Stage 5: PDF Acquisition
        logger.info(
            "🧮 Stage 5: PDF Acquisition - downloading PDFs for {} finalists",
            len(finalists),
        )
        stage_start = time.perf_counter()

        # Initialize State internally
        try:
            from models.state import State

            state = State(query=query, finalists=finalists)
        except ImportError:
            # Fallback if State cannot be imported (should not happen)
            logger.error("Failed to import State model")
            raise AcquisitionError("Failed to initialize processing state")

        try:
            acquisition_agent = self._get_acquisition_agent()
            state = await acquisition_agent.process(state)
            passages = state.passages or []
            timing_breakdown["pdf_acquisition"] = time.perf_counter() - stage_start
            logger.info(
                "✅ PDF acquisition completed in {:.3f}s - acquired {} PDFs",
                timing_breakdown["pdf_acquisition"],
                len(passages),
            )
        except asyncio.CancelledError:
            logger.info("🛑 PDF Acquisition cancelled by user")
            raise
        except Exception as e:
            timing_breakdown["pdf_acquisition"] = time.perf_counter() - stage_start
            logger.warning(
                "⚠️ PDF acquisition failed: {}. Continuing with partial results.",
                str(e),
            )
            # Even if acquisition fails partially, we might have some passages?
            # The original code catches exception and continues.
            # If state was modified, we use it.
            passages = state.passages or []

        # Check if we have passages to process
        if not passages:
            logger.warning("⚠️ No PDFs acquired")
            return IngestionResult(
                passages=[], chunks=[], timing_breakdown=timing_breakdown
            )

        # Stage 6: Processing & Embedding
        logger.info(
            "📄 Stage 6: Processing & Embedding - chunking {} passages",
            len(passages),
        )
        stage_start = time.perf_counter()
        try:
            processing_agent = self._get_processing_agent()
            state = await asyncio.to_thread(
                processing_agent.process,
                state,
                embedding_model=self.embedding_model,
                batch_size=32,
            )
            chunks = state.chunks or []
            timing_breakdown["processing"] = time.perf_counter() - stage_start
            logger.info(
                "✅ Processing completed in {:.3f}s - generated {} chunks",
                timing_breakdown["processing"],
                len(chunks),
            )
        except Exception as e:
            logger.exception("Processing failed for {} passages", len(passages))
            raise ProcessingError(f"Failed to process papers: {str(e)}") from e

        return IngestionResult(
            passages=passages,
            chunks=chunks,
            timing_breakdown=timing_breakdown,
        )

    async def ingest_abstracts(self, papers: List["Paper"]) -> List[TextNode]:
        """Ingest abstracts from papers in Fast Mode.

        Args:
            papers: List of Paper objects.

        Returns:
            List of TextNodes with embeddings.
        """
        logger.info("🚀 Fast Mode active: Skipping PDF download, using abstracts.")
        chunks = []

        # Create nodes from abstracts
        for paper in papers:
            abstract = paper.abstract
            if not abstract:
                continue

            # Handle URL extraction safely from Pydantic model
            url = paper.url
            if not url and paper.openAccessPdf:
                url = paper.openAccessPdf.get("url")

            node = TextNode(text=abstract)
            node.metadata = {
                "paper_id": paper.paperId,
                "paper_title": paper.title,
                "url": url,
                "year": paper.year,
                "section": "Abstract",
            }
            chunks.append(node)

        # Generate embeddings
        if chunks:
            texts = [node.get_content() for node in chunks]
            embeddings = await asyncio.to_thread(
                batched_embed, texts, self.embedding_model
            )
            for node, embedding in zip(chunks, embeddings):
                node.embedding = (
                    embedding.tolist()
                    if isinstance(embedding, np.ndarray)
                    else embedding
                )

        return chunks
