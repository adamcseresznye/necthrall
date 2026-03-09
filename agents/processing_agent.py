from __future__ import annotations

import time
from typing import Any, List

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from loguru import logger

from models.state import State


class ProcessingAgent:
    """State-aware processing agent that chunks texts.

    Usage:
        agent = ProcessingAgent(chunk_size=500, chunk_overlap=50)
        updated_state = agent.process(state)
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

        # Single-stage parsing: split by token size/sentences
        # This is sufficient for plain text extracted via fitz
        self.splitter = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        logger.debug(
            f"ProcessingAgent initialized: chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

    def process(
        self,
        state: State,
    ) -> State:
        """Process `state.passages` and populate `state.chunks`."""
        start_time = time.perf_counter()

        if not state.passages:
            msg = "No passages available for processing"
            logger.warning(msg)
            state.append_error(msg)
            return state

        logger.info(f"Processing {len(state.passages)} passages")
        chunking_time = 0.0
        embedding_time = 0.0

        all_chunks: List[Any] = []
        had_nonempty_passage = False

        loop_start = time.perf_counter()

        for idx, passage in enumerate(state.passages):
            # Refactored to use Passage object attributes
            paper_id = passage.paper_id or f"paper_{idx}"
            text = passage.text or ""

            # Metadata extraction
            meta = passage.metadata
            title = meta.get("title", "")
            citation_count = meta.get("citationCount", 0)
            year = meta.get("year")
            venue = meta.get("venue")
            influential = meta.get("influentialCitationCount", 0)

            # Handle PDF URL extraction from metadata
            oa_pdf = meta.get("openAccessPdf")
            pdf_url = None
            if oa_pdf and isinstance(oa_pdf, dict):
                pdf_url = oa_pdf.get("url")

            if not pdf_url:
                pdf_url = meta.get("url")

            if not text.strip():
                logger.warning({"event": "empty_passage_skipped", "paper_id": paper_id})
                continue

            # Text truncation to save memory on large PDFs
            original_len = len(text)
            if original_len > 40000:
                text = text[:40000]
                logger.warning(
                    f"✂️ Truncated paper {paper_id} from {original_len} to 40000 chars to save memory"
                )
            had_nonempty_passage = True

            logger.debug({"event": "processing_paper_start", "paper_id": paper_id})

            # Create document
            doc = Document(text=text)

            try:
                # DIRECT SPLIT: Use SentenceSplitter directly on the document
                nodes = self.splitter.get_nodes_from_documents([doc])
            except Exception as e:
                logger.warning(
                    {
                        "event": "parsing_failed",
                        "paper_id": paper_id,
                        "error": str(e),
                    }
                )
                nodes = []

            paper_chunk_count = 0

            for chunk_idx, node in enumerate(nodes):
                # Attach metadata
                node.metadata.update(
                    {
                        "paper_id": paper_id,
                        "chunk_index": chunk_idx,
                        "paper_title": title,
                        "citation_count": citation_count,
                    }
                )
                if year is not None:
                    node.metadata["year"] = year
                if venue is not None:
                    node.metadata["venue"] = venue
                if influential is not None:
                    node.metadata["influential_citation_count"] = influential
                if pdf_url:
                    node.metadata["pdf_url"] = pdf_url

                all_chunks.append(node)
                paper_chunk_count += 1

            logger.debug(
                {
                    "event": "processing_paper_done",
                    "paper_id": paper_id,
                    "chunks": paper_chunk_count,
                }
            )

        loop_end = time.perf_counter()
        chunking_time = loop_end - loop_start

        if not all_chunks and had_nonempty_passage:
            msg = "Zero chunks generated from passages"
            logger.error(msg)
            state.append_error(msg)

        state.update_fields(chunks=all_chunks)

        elapsed = time.perf_counter() - start_time
        logger.info(
            {
                "event": "processing_complete",
                "passages": len(state.passages),
                "chunks": len(all_chunks),
                "elapsed_s": elapsed,
            }
        )
        return state


__all__ = ["ProcessingAgent"]
