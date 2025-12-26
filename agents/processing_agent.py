from __future__ import annotations

import time
from typing import Any, List, Optional

import numpy as np
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from loguru import logger

from models.state import State
from utils.embedding_utils import batched_embed


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
        embedding_model: Optional[object] = None,
        batch_size: int = 32,
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
            paper_id = (
                passage.get("paperId") or passage.get("paper_id") or f"paper_{idx}"
            )
            title = passage.get("title", "")
            citation_count = passage.get(
                "citationCount", passage.get("citation_count", 0)
            )
            year = passage.get("year") or passage.get("publication_year")
            venue = passage.get("venue") or passage.get("journal")
            influential = passage.get(
                "influentialCitationCount", passage.get("influential_citation_count", 0)
            )
            oa_pdf = passage.get("openAccessPdf")
            pdf_url = oa_pdf.get("url") if oa_pdf and isinstance(oa_pdf, dict) else None
            text = passage.get("text", "") or ""

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

            logger.info({"event": "processing_paper_start", "paper_id": paper_id})

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

        # Embedding Logic (unchanged)
        if embedding_model is not None and all_chunks:
            texts: List[str] = []
            for node in all_chunks:
                if hasattr(node, "get_text"):
                    try:
                        texts.append(node.get_text())
                    except Exception:
                        texts.append("")
                elif hasattr(node, "text"):
                    texts.append(getattr(node, "text") or "")
                else:
                    texts.append("")

            # Retry logic
            max_attempts = 3
            base_backoff = 0.5
            embeddings = None
            for attempt in range(1, max_attempts + 1):
                try:
                    emb_start = time.perf_counter()
                    embeddings = batched_embed(
                        texts,
                        embedding_model=embedding_model,
                        batch_size=int(batch_size),
                    )
                    emb_end = time.perf_counter()
                    embedding_time = emb_end - emb_start
                    break
                except Exception as exc:
                    logger.exception(
                        {
                            "event": "embedding_attempt_failed",
                            "attempt": attempt,
                            "error": str(exc),
                        }
                    )
                    if attempt < max_attempts:
                        wait = base_backoff * (2 ** (attempt - 1))
                        time.sleep(wait)
                        continue
                    else:
                        state.append_error("Batched embedding failed")

            if embeddings is not None:
                for idx, (node, emb) in enumerate(zip(all_chunks, embeddings)):
                    try:
                        arr = np.asarray(emb, dtype=float)
                        if arr.shape != (384,):
                            continue
                        node.metadata["embedding"] = arr.tolist()
                    except Exception as e:
                        logger.exception("Failed to attach embedding", idx=idx)

            logger.info(
                {
                    "event": "embedding_complete",
                    "chunks_embedded": len(all_chunks) if embeddings is not None else 0,
                    "embedding_time_s": embedding_time,
                    "chunking_time_s": chunking_time,
                }
            )

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
