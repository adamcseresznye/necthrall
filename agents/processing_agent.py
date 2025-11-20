from __future__ import annotations

from typing import List, Dict, Any, Optional
import time
from loguru import logger

from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document
import numpy as np

from utils.embedding_utils import batched_embed

from models.state import State
from utils.section_detector import detect_sections


class ProcessingAgent:
    """State-aware processing agent that detects sections and chunks texts.

    Usage:
        agent = ProcessingAgent(chunk_size=500, chunk_overlap=50)
        updated_state = agent.process(state)
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.parser = SimpleNodeParser.from_defaults(
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
        """Process `state.passages` and populate `state.chunks`.

        Args:
            state: State containing passages produced by acquisition agent.

        Returns:
            The modified State instance with `chunks` attribute set.
        """
        start_time = time.perf_counter()

        if not state.passages:
            msg = "No passages available for processing"
            logger.warning(msg)
            state.append_error(msg)
            return state

        logger.info(f"Processing {len(state.passages)} passages")
        # Profile timers
        chunking_time = 0.0
        embedding_time = 0.0

        # Debug: starting chunk extraction phase
        logger.debug("Starting chunk extraction for all passages")

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
            text = passage.get("text", "") or ""

            if not text.strip():
                logger.warning({"event": "empty_passage_skipped", "paper_id": paper_id})
                continue
            had_nonempty_passage = True

            logger.info({"event": "processing_paper_start", "paper_id": paper_id})

            try:
                sections = detect_sections(text, paper_id=paper_id)
            except Exception as e:
                logger.warning(
                    {
                        "event": "section_detection_failed",
                        "paper_id": paper_id,
                        "error": str(e),
                    }
                )
                sections = [{"name": "full_text", "text": text}]

            paper_chunk_count = 0
            for sec in sections:
                sec_name = sec.get("name", "full_text")
                sec_text = sec.get("text", "") or ""
                if not sec_text.strip():
                    continue

                doc = Document(text=sec_text)
                try:
                    # Debug: about to run the SimpleNodeParser to chunk section text
                    logger.debug(
                        {
                            "event": "chunking_section_start",
                            "paper_id": paper_id,
                            "section": sec_name,
                        }
                    )
                    nodes = self.parser.get_nodes_from_documents([doc])
                except Exception as e:
                    logger.warning(
                        {
                            "event": "chunking_failed",
                            "paper_id": paper_id,
                            "section": sec_name,
                            "error": str(e),
                        }
                    )
                    nodes = []

                for chunk_idx, node in enumerate(nodes):
                    # Attach metadata directly; fail-fast if this raises
                    node.metadata.update(
                        {
                            "paper_id": paper_id,
                            "section_name": sec_name,
                            "chunk_index": chunk_idx,
                            "paper_title": title,
                            "citation_count": citation_count,
                        }
                    )
                    # Add optional fields if present
                    if year is not None:
                        node.metadata["year"] = year
                    if venue is not None:
                        node.metadata["venue"] = venue
                    if influential is not None:
                        node.metadata["influential_citation_count"] = influential

                    all_chunks.append(node)
                    paper_chunk_count += 1

            logger.debug(
                {
                    "event": "processing_paper_done",
                    "paper_id": paper_id,
                    "chunks": paper_chunk_count,
                    "sections": len(sections),
                }
            )

        loop_end = time.perf_counter()
        chunking_time = loop_end - loop_start

        if not all_chunks:
            # If there were non-empty passages but no chunks, record a critical error.
            if had_nonempty_passage:
                msg = "Zero chunks generated from passages"
                logger.error(msg)
                state.append_error(msg)

        # If an embedding model was provided, compute embeddings in a single batched pass
        if embedding_model is not None and all_chunks:
            # Prepare texts (single-pass extraction) — avoid duplicating large text blobs
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

            # Retry logic with exponential backoff
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
                    # Successful call — break retry loop
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
                        logger.info(
                            {
                                "event": "embedding_retry_wait",
                                "wait_s": wait,
                                "attempt": attempt + 1,
                            }
                        )
                        time.sleep(wait)
                        continue
                    else:
                        logger.exception(
                            "Batched embedding failed after %s attempts", max_attempts
                        )
                        state.append_error(
                            f"Batched embedding failed after {max_attempts} attempts; see logs"
                        )

            # If embeddings were obtained, attach them to nodes
            if embeddings is not None:
                missing_embeddings: List[int] = []
                for idx, (node, emb) in enumerate(zip(all_chunks, embeddings)):
                    try:
                        arr = np.asarray(emb, dtype=float)
                        if arr.shape != (384,):
                            raise ValueError(
                                f"Embedding dimension mismatch for chunk {idx}: {arr.shape}"
                            )
                        # Attach as serializable list to metadata (keeps node object small)
                        node.metadata["embedding"] = arr.tolist()
                    except Exception as e:
                        logger.exception(
                            "Failed to attach embedding for chunk %s: %s", idx, e
                        )
                        missing_embeddings.append(idx)

                if missing_embeddings:
                    logger.warning(
                        {
                            "event": "missing_embeddings",
                            "count": len(missing_embeddings),
                        }
                    )
                    state.append_error(
                        f"{len(missing_embeddings)} chunks missing embeddings"
                    )

            # Info: embedding phase summary for monitoring
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
