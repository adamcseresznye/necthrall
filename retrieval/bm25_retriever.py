"""BM25-only retriever for fast, embedding-free passage retrieval.

Uses bm25s with whitespace tokenisation – the same strategy used by
LlamaIndexRetriever._bm25_search.

The two dataclasses (SimpleNode / SimpleNodeWithScore) mirror the
llama_index NodeWithScore interface so that all downstream code in
RAGService (_select_diverse_top_k, Passage conversion) and SynthesisAgent
works without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import bm25s
from loguru import logger

# ---------------------------------------------------------------------------
# Lightweight node wrappers (NodeWithScore-compatible)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimpleNode:
    """Minimal stand-in for llama_index TextNode.

    Satisfies:
    - .text           (SynthesisAgent fallback)
    - .get_content()  (SynthesisAgent primary + RAGService conversion)
    - .metadata       (RAGService / _select_diverse_top_k)
    """

    text: str
    metadata: Dict[str, Any]

    def get_content(self) -> str:
        return self.text


@dataclass(frozen=True)
class SimpleNodeWithScore:
    """Minimal stand-in for llama_index NodeWithScore.

    Satisfies:
    - .node   → SimpleNode
    - .score  → float
    """

    node: SimpleNode
    score: float


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------


class BM25Retriever:
    """Retrieve top-k passages using BM25Okapi; no embedding model required.

    Args:
        top_k: Maximum number of results to return.
    """

    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, chunks: List[Any]) -> List[SimpleNodeWithScore]:
        """Score *chunks* against *query* with BM25 and return top-k results.

        Args:
            query: The search query string.
            chunks: Iterable of objects that expose a ``get_content()``
                    method and a ``metadata`` dict (i.e. llama_index
                    TextNode objects as produced by ProcessingAgent).

        Returns:
            List of :class:`SimpleNodeWithScore` sorted descending by score,
            length ≤ ``self.top_k``.
        """
        if not chunks:
            logger.warning("BM25Retriever.retrieve called with empty chunk list")
            return []

        # ---- 1. Extract text + metadata --------------------------------
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            # LlamaIndex TextNode primary interface
            if hasattr(chunk, "get_content"):
                text = chunk.get_content()
            elif hasattr(chunk, "text"):
                text = chunk.text or ""
            elif isinstance(chunk, dict):
                text = chunk.get("text", "")
            else:
                text = str(chunk)

            if hasattr(chunk, "metadata"):
                meta = chunk.metadata if isinstance(chunk.metadata, dict) else {}
            elif isinstance(chunk, dict):
                meta = chunk.get("metadata", {})
            else:
                meta = {}

            texts.append(text)
            metadatas.append(meta)

        # ---- 2. Tokenise -----------------------------------------------
        tokenized_corpus = [t.lower().split() for t in texts]
        tokenized_query = [query.lower().split()]  # bm25s expects list of queries

        # ---- 3. BM25 scoring -------------------------------------------
        retriever = bm25s.BM25()
        retriever.index(tokenized_corpus)
        k = min(self.top_k, len(texts))
        result_indices, result_scores = retriever.retrieve(tokenized_query, k=k)

        # ---- 4. Rank and return top-k ----------------------------------
        # result_indices/scores shape: (1, k) — unwrap the single query dimension
        top = list(zip(result_indices[0].tolist(), result_scores[0].tolist()))

        # If every score is zero (no term overlap) keep top-k by document
        # order so callers always receive *something* usable.
        all_zero = all(s == 0.0 for _, s in top)
        if all_zero:
            logger.debug(
                "BM25Retriever: no term overlap for query '{}'; "
                "returning top-k by document order",
                query,
            )
            top = [(i, 0.0) for i in range(k)]

        results: List[SimpleNodeWithScore] = []
        for idx, score in top:
            node = SimpleNode(text=texts[idx], metadata=metadatas[idx])
            results.append(SimpleNodeWithScore(node=node, score=float(score)))

        logger.debug(
            "BM25Retriever: retrieved {} / {} chunks (top_k={})",
            len(results),
            len(chunks),
            self.top_k,
        )
        return results


__all__ = ["BM25Retriever", "SimpleNode", "SimpleNodeWithScore"]
