from typing import List, Optional
import math
import time
import numpy as np
from loguru import logger


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    # lightweight token approximation by whitespace split
    return len(text.split())


def batched_embed(
    texts: List[Optional[str]],
    embedding_model,
    batch_size: int = 32,
    show_progress: bool = False,
) -> List[np.ndarray]:
    """
    Compute embeddings for `texts` in batches using the provided `embedding_model`.

    Parameters
    - texts: List[Optional[str]] -- list of text chunks to embed (None and empty strings permitted)
    - embedding_model: a LlamaIndex `HuggingFaceEmbedding`-like instance with
      an `embed_documents` method that accepts List[str] and returns List[List[float]]
    - batch_size: int -- number of texts to process per batch (default 32)
    - show_progress: bool -- log batch progress for large inputs (>1000)

    Returns
    - List[np.ndarray] -- list of 1D numpy arrays shape (384,) in same order as input

    Usage example:
    >>> from llama_index import HuggingFaceEmbedding
    >>> model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    >>> embeddings = batched_embed(texts, model, batch_size=32)

    Notes:
    - Empty or None text entries are replaced with zero vectors.
    - If the embedding model does not provide `embed_documents`, the function
      will attempt to call `embed` or `encode` as a fallback.
    """
    if not isinstance(texts, list):
        raise TypeError("`texts` must be a list of strings (or None entries).")

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("`batch_size` must be a positive integer.")

    total = len(texts)
    logger.debug(
        f"batched_embed entry: total={total}, batch_size={batch_size}, show_progress={show_progress}"
    )
    if total == 0:
        return []

    # Pre-create zero vector for empty or None entries to avoid repeated allocation
    zero_vec = np.zeros(384, dtype=float)

    results: List[np.ndarray] = []
    batch_times: List[float] = []

    # Decide how many batches
    num_batches = math.ceil(total / batch_size)

    # If very large and show_progress requested, log overall plan
    if total > 1000:
        logger.info(
            f"Processing {total} texts in {num_batches} batches (batch_size={batch_size})"
        )

    # Helper for progress every 1000
    processed_count = 0
    next_progress_threshold = 1000

    try:
        for b in range(num_batches):
            start_time = time.perf_counter()
            start = b * batch_size
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]

            # Prepare inputs: replace None with empty string but keep marker to insert zeros later
            cleaned_inputs: List[str] = []
            replace_with_zero_indices: List[int] = []
            for i, t in enumerate(batch_texts):
                if t is None or t == "":
                    cleaned_inputs.append("")
                    replace_with_zero_indices.append(i)
                else:
                    # warn and truncate on long texts
                    token_count = _count_tokens(t)
                    if token_count > 512:
                        logger.warning(
                            "Text at global index {} in batch {} exceeds {} tokens — truncating to {} tokens.".format(
                                start + i, b + 1, token_count, 512
                            )
                        )
                        tokens = t.split()
                        t_trunc = " ".join(tokens[:512])
                        cleaned_inputs.append(t_trunc)
                    else:
                        cleaned_inputs.append(t)

            # Call the embedding model. Try common API names.
            if hasattr(embedding_model, "embed_documents"):
                batch_embeddings = embedding_model.embed_documents(cleaned_inputs)
            elif hasattr(embedding_model, "get_text_embedding_batch"):
                # LlamaIndex HuggingFaceEmbedding uses this method
                batch_embeddings = embedding_model.get_text_embedding_batch(
                    cleaned_inputs
                )
            elif hasattr(embedding_model, "embed"):
                batch_embeddings = embedding_model.embed(cleaned_inputs)
            elif hasattr(embedding_model, "encode"):
                batch_embeddings = embedding_model.encode(cleaned_inputs)
            else:
                raise AttributeError(
                    "Provided embedding_model does not implement a known embed method."
                )

            # Convert returned embeddings into numpy arrays and restore zeros for empty inputs
            for i, emb in enumerate(batch_embeddings):
                if i in replace_with_zero_indices:
                    results.append(zero_vec.copy())
                else:
                    arr = np.array(emb, dtype=float)
                    if arr.shape != (384,):
                        # If model returned different dim, try to reshape/raise informative error
                        raise ValueError(
                            f"Embedding dimension mismatch: expected (384,), got {arr.shape}"
                        )
                    results.append(arr)

            batch_time = time.perf_counter() - start_time
            batch_times.append(batch_time)
            processed_count += end - start

            if show_progress:
                logger.info(f"Batch {b+1}/{num_batches} processed in {batch_time:.3f}s")

            if processed_count >= next_progress_threshold:
                logger.info(f"Processed {processed_count}/{total} texts")
                next_progress_threshold += 1000

    except Exception:
        logger.exception("Unexpected error during batched embedding")
        raise

    if show_progress and batch_times:
        total_time = sum(batch_times)
        logger.info(
            f"Batched embedding complete: {total} items in {len(batch_times)} batches, total_time={total_time:.3f}s"
        )

    return results
