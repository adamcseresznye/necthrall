"""Ranking agent for scoring and ranking Semantic Scholar papers.

Implements a multi-stage hybrid ranking model combining:
1.  **Relevance (RRF)**:
    - BM25 (Lexical)
    - TF-IDF (Lexical)
    - LSA (Latent Semantic)
    - BM25-SPECTER Centroid (Semantic)
2.  **Authority**: Log-normalized influential/total citations.
3.  **Recency**: Exponential decay.

This agent accepts a List[Dict] of papers and a query string,
converts them to a DataFrame for internal processing, and returns
a List[Dict] of ranked finalists.
"""

from datetime import date
from typing import Any, Dict, List

import bm25s
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models.state import Paper

# --- Configuration ---
RRF_K = 60  # Constant for Reciprocal Rank Fusion (RRF)
LSA_COMPONENTS = 100  # Number of components for LSA
CENTROID_K = 5  # Number of top BM25 papers to use for centroid
CURRENT_YEAR = date.today().year
RECENCY_LAMBDA = 0.1  # Decay rate for recency (0.1 = ~10 year half-life)

# Final ranking weights
W_RELEVANCE = 0.60
W_AUTHORITY = 0.35
W_RECENCY = 0.05


# --- Internal Helper Functions (Operating on Pandas Data) ---


def _compute_bm25s_ranks(query: str, corpus: List[str]) -> np.ndarray:
    """Computes BM25 ranks (1 = best) for a query against a corpus."""
    if not corpus:
        return np.array([])

    try:
        corpus_tokens = bm25s.tokenize(corpus, stopwords="english")
        # Guard against documents that tokenize to empty lists. If every
        # document is empty (no tokens), bm25s may compute an average
        # document length of zero which leads to a divide-by-zero and
        # RuntimeWarning inside the bm25s scoring implementation. Rather
        # than letting the third-party library emit warnings, return a
        # neutral rank vector so downstream scoring remains stable.
        doc_lengths = [len(toks) for toks in corpus_tokens]
        if sum(doc_lengths) == 0:
            logger.debug(
                "BM25: all documents tokenized to empty; returning neutral ranks"
            )
            return np.full(len(corpus), len(corpus), dtype=int)
        retriever = bm25s.BM25(method="lucene")
        query_tokens = bm25s.tokenize(query, stopwords="english")

        # The bm25s scoring internals may emit a RuntimeWarning (numpy
        # 'invalid value encountered in scalar divide') in some edge cases
        # (e.g. unexpected zero document lengths). Wrap both indexing and
        # retrieval in a numpy.errstate context to suppress those spurious
        # runtime warnings while allowing normal numeric behavior to proceed.
        with np.errstate(invalid="ignore"):
            retriever.index(corpus_tokens)
            _, scores = retriever.retrieve(query_tokens, k=len(corpus))
        scores = scores.flatten()

        if scores.max() > 0:
            scores_normalized = scores / scores.max()
        else:
            scores_normalized = np.zeros(len(corpus))

        return rankdata(-scores_normalized, "dense")
    except Exception as e:
        logger.error(f"Error in BM25S ranking: {e}. Returning empty ranks.")
        return np.full(len(corpus), len(corpus), dtype=int)


def _compute_tfidf_ranks(query: str, corpus: List[str]) -> np.ndarray:
    """Computes TF-IDF cosine similarity ranks (1 = best)."""
    if not corpus:
        return np.array([])

    try:
        full_corpus = [query] + corpus
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(full_corpus)

        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        scores_normalized = similarities / (similarities.max() or 1)
        return rankdata(-scores_normalized, "dense")
    except Exception as e:
        logger.error(f"Error in TF-IDF ranking: {e}. Returning empty ranks.")
        return np.full(len(corpus), len(corpus), dtype=int)


def _compute_lsa_ranks(query: str, corpus: List[str], n_components: int) -> np.ndarray:
    """Computes LSA ranks (1 = best). Inefficiently re-builds index per query."""
    if not corpus:
        return np.array([])

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_df=0.8, min_df=5)
        tfidf_matrix = vectorizer.fit_transform(corpus)

        effective_components = min(n_components, tfidf_matrix.shape[1] - 1)
        if effective_components <= 0:
            logger.warning("LSA: Not enough features for SVD. Skipping.")
            return np.full(len(corpus), len(corpus), dtype=int)

        svd_model = TruncatedSVD(n_components=effective_components, random_state=42)
        lsa_matrix = svd_model.fit_transform(tfidf_matrix)

        query_tfidf = vectorizer.transform([query])
        query_lsa = svd_model.transform(query_tfidf)

        scores = cosine_similarity(query_lsa, lsa_matrix).flatten()
        return rankdata(-scores, "dense")
    except ValueError as e:
        logger.warning(f"Error in LSA ranking: {e}. Returning empty ranks.")
        return np.full(len(corpus), len(corpus), dtype=int)


def _compute_bm25_centroid_ranks(df: pd.DataFrame, centroid_k: int) -> np.ndarray:
    """Ranks papers by similarity to the centroid of the top 'k' BM25 papers."""
    df["specter_vector"] = df["embedding"].apply(
        lambda x: x.get("specter") if isinstance(x, dict) else None
    )
    df_clean = df.dropna(subset=["specter_vector"]).copy()

    if df_clean.empty:
        logger.warning("No SPECTER embeddings found for centroid ranking.")
        return np.full(len(df), len(df), dtype=int)

    try:
        embeddings_clean_all = np.stack(df_clean["specter_vector"].values)
    except ValueError as e:
        logger.error(f"Error stacking embeddings (mismatched dimensions?): {e}")
        return np.full(len(df), len(df), dtype=int)

    top_k_papers = df_clean.sort_values(by="bm25_rank", ascending=True).head(centroid_k)

    if top_k_papers.empty:
        logger.warning("No top BM25 papers found to create centroid.")
        return np.full(len(df), len(df), dtype=int)

    top_k_ilocs = df_clean.index.get_indexer(top_k_papers.index)
    top_k_vectors = embeddings_clean_all[top_k_ilocs]
    centroid_query_vector = np.mean(top_k_vectors, axis=0).reshape(1, -1)

    scores = cosine_similarity(centroid_query_vector, embeddings_clean_all).flatten()

    ranked_scores_series = pd.Series(scores, index=df_clean.index)
    full_scores_series = ranked_scores_series.reindex(df.index).fillna(-np.inf)

    return rankdata(-full_scores_series.values, "dense")


def _compute_rrf_score(df: pd.DataFrame) -> pd.Series:
    """Computes the Reciprocal Rank Fusion (RRF) score for relevance."""
    return (
        1 / (df["bm25_rank"] + RRF_K)
        + 1 / (df["tfid_rank"] + RRF_K)
        + 1 / (df["lsa_rank"] + RRF_K)
        + 1 / (df["pseudo_specter_rank"] + RRF_K)
    )


def _compute_authority_ranks(df: pd.DataFrame) -> np.ndarray:
    """Computes authority ranks (1 = best) based on citations."""

    def get_authority(row):
        icc = row.get("influentialCitationCount", 0) or 0
        cc = row.get("citationCount", 0) or 0
        return np.log10(max(icc, cc * 0.3) + 1)

    authority_raw = df.apply(get_authority, axis=1)
    return rankdata(-authority_raw.values, method="dense")


def _compute_recency_ranks(
    df: pd.DataFrame, current_year: int, lambda_decay: float
) -> np.ndarray:
    """Computes recency ranks (1 = best) using exponential decay."""

    def get_recency(year):
        if pd.isna(year):
            return 0
        return np.exp(-lambda_decay * (current_year - year))

    recency_raw = df["year"].apply(get_recency)
    return rankdata(-recency_raw.values, method="dense")


def _normalize_ranks(df: pd.DataFrame, rank_columns: List[str]) -> pd.DataFrame:
    """Applies min-max normalization to rank columns (where 1 is best)."""
    for col in rank_columns:
        min_val = df[col].min()
        max_val = df[col].max()

        if max_val - min_val > 0:
            df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[f"{col}_norm"] = 0.5
    return df


def _compute_final_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Computes the final weighted score from all signals."""
    df["rrf_relevance_score"] = _compute_rrf_score(df)
    df["rrf_relevance_rank"] = rankdata(
        -df["rrf_relevance_score"].values, method="dense"
    )
    df["authority_rank"] = _compute_authority_ranks(df)
    df["recency_rank"] = _compute_recency_ranks(df, CURRENT_YEAR, RECENCY_LAMBDA)

    rank_cols = ["rrf_relevance_rank", "authority_rank", "recency_rank"]
    df = _normalize_ranks(df, rank_cols)

    df["relevance_score"] = 1.0 - df["rrf_relevance_rank_norm"]
    df["authority_score"] = 1.0 - df["authority_rank_norm"]
    df["recency_score"] = 1.0 - df["recency_rank_norm"]

    df["final_score"] = (
        W_RELEVANCE * df["relevance_score"]
        + W_AUTHORITY * df["authority_score"]
        + W_RECENCY * df["recency_score"]
    )

    return df.sort_values("final_score", ascending=False)


class RankingAgent:
    """
    Agent that ranks Semantic Scholar papers using a hybrid, multi-stage
    scoring model based on Relevance (RRF), Authority, and Recency.
    """

    def rank_papers(
        self, papers: List[Paper], query: str, top_k: int = 50
    ) -> List[Paper]:
        """Rank papers using the hybrid ranking model.

        Args:
            papers: List[Paper] of Semantic Scholar papers
            query: The optimized query string for ranking
            top_k: Number of top papers to return (default: 50 for Base+Bonus strategy)

        Returns:
            List[Paper]: Top k papers with all computed rank/score fields added,
                        returned as a List of Paper objects.
        """
        # --- 1. Validate inputs ---
        if not isinstance(papers, list):
            raise ValueError("papers must be a list")
        if not isinstance(query, str) or not query:
            raise ValueError("query must be a non-empty string")

        if not papers:
            logger.warning("No papers provided for ranking")
            return []

        logger.info(
            f"RankingAgent starting: {len(papers)} papers for query '{query[:50]}...'"
        )

        # --- 2. Convert List[Paper] to DataFrame for internal processing ---
        try:
            # Convert Pydantic models to dicts for DataFrame processing
            papers_data = [p.model_dump() for p in papers]
            df = pd.DataFrame(papers_data)

        except Exception as e:
            logger.error(f"Failed to create DataFrame from papers: {e}")
            return []

        # 3. Clean text fields for lexical search
        df["title"] = df["title"].fillna("")
        df["abstract"] = df["abstract"].apply(lambda x: str(x) if pd.notna(x) else "")
        corpus = (df["title"] + ": " + df["abstract"]).tolist()
        titles_only = df["title"].tolist()

        # --- 4. Run all ranking steps from notebook ---
        logger.debug("Computing relevance ranks (BM25, TF-IDF, LSA)...")
        df["bm25_rank"] = _compute_bm25s_ranks(query, corpus)
        df["tfid_rank"] = _compute_tfidf_ranks(query, titles_only)

        logger.debug("Computing LSA ranks (note: inefficient, re-builds index)...")
        df["lsa_rank"] = _compute_lsa_ranks(query, corpus, n_components=LSA_COMPONENTS)

        logger.debug("Computing BM25-SPECTER centroid ranks...")
        df["pseudo_specter_rank"] = _compute_bm25_centroid_ranks(
            df, centroid_k=CENTROID_K
        )

        logger.debug("Computing final RRF, authority, recency, and weighted score...")
        ranked_df = _compute_final_ranking(df)

        # --- 5. Convert top k back to List[Paper] ---
        finalists_df = ranked_df.head(top_k)
        # Convert DataFrame to a list of dictionaries, replacing NaNs with None
        finalists_dicts = finalists_df.replace({np.nan: None}).to_dict("records")
        # Re-validate into Paper objects
        finalists = [Paper(**d) for d in finalists_dicts]

        # Log top 3
        top_titles = [p.get("title", "Unknown") for p in finalists_dicts[:3]]
        logger.info(
            "Top 3 ranked papers: 1. '{}' (Score: {:.3f}), 2. '{}' (Score: {:.3f}), 3. '{}' (Score: {:.3f})",
            top_titles[0] if len(finalists_dicts) > 0 else "None",
            finalists_dicts[0]["final_score"] if len(finalists_dicts) > 0 else 0.0,
            top_titles[1] if len(finalists_dicts) > 1 else "None",
            finalists_dicts[1]["final_score"] if len(finalists_dicts) > 1 else 0.0,
            top_titles[2] if len(finalists_dicts) > 2 else "None",
            finalists_dicts[2]["final_score"] if len(finalists_dicts) > 2 else 0.0,
        )

        logger.info(
            "Ranking completed: {} papers processed, {} finalists selected",
            len(papers),
            len(finalists),
        )

        return finalists


__all__ = ["RankingAgent"]
