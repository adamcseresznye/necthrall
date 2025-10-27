import logging
import json
from typing import List, Dict, Any
from fastapi import Request
from sentence_transformers import SentenceTransformer

from models.state import State, Paper, PDFContent

logger = logging.getLogger(__name__)


class ProcessingAgent:
    """
    Agent responsible for generating embeddings for paper passages and content.

    Uses the cached embedding model from app.state to create vector representations
    of paper content for retrieval and similarity search purposes.
    """

    def __init__(self, request: Request):
        """Initialize ProcessingAgent with cached embedding model from app.state"""
        if (
            hasattr(request.app.state, "embedding_model")
            and request.app.state.embedding_model
        ):
            self.embedding_model = request.app.state.embedding_model
        else:
            raise RuntimeError(
                "Embedding model not loaded in app.state. Check startup event."
            )

    def generate_passage_embeddings(
        self,
        pdf_contents: List[PDFContent],
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> Dict[str, Any]:
        """
        Generate embeddings for passage chunks from PDF content.

        Args:
            pdf_contents: List of PDFContent objects to process
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters

        Returns:
            Dictionary containing embeddings and metadata for each chunk
        """
        if not pdf_contents:
            logger.warning("No PDF content provided for embedding generation")
            return {"embeddings": [], "metadata": []}

        logger.info(
            json.dumps(
                {
                    "event": "embedding_generation_start",
                    "content_count": len(pdf_contents),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
            )
        )

        all_embeddings = []
        all_metadata = []

        for pdf_content in pdf_contents:
            try:
                # Split text into chunks
                chunks = self._chunk_text(
                    pdf_content.raw_text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                if not chunks:
                    continue

                # Generate embeddings for all chunks at once (more efficient)
                chunk_embeddings = self.embedding_model.encode(
                    chunks, convert_to_tensor=False, show_progress_bar=False
                )

                # Store embeddings with metadata
                for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                    all_embeddings.append(embedding)
                    all_metadata.append(
                        {
                            "paper_id": pdf_content.paper_id,
                            "chunk_index": i,
                            "chunk_text": chunk,
                            "page_count": pdf_content.page_count,
                            "char_count": pdf_content.char_count,
                            "extraction_time": pdf_content.extraction_time,
                        }
                    )

            except Exception as e:
                logger.error(
                    json.dumps(
                        {
                            "event": "embedding_generation_error",
                            "paper_id": pdf_content.paper_id,
                            "error": str(e),
                        }
                    )
                )
                continue

        result = {
            "embeddings": all_embeddings,
            "metadata": all_metadata,
            "total_chunks": len(all_embeddings),
        }

        logger.info(
            json.dumps(
                {
                    "event": "embedding_generation_complete",
                    "total_chunks": len(all_embeddings),
                    "papers_processed": len(pdf_contents),
                }
            )
        )

        return result

    def generate_paper_embeddings(self, papers: List[Paper]) -> Dict[str, Any]:
        """
        Generate embeddings for paper titles and abstracts.

        Args:
            papers: List of Paper objects to embed

        Returns:
            Dictionary containing embeddings and paper metadata
        """
        if not papers:
            logger.warning("No papers provided for embedding generation")
            return {"embeddings": [], "metadata": []}

        logger.info(
            json.dumps({"event": "paper_embedding_start", "paper_count": len(papers)})
        )

        texts = []
        metadata = []

        for paper in papers:
            # Combine title and abstract for better representation
            text_parts = []
            if paper.title:
                text_parts.append(paper.title)
            if paper.abstract:
                text_parts.append(paper.abstract)

            if not text_parts:
                continue

            combined_text = " ".join(text_parts)
            texts.append(combined_text)
            metadata.append(
                {
                    "paper_id": paper.id,
                    "title": paper.title,
                    "year": paper.year,
                    "citation_count": paper.citation_count,
                    "source": paper.source,
                }
            )

        if not texts:
            return {"embeddings": [], "metadata": []}

        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts, convert_to_tensor=False, show_progress_bar=False
        )

        result = {
            "embeddings": embeddings,
            "metadata": metadata,
            "total_papers": len(embeddings),
        }

        logger.info(
            json.dumps(
                {"event": "paper_embedding_complete", "total_papers": len(embeddings)}
            )
        )

        return result

    def _chunk_text(
        self, text: str, chunk_size: int = 800, chunk_overlap: int = 120
    ) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to split
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters

        Returns:
            List of text chunks
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to end at a sentence boundary if possible
            if end < len(text):
                # Look for sentence endings (. ! ?)
                sentence_endings = [". ", "! ", "? "]
                for ending in sentence_endings:
                    sentence_pos = text.rfind(ending, start, end)
                    if sentence_pos != -1:
                        end = sentence_pos + len(ending)
                        break

            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

            # Move start position with overlap
            start = end - chunk_overlap
            if start <= 0:  # Prevent infinite loop
                break

        return chunks
