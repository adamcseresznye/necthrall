"""ONNX-optimized embedding model (Production Ready).

This module uses only onnxruntime and the tokenizers library - NO torch dependency.
This avoids Windows DLL conflicts between torch and onnxruntime.

Threading optimization is handled by config._threading (imported at package level).
"""

from pathlib import Path
from typing import List

import numpy as np
from loguru import logger

# Use tokenizers library directly - it's lightweight and doesn't import torch
from tokenizers import Tokenizer

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        "onnxruntime is not installed. Install with: pip install onnxruntime"
    ) from e


class ONNXEmbeddingModel:
    """ONNX-based embedding model for high-performance CPU inference.

    This class uses the tokenizers library directly instead of transformers
    to avoid loading torch DLLs, which cause conflicts with onnxruntime on Windows.

    Usage:
        model = ONNXEmbeddingModel()
        embeddings = model.get_text_embedding_batch(["Hello world", "Test text"])

    Attributes:
        embed_dim: The embedding dimension (384 for all-MiniLM-L6-v2).
    """

    embed_dim: int = 384  # all-MiniLM-L6-v2 produces 384-dimensional embeddings

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_dir = Path("./onnx_model_cache") / model_name.replace("/", "_")
        self.model_path = self.model_dir / "model_quantized.onnx"
        self.tokenizer_path = self.model_dir / "tokenizer.json"

        if not self.model_path.exists():
            raise RuntimeError(
                f"Model missing: {self.model_path}. Run 'python scripts/setup_onnx.py' to download."
            )

        if not self.tokenizer_path.exists():
            raise RuntimeError(
                f"Tokenizer missing: {self.tokenizer_path}. Run 'python scripts/setup_onnx.py' to download."
            )

        logger.info(f"Loading ONNX model from {self.model_dir}")

        # Load tokenizer directly from tokenizer.json (no torch dependency)
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

        # Enable padding and truncation using the tokenizers library API
        # Note: The tokenizers library uses string arguments, not enum classes
        self.tokenizer.enable_padding(direction="right", pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=512)

        # Session Options - Use all available cores
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 0  # 0 = use all available cores
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        logger.info(
            "ONNX session configured with automatic thread detection (0 = all cores)"
        )

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {x.name for x in self.session.get_inputs()}
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"ONNX model loaded successfully. Input names: {self.input_names}")

    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embeddings, each a list of floats with length 384.
        """
        if not texts:
            return []

        # Tokenize using the tokenizers library
        encodings = self.tokenizer.encode_batch(texts)

        # Extract input_ids and attention_mask
        input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
        attention_mask = np.array(
            [enc.attention_mask for enc in encodings], dtype=np.int64
        )

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Add token_type_ids if the model expects them
        if "token_type_ids" in self.input_names:
            ort_inputs["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        # Run inference
        embeddings = self.session.run([self.output_name], ort_inputs)[0]

        # Mean pooling with attention mask
        mask = attention_mask[:, :, None]
        sum_mask = mask.sum(axis=1)
        sum_mask[sum_mask == 0] = 1e-9

        return ((embeddings * mask).sum(axis=1) / sum_mask).tolist()


def initialize_embedding_model() -> ONNXEmbeddingModel:
    """Factory function to create an ONNXEmbeddingModel instance.

    Returns:
        Initialized ONNXEmbeddingModel ready for inference.

    Raises:
        RuntimeError: If model files are missing.
        ImportError: If onnxruntime is not installed.
    """
    return ONNXEmbeddingModel()
