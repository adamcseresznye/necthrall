"""ONNX-optimized embedding model (Production Ready).

Threading optimization is handled by config._threading (imported at package level).
This ensures multi-core mode is enabled before any numpy/torch imports occur.
"""

import numpy as np
from pathlib import Path
from typing import List
from loguru import logger
from transformers import AutoTokenizer

# WINDOWS DLL FIX: Import torch before onnxruntime to avoid DLL initialization errors
try:
    import torch
except ImportError:
    pass

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("Failed to import onnxruntime.") from e


class ONNXEmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_dir = Path("./onnx_model_cache") / model_name.replace("/", "_")
        self.model_path = self.model_dir / "model_quantized.onnx"

        if not self.model_path.exists():
            # Auto-recovery instructions
            raise RuntimeError(f"Model missing: {self.model_path}. Run setup_onnx.py.")

        logger.info(f"Loading ONNX model from {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

        # Session Options - Use all available cores
        sess_options = ort.SessionOptions()

        # CRITICAL: Set to 0 to let ONNX Runtime use all available CPU cores
        # This respects OMP_NUM_THREADS set before Python starts
        sess_options.intra_op_num_threads = 0  # 0 = use all available cores
        sess_options.inter_op_num_threads = 0  # 0 = use all available cores

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

    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="np"
        )

        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }

        if "token_type_ids" in self.input_names:
            if "token_type_ids" in inputs:
                ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
            else:
                ort_inputs["token_type_ids"] = np.zeros_like(
                    inputs["input_ids"], dtype=np.int64
                )

        embeddings = self.session.run([self.output_name], ort_inputs)[0]

        mask = inputs["attention_mask"][:, :, None]
        sum_mask = mask.sum(axis=1)
        sum_mask[sum_mask == 0] = 1e-9

        return ((embeddings * mask).sum(axis=1) / sum_mask).tolist()


def initialize_embedding_model() -> ONNXEmbeddingModel:
    return ONNXEmbeddingModel()
