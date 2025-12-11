# scripts/setup_onnx.py
import os
import shutil
from pathlib import Path
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType


def export_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir = Path("./onnx_model_cache") / "sentence-transformers_all-MiniLM-L6-v2"

    # Define paths
    float_path = cache_dir / "float32"
    quant_path = cache_dir / "model_quantized.onnx"

    if quant_path.exists():
        print(f"✅ Model already exists at {quant_path}")
        return

    print("⏳ Exporting model to ONNX (this takes ~30s)...")

    # 1. Export Standard ONNX
    model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
    model.save_pretrained(float_path)

    # 2. Save Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(cache_dir)

    # 3. Copy Config (Critical for transformers to recognize the folder)
    shutil.copy(float_path / "config.json", cache_dir / "config.json")

    # 4. Quantize
    print("📉 Quantizing to INT8...")
    quantize_dynamic(
        model_input=float_path / "model.onnx",
        model_output=quant_path,
        weight_type=QuantType.QUInt8,
    )

    # 5. Cleanup
    shutil.rmtree(float_path)
    print("✨ Done! Model optimized.")


if __name__ == "__main__":
    export_model()
