# scripts/setup_onnx.py
import shutil
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.onnxruntime import (
    ORTModelForFeatureExtraction,
    ORTModelForSequenceClassification,
)
from transformers import AutoTokenizer

# Define our two models and their specific types
MODELS_TO_EXPORT = [
    {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "feature-extraction",  # Bi-Encoder (Retriever)
        "class": ORTModelForFeatureExtraction,
    },
    {
        "id": "mixedbread-ai/mxbai-rerank-xsmall-v1",
        "type": "sequence-classification",  # Cross-Encoder (Reranker)
        "class": ORTModelForSequenceClassification,
    },
]

CACHE_ROOT = Path("./onnx_model_cache")


def export_and_quantize(model_info):
    model_id = model_info["id"]
    # Create a clean folder structure: cache/org/model_name
    output_dir = CACHE_ROOT / model_id.replace("/", "_")

    quant_path = output_dir / "model_quantized.onnx"
    float_path = output_dir / "float32"

    if quant_path.exists():
        print(f"✅ {model_id} already exists.")
        return

    print(f"⏳ Exporting {model_id}...")

    # 1. Export to ONNX (Float32)
    # We dynamically select the class (FeatureExtraction vs SequenceClassification)
    model_class = model_info["class"]
    model = model_class.from_pretrained(model_id, export=True)
    model.save_pretrained(float_path)

    # 2. Save Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)

    # 3. Copy Config
    shutil.copy(float_path / "config.json", output_dir / "config.json")

    # 4. Quantize to INT8
    print(f"📉 Quantizing {model_id}...")
    quantize_dynamic(
        model_input=float_path / "model.onnx",
        model_output=quant_path,
        weight_type=QuantType.QUInt8,
    )

    # 5. Cleanup float32 to save space
    shutil.rmtree(float_path)
    print(f"✨ Finished {model_id}")


def main():
    if not CACHE_ROOT.exists():
        CACHE_ROOT.mkdir(parents=True)

    for model_info in MODELS_TO_EXPORT:
        export_and_quantize(model_info)

    print("\n🚀 All ONNX models are ready for deployment!")


if __name__ == "__main__":
    main()
