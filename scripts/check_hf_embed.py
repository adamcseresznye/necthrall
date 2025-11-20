import traceback

try:
    from llama_index.embeddings import HuggingFaceEmbedding

    print("import_ok")
    try:
        m = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        print("instantiated_ok")
    except Exception:
        print("instantiate_error")
        traceback.print_exc()
except Exception:
    print("import_error")
    traceback.print_exc()
