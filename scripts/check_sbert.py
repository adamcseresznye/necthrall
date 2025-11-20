import traceback

try:
    from sentence_transformers import SentenceTransformer

    print("sbert_import_ok")
    try:
        m = SentenceTransformer("all-MiniLM-L6-v2")
        print("sbert_instantiated_ok")
    except Exception:
        print("sbert_instantiate_error")
        traceback.print_exc()
except Exception:
    print("sbert_import_error")
    traceback.print_exc()
