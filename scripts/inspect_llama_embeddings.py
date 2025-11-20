import importlib, traceback

try:
    m = importlib.import_module("llama_index.embeddings")
    print("module_file:", getattr(m, "__file__", "unknown"))
    print("attrs:")
    for a in dir(m):
        print(" ", a)
except Exception:
    traceback.print_exc()
