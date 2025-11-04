"""
Small helper to check imports after system reboot.
Run from project root (venv activated):
  python scripts/check_imports.py

It prints import results and full tracebacks for failures.
"""

import sys
import traceback
import os

print("Working directory:", os.getcwd())
print("Python:", sys.version)

# Ensure repo root is on sys.path so sibling packages (e.g. `utils`) can be imported
# When running `python scripts/check_imports.py`, sys.path[0] is the scripts/ dir,
# so add the project root (parent of scripts/) to sys.path.
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
    print(f"Inserted project root into sys.path: {proj_root}")


def try_import(name):
    try:
        m = __import__(name)
        ver = getattr(m, "__version__", None)
        print(f"import {name} OK, __version__={ver}")
        return True
    except Exception as e:
        print(f"import {name} FAILED: {e}")
        traceback.print_exc()
        return False


# Check torch
ok_torch = try_import("torch")

# Check sentence-transformers
ok_st = try_import("sentence_transformers")

# Check transformers (often pulls torch)
ok_transformers = try_import("transformers")

# Check utils.llm_client and try to instantiate LLMClient
try:
    from utils.llm_client import LLMClient

    print("Imported utils.llm_client successfully")
    try:
        client = LLMClient()
        # best-effort info about state
        available = getattr(client, "available", None)
        print("LLMClient() instantiated. available=", available)
    except Exception as e:
        print("LLMClient initialization FAILED:", e)
        traceback.print_exc()
except Exception as e:
    print("Importing utils.llm_client FAILED:", e)
    traceback.print_exc()

print("\nSummary:")
print(" torch:", "OK" if ok_torch else "FAILED")
print(" sentence_transformers:", "OK" if ok_st else "FAILED")
print(" transformers:", "OK" if ok_transformers else "FAILED")
print("Done.")
