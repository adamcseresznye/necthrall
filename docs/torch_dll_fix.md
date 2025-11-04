# Torch DLL Import-Order Fix

## Problem
When importing `utils.llm_client`, Windows was throwing a DLL initialization error:
```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed. 
Error loading "...\torch\lib\c10.dll" or one of its dependencies.
```

## Root Cause
This is a known issue with PyTorch on Windows when it's imported indirectly through other libraries (langchain, transformers). The import order matters - if torch is loaded after certain other packages initialize, it can fail to load its DLLs properly.

## Solution
Add an explicit `import torch` statement **before** importing langchain providers in `utils/llm_client.py`.

### Changes Made

#### 1. `utils/llm_client.py`
Added explicit torch import before langchain imports:

```python
# CRITICAL: Import torch BEFORE langchain providers to avoid DLL initialization errors on Windows
# See: https://github.com/pytorch/pytorch/issues/91966
import torch

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
```

#### 2. `scripts/test_necthrall_real.py`
Updated the LLM initialization logic in `initialize_system()` to properly instantiate the client:

```python
# Import and initialize LLM client (mimicking main.py startup logic)
import main
if main.llm_client is None:
    from utils.llm_client import LLMClient as _LLMClient
    main.llm_client = _LLMClient()
    main.LLMClient = _LLMClient
status["llm"] = main.llm_client is not None
```

## Verification
Created `scripts/verify_dll_fix.py` to confirm the fix works:
- ✅ `utils.llm_client` imports successfully
- ✅ `LLMClient` instantiates without errors
- ✅ torch 2.9.0+cpu loads correctly
- ✅ LLM generates responses (tested with Gemini)

## Result
The full Necthrall pipeline can now run with real LLMs (no mocking required):
```powershell
# Activate venv and run interactive console
venv\Scripts\Activate
python -m scripts.test_necthrall_real
```

## References
- PyTorch Windows DLL issue: https://github.com/pytorch/pytorch/issues/91966
- Import order is critical when mixing torch with other ML libraries on Windows
- The fix is non-invasive and doesn't change any API contracts
