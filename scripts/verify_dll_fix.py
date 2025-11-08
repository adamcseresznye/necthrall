"""
Verification script to confirm the torch DLL import-order fix works.
Run: python scripts/verify_dll_fix.py
"""

import sys
import os

# Ensure repo root is on sys.path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

print("=" * 60)
print("TORCH DLL FIX VERIFICATION")
print("=" * 60)

# Test 1: Import utils.llm_client (this now imports torch first)
print("\n1. Testing utils.llm_client import...")
try:
    from utils.llm_client import LLMClient

    print("   ✅ utils.llm_client imported successfully")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Test 2: Instantiate LLMClient
print("\n2. Testing LLMClient instantiation...")
try:
    client = LLMClient()
    print("   ✅ LLMClient instantiated successfully")
    # Use safe accessor for model identifiers (some wrappers don't expose model_name)
    try:
        from utils.llm_client import get_safe_model_name

        primary_id = get_safe_model_name(client.primary_llm)
        fallback_id = get_safe_model_name(client.fallback_llm)
    except Exception:
        # Fallback to getattr for safety
        primary_id = getattr(
            client.primary_llm,
            "model",
            getattr(client.primary_llm, "model_name", "unknown"),
        )
        fallback_id = getattr(
            client.fallback_llm,
            "model",
            getattr(client.fallback_llm, "model_name", "unknown"),
        )

    print(f"   Primary model: {primary_id}")
    print(f"   Fallback model: {fallback_id}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 3: Verify torch is loaded
print("\n3. Verifying torch is available...")
try:
    import torch

    print(f"   ✅ torch {torch.__version__} loaded")
    print(f"   CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Test 4: Test basic LLM generation (optional - requires API keys)
print("\n4. Testing LLM generation (quick test)...")
try:
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant. Be brief."},
        {"role": "user", "content": "Say 'Hello' in one word."},
    ]
    response = client.generate(test_messages, max_tokens=10)
    print(f"   ✅ LLM responded: {response['content'][:50]}")
    print(f"   Model used: {response['model_used']}")
except Exception as e:
    print(f"   ⚠️  LLM test skipped or failed: {e}")
    # Don't exit - API keys might not be configured

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE ✅")
print("=" * 60)
print("\nThe torch DLL import-order issue is fixed!")
print("You can now run the full interactive testing console:")
print("  python -m scripts.test_necthrall_real")
