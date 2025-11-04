"""
Diagnostic script to check environment configuration for Necthrall.
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("NECTHRALL ENVIRONMENT CONFIGURATION CHECK")
print("=" * 60)

# Required for OpenAlex API
openalex_email = os.getenv("OPENALEX_EMAIL")
unpaywall_email = os.getenv("UNPAYWALL_EMAIL")

# Required for LLM providers
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
llm_primary = os.getenv("LLM_MODEL_PRIMARY")
llm_fallback = os.getenv("LLM_MODEL_FALLBACK")

# Optional
openai_key = os.getenv("OPENAI_API_KEY")

print("\n📧 OpenAlex API Configuration:")
print(f"  OPENALEX_EMAIL: {'✅ Set' if openalex_email else '❌ MISSING (REQUIRED)'}")
if openalex_email:
    print(f"    Value: {openalex_email}")
print(f"  UNPAYWALL_EMAIL: {'✅ Set' if unpaywall_email else '❌ MISSING (REQUIRED)'}")
if unpaywall_email:
    print(f"    Value: {unpaywall_email}")

print("\n🤖 LLM Configuration:")
print(f"  GOOGLE_API_KEY: {'✅ Set' if google_api_key else '❌ Missing'}")
if google_api_key:
    print(f"    Value: {google_api_key[:10]}...{google_api_key[-4:]}")
print(f"  LLM_MODEL_PRIMARY: {'✅ Set' if llm_primary else '❌ Missing'}")
if llm_primary:
    print(f"    Value: {llm_primary}")
print(f"  GROQ_API_KEY: {'✅ Set' if groq_api_key else '❌ Missing'}")
if groq_api_key:
    print(f"    Value: {groq_api_key[:10]}...{groq_api_key[-4:]}")
print(f"  LLM_MODEL_FALLBACK: {'✅ Set' if llm_fallback else '❌ Missing'}")
if llm_fallback:
    print(f"    Value: {llm_fallback}")

print("\n🔧 Optional:")
print(f"  OPENAI_API_KEY: {'✅ Set' if openai_key else '⚠️  Not set (optional)'}")

print("\n" + "=" * 60)

# Check critical issues
critical_missing = []
if not openalex_email:
    critical_missing.append("OPENALEX_EMAIL")
if not unpaywall_email:
    critical_missing.append("UNPAYWALL_EMAIL")

if critical_missing:
    print("❌ CRITICAL: Missing required environment variables!")
    print(f"   {', '.join(critical_missing)}")
    print("\n📝 To fix:")
    print("   1. Edit your .env file in the project root")
    print("   2. Add the following lines:")
    for var in critical_missing:
        print(f"      {var}=your-email@example.com")
    print("   3. Restart your script")
    print("\n💡 Note: OpenAlex requires a valid email for API access.")
    print(
        "   See: https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication"
    )
else:
    print("✅ All critical environment variables are set!")

print("=" * 60)
