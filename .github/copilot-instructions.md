## Necthrall — Copilot / AI assist guidance

Keep suggestions tight and code-aware. Focus on the RAG pipeline, PDF handling, chunking, embeddings, and tests.

- Big picture: this is a single Python app that implements an on-demand RAG loop: fetch OA papers → parse/extract text → section-aware chunking → embed with a SentenceTransformer-like model (expected embedding dim 384) → ephemeral FAISS index → hybrid retrieval → optional cross-encoder rerank → LLM-based contradiction check and synthesis. Representative files: `README.md`, `agents/processing_agent.py`, `rag/chunking.py`, `utils/llm_client.py`.

- Important patterns and expectations to follow:
  - Embedding dimension: code expects 384 in places (see `ProcessingAgent._warmup_models`). Avoid changing embedding shapes without updating warmup checks and callers.
  - Chunking is section-aware with a paragraph fallback. There are two chunking flavors: simple fallback chunking (used when section detection finds <2 sections) and token/sentence-aware chunking in `AdvancedDocumentChunker` (`rag/chunking.py`). Preserve section metadata (section, start/end positions, token counts) when returning chunks.
  - Processing pipeline is side-effect free: `ProcessingAgent.__call__` copies state and returns a new State; keep transformations pure where possible.
  - LLM failover behavior: `utils/llm_client.py` prefers a primary provider (Gemini via LangChain wrapper) then falls back to Groq. When adding LLM calls, mirror the existing retry/failover logging format and structured JSON logging.

- Developer workflows / commands (discoverable in repo README and test fixtures):
  - Install / dev: `pip install -e .`
  - Configure: copy `.env.example` → `.env` and set keys (LLM_MODEL_PRIMARY, LLM_MODEL_FALLBACK, PRIMARY_LLM_API_KEY, SECONDARY_LLM_API_KEY).
  - Run app (local dev): `python -m necthrall_lite` (project README shows this entrypoint). If you use `main.py` directly, prefer the configured FastAPI/NiceGUI integration entrypoint.
  - Tests: run `pytest -q`. Note `tests/conftest.py` pre-imports modules in a strict order to avoid DLL/threading issues – fitz (PyMuPDF) first, then torch, then sentence-transformers. Also set these env vars in CI/local shells to avoid deadlocks:
    - OMP_NUM_THREADS=1
    - MKL_NUM_THREADS=1
    - TOKENIZERS_PARALLELISM=false
  - Long-running/performance tests are marked with pytest markers (`slow`, `pdf_dependent`, `integration`, `performance`). Use `-m "not slow"` to skip slow tests during quick iteration.

- Conventions to keep consistent:
  - Logging: structured JSON logging is used across LLM and agent calls (see `utils/llm_client.py`). When emitting telemetry/events, follow existing keys: `event`, `provider`, `execution_time`, `error`, etc.
  - Error handling: agents collect per-paper `paper_errors` in stats rather than raising immediately. Preserve this approach in pipeline stages so failures are transparent in `processing_stats`.
  - Side-effect minimization: most agents return modified copies of `State` and append `processing_stats`/`top_passages` rather than mutating input state.
  - Config: defaults (chunk sizes, overlap, top_k, embedding model name) are read from env/config files. Use `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_TOP_K`, `RAG_EMBEDDING_MODEL` when adding features.

- Integration points & external dependencies to be careful with:
  - OA providers: OpenAlex, Unpaywall, PMC, arXiv. Paper fetching functions rely on stable DOIs/URLs and prefer PMC/arXiv for direct PDFs.
  - Embeddings: sentence-transformers models (installed via `sentence-transformers`) and a cached embedding model attached to `app.state.embedding_model`. Ensure model initialization/warmup remains compatible with both numpy/list and array returns.
  - Vector DB: FAISS (cpu) is used for ephemeral indices. Rebuilds are per-query; files under `data/index` are ephemeral in README.
  - LLMs: LangChain adapters: `langchain_google_genai` and `langchain_groq`. Ensure environment variables for provider keys exist in CI.

- Small actionable examples to include in suggestions:
  - When creating chunks, return dicts/objects with `content`, `section`, `paper_id`, `paper_title`, `start_pos`, `end_pos`, `token_count`.
  - When adding logging for LLM calls, follow `json.dumps({"event":..., "provider":..., "execution_time":...})` format.
  - When changing embedding behavior, update the 384-dim check in `ProcessingAgent._warmup_models` or adapt warmup to be tolerant.

- Files to open for context when modifying behavior:
  - `agents/processing_agent.py` — main pipeline orchestration and statistics format
  - `rag/chunking.py` — advanced chunking, section detection and quality checks
  - `utils/llm_client.py` — LLM failover, logging, provider wrappers
  - `tests/conftest.py` — pytest pre-import order and environment choices (important for CI)
  - `README.md` — quickstart, env vars, and high-level architecture to align with user-facing docs

If anything here is incomplete or you'd like narrower focus (e.g., CI config, templates for PRs, or recommended unit tests), tell me which parts to expand and I will iterate.
