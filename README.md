# Necthrall Lite

Your on-demand, credibility-aware scientific explainer. Fetches open-access papers, builds a per-query RAG index, and returns a concise synthesis with inline citations, passage spans, a consensus gauge, and contradiction highlights.

## Highlights

- Evidence-backed answers, not guesses  
- Open-access only (Unpaywall, OpenAlex, PMC, arXiv)  
- Real RAG loop: fetch → parse → chunk → embed → retrieve → rerank → synthesize  
- Inline citations with exact passage spans  
- Lightweight credibility scoring (journal tier, citations, recency)  
- Consensus estimation and contradiction detection  
- Single-app stack: NiceGUI (UI) + FastAPI (API) in Python

***

## Demo (What It Feels Like)

Query: “Explain CRISPR off-target risks.”  
- Retrieves ~30 OA papers  
- Ranks credibility (journal tier, citations, recency)  
- Consensus: “Moderate (≈65%)”  
- Contradictions: shows 1–3 pairs with short explanations  
- Synthesizes a 300–500 word summary with inline citations and passage spans

***

## Quickstart

### 1) Requirements

- Python 3.11+  
- macOS/Linux/Windows  
- Basic network access to OA APIs

### 2) Install

```bash
git clone <your-repo-url> necthrall-lite
cd necthrall-lite
pip install -e .
```

### 3) Configure

```bash
cp .env.example .env
# Edit .env to set your keys/emails and model choices
```

Minimum variables:
- UNPAYWALL_EMAIL=you@example.com
- OPENALEX_EMAIL=you@example.com
- OPENAI_API_KEY=sk-... (or your preferred LLM provider key)

### 4) Run

```bash
python -m necthrall_lite
# Follow the printed URL to open the UI
```

***

## Environment Variables

- OPENAI_API_KEY: LLM for contradiction verification and synthesis (or alternate provider)  
- UNPAYWALL_EMAIL: Required for Unpaywall OA resolution  
- OPENALEX_EMAIL: Recommended for OpenAlex (good API citizenship)  
- RAG_EMBEDDING_MODEL: e.g., all-MiniLM-L6-v2  
- RAG_MAX_PAPERS: default 30  
- RAG_TOP_K: default 12  
- RAG_CHUNK_SIZE: default 800  
- RAG_CHUNK_OVERLAP: default 120  
- RAG_RERANK_ENABLED: true|false  
- TIMEOUT_SEC: default 20

***

## RAG Pipeline (Per Query)

1) Search  
- Query OpenAlex by topic (relevance + recency)  
- Collect DOIs, OA status, and any fulltext URLs  
- Backfill missing fulltexts via Unpaywall by DOI  
- Prefer PMC/arXiv when available for direct PDFs

2) Download  
- Download only the top ~30 OA PDFs  
- Strictly API-based resolution (no scraping)

3) Parse & Chunk  
- Extract text (abstract/conclusions prioritized)  
- Chunk paragraphs/sections (configurable size/overlap)

4) Embed & Index  
- Encode chunks with sentence-transformers  
- Build ephemeral FAISS index under data/index

5) Retrieve & Rerank  
- Similarity search (top-k)  
- Optional cross-encoder reranker  
- Track passage spans for transparent citations

6) Credibility, Consensus, Contradictions  
- Credibility: journal tier (simple mapping) + normalized citations + recency weights  
- Consensus: label conclusion chunks (support/caution/neutral) and aggregate  
- Contradictions: pair top conclusions; compact LLM check to tag supports/contradicts/unclear with brief rationale

7) Synthesize  
- Short, citable narrative with inline citations and passage spans  
- Show consensus gauge and contradiction callouts  
- List ranked sources with OA links (when available)

8) Cleanup  
- Ephemeral cache; discard index after response (configurable)

***

## Architecture

- Single Python app: NiceGUI (UI) + FastAPI (API)  
- On-demand OA retrieval (OpenAlex, Unpaywall, PMC, arXiv)  
- Per-query vector index (FAISS)  
- BYO LLM for contradiction checks and synthesis  
- No persistent warehouse or scheduled crawlers

***

## Project Structure

```text
necthrall-lite/
├── pyproject.toml
├── README.md
├── .gitignore
├── .env.example
├── LICENSE
├── tests/
│   ├── test_api_routes.py
│   ├── test_services.py
│   └── test_rag_pipeline.py
├── src/
│   └── necthrall_lite/
│       ├── __init__.py
│       ├── main.py                 # entrypoint: FastAPI + NiceGUI integration
│       ├── settings.py             # env + config
│       ├── logging_conf.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes.py           # /api/query, /api/health
│       │   ├── deps.py
│       │   └── schemas.py          # Pydantic models
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   └── security.py
│       ├── clients/
│       │   ├── __init__.py
│       │   ├── openalex_client.py
│       │   ├── unpaywall_client.py
│       │   ├── pmc_client.py
│       │   └── arxiv_client.py
│       ├── loaders/
│       │   ├── __init__.py
│       │   ├── oa_link_resolver.py
│       │   ├── pdf_downloader.py
│       │   └── pdf_text_extractor.py
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── chunking.py
│       │   ├── embeddings.py
│       │   ├── vectorstore.py
│       │   ├── retrieval.py
│       │   └── reranker.py
│       ├── services/
│       │   ├── __init__.py
│       │   ├── retrieval_service.py
│       │   ├── credibility_service.py
│       │   ├── contradiction_service.py
│       │   └── synthesis_service.py
│       ├── nlp/
│       │   ├── __init__.py
│       │   ├── spacy_utils.py
│       │   └── sentiment.py
│       ├── ui/
│       │   ├── __init__.py
│       │   ├── pages.py            # NiceGUI UI
│       │   ├── styles.py
│       │   └── components/
│       │       ├── consensus_gauge.py
│       │       ├── credibility_badge.py
│       │       └── contradiction_callout.py
│       ├── integration/
│       │   ├── __init__.py
│       │   └── fastapi_nicegui.py
│       ├── configs/
│       │   ├── journal_tiers.json
│       │   └── prompts/
│       │       ├── contradiction_check.txt
│       │       └── synthesis_template.txt
│       └── data/
│           ├── .gitkeep
│           ├── cache/
│           └── index/
└── .vscode/
    └── settings.json
```

***

## API (used by the UI)

- GET /api/health → { status: "ok" }  
- POST /api/query  
  - Body: { topic: string, max_papers?: int, top_k?: int }  
  - Returns: { summary, consensus, contradictions[], sources[], timings, params }

***

## Configuration (Key Settings)

- max_papers: default 30  
- chunk_size / chunk_overlap: default 800 / 120  
- embedding_model: e.g., sentence-transformers/all-MiniLM-L6-v2  
- top_k: default 12  
- rerank_enabled: default true  
- credibility_weights: journal tier 0.5, citations 0.3, recency 0.2  
- llm_model: e.g., gpt-4o-mini or local provider  
- timeout_sec: per-API and overall query budget

***

## Evaluation (Optional)

- Topics:  
  - “CRISPR off-target risks”  
  - “Intermittent fasting metabolic outcomes”  
  - “Climate adaptation crop resilience”  
- Metrics:  
  - Faithfulness: statements backed by explicit passage spans  
  - Citation coverage: ≥5 citations with diverse sources  
  - Latency: ≤5–7 seconds end-to-end on a typical laptop  
- Spot checks: contradictions flagged correctly; consensus gauge matches qualitative impression

***

## Development

- Install: `pip install -e .`  
- Run: `python -m necthrall_lite`  
- Test: `pytest -q`  
- Lint/Format: add pre-commit, ruff/black if desired
- Run: `python -m necthrall_lite`  

### Running tests

Use the pytest entry on your activated virtualenv. The project uses markers to split suites so you can run only the fast or longer tests:

- Run the fast unit tests only:

```
pytest -m "unit" -q
```

- Run the integration tests only (excludes tests marked `performance` or `slow` by default):

```
pytest -m "integration" -q
```

- Run performance/benchmark tests (long-running):

```
pytest -m "performance" -q
```

Notes:
- These commands assume your virtualenv is activated (so `pytest` refers to the environment's pytest). You said you prefer not to use `python -m pytest` — the above uses the plain `pytest` entry as requested.
- CI scripts often prefer the explicit `python -m pytest` form for reproducibility; locally `pytest` is fine when the venv is active.
- By default `performance` and `slow` tests are skipped in `pytest.ini` unless you explicitly include them in marker expressions.

***

## Design Choices

- **On-demand OA retrieval**: avoids storage/ops complexity  
- **Ephemeral vector index**: per-query isolation and simplicity  
- **Credibility heuristics**: simple, transparent, adjustable weights  
- **Minimal LLM usage**: contradiction verification + final synthesis  
- **Inline citations with spans**: traceability by design

***

## Open-Access Sources

- OpenAlex: metadata + OA links  
- Unpaywall: DOI-based OA resolution  
- PubMed Central: OA full-texts (plus OAI-PMH/ID services)  
- arXiv: preprints with direct PDFs

***

## Roadmap (Post-MVP)

- Optional long-running cache for repeated topics  
- Better reranking and section-aware retrieval  
- Claim-level extraction and structured evidence maps  
- Export to Markdown/Word with citation styles  
- More robust journal tiering and bias detection

***

## License

MIT (see LICENSE)

***

## Acknowledgements

Thanks to the open-access ecosystem and APIs that make evidence-first tools feasible.

