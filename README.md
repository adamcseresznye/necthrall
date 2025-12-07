---
title: Necthrall
emoji: 🔬
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

<div align="center">
  <img src="logo/necthrall.png" alt="Necthrall Logo" width="120">
  <h1>Necthrall</h1>
  <h3><i>Science, Distilled.</i> An open-source, AI research assistant.</h3>

  ![License](https://img.shields.io/badge/license-GPL-green)
  ![Python](https://img.shields.io/badge/python-3.11+-blue)
  ![Stack](https://img.shields.io/badge/Stack-NiceGUI%20%7C%20FastAPI%20%7C%20LlamaIndex-purple)

  <br/>

  <strong><a href="https://necthrall-ai.hf.space">🔴 Live Demo on Hugging Face</a></strong>
</div>

---

## ⚡ Key Features

- **📄 Full-Text Analysis:** We don't just read abstracts. Necthrall downloads and chunks the actual PDFs to find evidence buried deep in Methods and Results.
- **🛡️ Privacy First:** Completely stateless. No database, no user accounts, no history.
- **🔍 Deep Search:** Queries **Semantic Scholar** to find hundreds of candidates per query, filtered down to the top 10 most relevant open-access papers.
- **✅ Verifiable:** Every sentence is backed by an inline citation `[1]` that links directly to the source passage.
- **🧠 Resilient:** Built on **LiteLLM**, Necthrall prioritizes **Google Gemini** for synthesis, automatically switching to **Groq** if the primary model is unavailable or fails.

---

## 🏗️ Architecture

Necthrall runs a **6-stage pipeline** for every query. It is optimized for speed (async/parallel) and transparency.

```mermaid
graph TD
    %% Define Custom Styles
    classDef input fill:#6366f1,stroke:none,color:#fff,rx:10,ry:10,font-weight:bold;
    classDef logic fill:#fff,stroke:#cbd5e1,stroke-width:1px,color:#475569,rx:5,ry:5;
    classDef rag fill:#ecfdf5,stroke:#10b981,stroke-width:2px,color:#064e3b,rx:5,ry:5;
    classDef llm fill:#eff6ff,stroke:#3b82f6,stroke-width:2px,color:#1e40af,rx:5,ry:5;
    classDef result fill:#10b981,stroke:none,color:#fff,rx:10,ry:10,font-weight:bold;

    %% The Flow
    Start(["🔎 User Query"]):::input --> B

    subgraph " 1. Retrieval Phase "
        direction TB
        B["⚡ Query Expansion Agent<br/><i>Multi-variant generation</i>"]:::logic
        B --> C["📚 Async Retrieval<br/><i>Semantic Scholar API (Parallel)</i>"]:::logic
        C --> D["⚖️ Composite Reranker<br/><i>Relevance + Authority Scoring</i>"]:::logic
        D --> E["⬇️ Document Ingestion<br/><i>Direct PDF Fetching & Parsing</i>"]:::logic
    end

    E --> F

    subgraph " 2. Processing Phase "
        direction TB
        F["🧠 In-Memory Vector Store<br/><i>FAISS + Dense Embeddings</i>"]:::rag
        F --> G["🤖 Contextual Synthesis<br/><i>LLM Reasoning + Citation Mapping</i>"]:::llm
    end

    G --> H(["✨ Verified Response"]):::result

    %% Styling Links
    linkStyle default stroke:#94a3b8,stroke-width:2px;
````

### The Stack

  * **UI/Server:** NiceGUI + FastAPI (Single Python app)
  * **RAG:** LlamaIndex + FAISS (Ephemeral in-memory indices)
  * **Data:** Semantic Scholar API
  * **LLM Routing:** LiteLLM

-----

## 🚀 Quickstart

### Prerequisites

  * Python 3.11+
  * [Semantic Scholar API Key](https://www.semanticscholar.org/product/api) (Free)
  * LLM Provider Key (Google Gemini or Groq)

### Local Development

1.  **Clone & Install**

    ```bash
    git clone [https://github.com/adamcseresznye/necthrall.git](https://github.com/adamcseresznye/necthrall.git)
    cd necthrall
    pip install -r requirements.txt
    ```

2.  **Configure Environment**
    Create a `.env` file in the root directory

3.  **Run the App**

    ```bash
    python main.py
    ```

    The UI will be available at `http://localhost:8080`.

-----

## 🔒 Privacy & Data Retention

Necthrall is designed to be **ephemeral** and privacy-first:

  * **No Database:** We do not store user search queries or results.
  * **Memory Only:** Vector indices (FAISS) are built in RAM and destroyed immediately after the response is generated.

-----

## 🛠️ Configuration

Necthrall uses **LiteLLM** for model routing. You can configure the following in your `.env` file:

| Variable | Description | Required |
| :--- | :--- | :--- |
| `SEMANTIC_SCHOLAR_API_KEY` | For searching papers | **Yes** |
| `GOOGLE_API_KEY` | Primary LLM (e.g., Gemini 2.0 Flash) | **Yes** |
| `GROQ_API_KEY` | Fallback LLM (e.g., Llama 3.3) | Optional |
-----

## 🙏 Acknowledgements

This project is made possible by the open-science ecosystem:

- **Semantic Scholar**: Necthrall uses the Semantic Scholar API for paper retrieval and citation data.
- **Hugging Face**: For hosting the demo infrastructure and models.

## 📄 License

GPL-3.0 license

