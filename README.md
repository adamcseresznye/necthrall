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

  <strong><a href="https://necthrall.tech/">🔴 Live Demo</a></strong>
</div>

## 🎥 Video Demo

<div align="center">
  <a href="https://youtube.com/shorts/uSinui_HFrQ">
    <img src="https://img.youtube.com/vi/z6Bg-02ml8w/0.jpg" alt="Necthrall Demo Video" width="300"/>
  </a>
  <br>
  <i>Watch Necthrall retrieve, analyze, and cite full-text scientific papers in real time.</i>
</div>

---

## ⚡ Key Features

- **📄 Full-Text Analysis:** We don't just read abstracts. Necthrall downloads and chunks the actual PDFs to find evidence buried deep in Methods and Results.
- **🛡️ Privacy First:** Completely stateless. No database, no user accounts, no history.
- **🔍 Deep Search:** Queries **Semantic Scholar** to find hundreds of candidates per query, filtered down to the top N most relevant open-access papers.
- **✅ Verifiable:** Every sentence is backed by an inline citation `[1]` that links directly to the source passage.
- **🧠 Resilient:** Built on **LiteLLM**, Necthrall automatically switches to the secondary provider if the primary one is unavailable or fails.
---

## 🏗️ Architecture

Necthrall runs a **10-stage pipeline** for every query. It is optimized for speed (async/parallel) and transparency.

```mermaid
flowchart TD
    %% Global Styles
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px,color:#000;
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#000;
    classDef terminal fill:#263238,stroke:#333,color:#fff;
    classDef success fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef fail fill:#ffebee,stroke:#c62828,stroke-width:2px;
    classDef note fill:#e1f5fe,stroke:#0277bd,stroke-dasharray: 5 5;

    Start([<b>User Query</b>]) --> OptStart

    %% ==========================================================
    %% PHASE 0: PRE-PROCESSING
    %% ==========================================================
    subgraph Phase0 [Phase 0: Planning & Optimization]
        direction TB
        OptStart[<b>Generate Dual Queries</b>]
        
        OptFail{LLM Failure?}
        
        FallbackOpt[Fallback:Use Original Query]
        SuccessOpt[Success:Use 'Final Rephrase' + Variants]
        
        OptStart --> OptFail
        OptFail -- Yes --> FallbackOpt
        OptFail -- No --> SuccessOpt
    end

    %% ==========================================================
    %% PHASE 1: DISCOVERY
    %% ==========================================================
    subgraph Phase1 [Phase 1: Discovery Service]
        direction TB
        DiscCall[<b>Search Semantic Scholar API<br/>]
        
        QualityGate{Quality Gate<br/>Passed?}
        
        RefineLoop[<b>Refinement Loop</b><br/>Retry with 'Broad' Query]:::note
        
        Ranking[<b>Composite Scoring</b><br/>Relevance + Authority + Recency]
        
        FallbackOpt --> DiscCall
        SuccessOpt --> DiscCall
        
        DiscCall --> QualityGate
        QualityGate -- No (Attempt 1) --> RefineLoop --> DiscCall
        QualityGate -- Yes --> Ranking
        QualityGate -- No (Attempt 2) --> Ranking
    end

    CheckFinalists{Any Papers<br/>Found?}
    ExitNoPaper([STOP: Return 'No Papers Found']):::terminal
    
    Ranking --> CheckFinalists
    CheckFinalists -- No --> ExitNoPaper

    %% ==========================================================
    %% PHASE 2: INGESTION
    %% ==========================================================
    subgraph Phase2 [Phase 2: Ingestion Service]
        direction TB
        DeepMode{Deep Mode<br/>Enabled?}
        
        subgraph DeepIngest [Deep Analysis]
            PDFDownload[<b>PDF Acquisition</b><br/>Download Full Text]
            Processing[<b>Processing</b><br/>Chunking & Embedding]
        end
        
        subgraph FastIngest [Fast Scan]
            Abstracts[<b>Abstract Ingestion</b><br/>Use Abstracts Only]
        end
        
        DeepMode -- Yes --> PDFDownload --> Processing
        DeepMode -- No --> Abstracts
    end

    CheckChunks{Content<br/>Available?}
    ExitNoChunks([STOP: Return 'No Content']):::terminal

    CheckFinalists -- Yes --> DeepMode
    Processing --> CheckChunks
    Abstracts --> CheckChunks
    CheckChunks -- No --> ExitNoChunks

    %% ==========================================================
    %% PHASE 3: RAG
    %% ==========================================================
    subgraph Phase3 [Phase 3: RAG Service]
        direction TB
        SelectQuery[Select Query:<br/>Use 'Final Rephrase' if available]
        
        Hybrid[<b>Hybrid Retrieval</b><br/>Vector + BM25 Search]
        Rerank[<b>Reranking</b><br/>Cross-Encoder Re-scoring]
        
        Synth[<b>Synthesis</b><br/>Generate Answer with Citations]
        Verify[<b>Verification</b><br/>Check Citation Accuracy]:::success
    end

    CheckChunks -- Yes --> SelectQuery
    SelectQuery --> Hybrid --> Rerank --> Synth --> Verify
    
    Verify --> End([Final Response]):::terminal

    %% Class Assignments
    class OptFail,QualityGate,CheckFinalists,DeepMode,CheckChunks decision
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
  * LLM Provider Key (Cerebras or Groq)

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

    The UI will be available at `http://localhost:7860`.

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
| `PRIMARY_LLM_API_KEY` | Primary LLM (e.g., Llama 3.3) | **Yes** |
| `SECONDARY_LLM_API_KEY` | Fallback LLM (e.g., Llama 3.3) | Optional |
-----

## 🙏 Acknowledgements

This project is made possible by the open-science ecosystem. Special thanks to:

- **[Semantic Scholar](https://www.semanticscholar.org/)**: For their open API that powers our paper discovery.
- **[LlamaIndex](https://www.llamaindex.ai/)**: The orchestration framework for our RAG pipeline and data ingestion.
- **[LiteLLM](https://github.com/BerriAI/litellm)**: For providing the unified interface that makes our multi-model routing possible.
- **[NiceGUI](https://nicegui.io/) & [FastAPI](https://fastapi.tiangolo.com/)**: For the seamless full-stack Python experience.
- **[Hugging Face](https://huggingface.co/)**: For hosting the demo infrastructure and serving the embedding models.

## 📄 License

GPL-3.0 license

