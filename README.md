# Judge Read ⚖️📖

Judge Read is a high-performance, Retrieval-Augmented Generation (RAG) search engine designed specifically for attorneys and legal professionals. It grounds Large Language Models (LLMs) in actual, verified United States case law to ensure citation accuracy and completely eliminate hallucinations.

---

## 🏛️ System Architecture

The following diagram illustrates how the three main components of Judge Read interact during data ingestion (offline pipeline) and search/chat queries (online API):

```mermaid
graph TD
    subgraph Data Ingestion Pipeline (Offline)
        CL[CourtListener API] --> DP[Data Pipeline CLI]
        HF[Hugging Face Dataset] --> DP
        DP --> |Clean & Chunk| DB[(PostgreSQL + pgvector)]
    end

    subgraph User Query & Inference Flow (Online)
        User[Attorney / User] <-->|UI Interaction| FE[React/Vite Frontend]
        FE <-->|REST API / Port 8000| BE[FastAPI Backend]
        BE <-->|Hybrid Search & Metadata Filters| DB
        BE -->|Two-Stage Cohere Rerank| BE
        BE -->|Augmented Context| LLM{LLM Engine <br> GPT / Claude / Ollama}
        BE -.->|Telemetry & Monitoring| LS[LangSmith Tracing]
    end
```

### 1. Frontend UI
A premium, modern React application styled with responsive **Vanilla CSS (Glassmorphism)**. Features:
* **Interactive Citator Status**: Visually alerts attorneys to overruled or questionable precedents with bright red "OVERRULED" citator badges.
* **Full Document View**: A reading pane allowing users to review the entire, unchunked raw text of retrieved legal opinions.
* **Dynamic Configuration Sidebar**: Enables on-the-fly switching between embedding models and inference engines without restarting code.
* **Responsive Layouts**: Supports desktop, tablet, and mobile screens through custom fluid layouts.

### 2. Backend Orchestrator
A FastAPI Python server orchestrating interactions between the client UI, embedding services, LLM API providers, and the vector database:
* **Hybrid Search**: Fuses dense semantic vector searches with relational SQL filters (e.g., Year, Court, Jurisdiction) and PostgreSQL Full-Text Search (`tsvector`).
* **Two-Stage Retrieval**: Pulls top candidate documents from the database and uses **Cohere Cross-Encoder Reranking** to supply the model with the most relevant passages.
* **Telemetry & LLM Tracing**: Integrated with **LangSmith** for full-lifecycle pipeline tracing and performance auditing.
* **Dual-Mode MCP Server**: Exposes case-law searching tools directly to AI tools (such as Claude Desktop) using Anthropic's Model Context Protocol.

### 3. Data Processing Pipeline
A multi-threaded data pipeline CLI tool (`data_pipeline.py`) that handles downloading, cleaning, parsing HTML opinion payloads, chunking text, generating embeddings, and bulk loading into PostgreSQL.

---

## 📁 Repository Structure

```
.
├── backend/            # FastAPI Server, MCP Server, Ingestion Pipeline, & Test Suite
│   ├── main.py             # FastAPI REST Server & MCP Server definition
│   ├── mcp_server.py       # Standalone CLI entrypoint for MCP
│   ├── data_pipeline.py    # Data ingestion CLI (download, clean, embed, ingest)
│   ├── db_setup.py         # Relational database schema & index initializer
│   ├── test_framework.py   # Comprehensive diagnostic test suite
│   ├── config.json         # Unified system model and API configuration
│   └── pyproject.toml      # Backend python package dependencies (uv manager)
├── frontend/           # React 18 / Vite / Vanilla CSS Web Application
│   ├── src/                # Component sources and main App structure
│   ├── package.json        # Frontend NPM package dependencies
│   └── README.md           # Frontend-specific setup documentation
├── data/               # Git-ignored workspace folder for raw files & caches
├── docs/               # Original architecture blueprint files
└── README.md           # Root configuration and overview (This file)
```

---

## ⚡ Quick Start Guide

To get Judge Read up and running on your local machine:

### ⚙️ Prerequisites
Ensure you have the following installed:
* **Python** (version `>= 3.13` recommended)
* **Node.js** (version `>= 18` recommended) & `npm`
* **PostgreSQL** with the `pgvector` extension (Docker-ready steps provided in backend docs)
* **uv** Python package manager (Recommended for fast dependency resolution)

### 🚀 Setup Steps

1. **Initialize the Database**:
   Spin up PostgreSQL with `pgvector`. (See the [Backend Setup Guide](file:///Users/andrew/ai-workspace/code/judge-read/backend/README.md#setting-up-postgresql-with-pgvector-linux) for Docker instructions).

2. **Configure and Load Data**:
   Navigate to `/backend` and install dependencies. Initialize database schemas and run the ingestion pipeline to populate your vector database with legal opinions.
   ```bash
   cd backend
   uv sync
   # Run pipeline (default downloads sample cases from Hugging Face & embeds them)
   uv run python data_pipeline.py --action all
   ```
   *(For details on passing custom Postgres ports, hosts, and Ollama settings, see [Backend README](file:///Users/andrew/ai-workspace/code/judge-read/backend/README.md)).*

3. **Start the Backend API Server**:
   Start Uvicorn to serve the API endpoint:
   ```bash
   uv run uvicorn main:app --reload
   ```

4. **Start the Frontend UI**:
   In another terminal, set up the React client and spin up Vite's developer server:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Open your browser to `http://localhost:5173` to query the engine!

5. **Verify System Integrity**:
   Run the advanced automated diagnostics suite to test network routing, CORS, DB queries, FTS performance, and LLM reachability:
   ```bash
   cd backend
   uv run python test_framework.py --ui-ip localhost --backend-ip localhost
   ```
