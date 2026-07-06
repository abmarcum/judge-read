# Judge Read - Backend API & Data Ingestion Pipeline ⚙️🔌

This directory contains the FastAPI server, the standalone and integrated Model Context Protocol (MCP) servers, the data ingestion pipeline, and a comprehensive suite of diagnostic tests.

---

## 🛠️ Tech Stack

* **Environment Manager**: [uv](https://github.com/astral-sh/uv) (fast Python environment and package manager).
* **REST Framework**: [FastAPI](https://fastapi.tiangolo.com/) with [Uvicorn](https://www.uvicorn.org/) for async execution.
* **Vector DB & Search**: PostgreSQL with `pgvector` extension. Custom SQL queries fuse dense vector similarity, relational metadata filters, and Full-Text Search (FTS).
* **Orchestration**: [LangChain](https://github.com/langchain-ai/langchain) for text processing, embedding integrations, and LLM providers.
* **Two-Stage Reranker**: [Cohere Rerank API](https://cohere.com/rerank) (via Cross-Encoder models) to filter search candidates down to the most relevant context.
* **Observability**: [LangSmith](https://www.langchain.com/langsmith) for deep telemetry, latency analysis, and query tracing.
* **HTML Processing**: BeautifulSoup4 for stripping HTML markup from raw case opinions.

---

## 📁 Project Structure

* **[main.py](file:///Users/andrew/ai-workspace/code/judge-read/backend/main.py)**: The FastAPI server entrypoint. Exposes the search endpoint (`POST /api/search`), case explorer endpoints (`GET /api/cases` and `GET /api/cases/{case_id}`), and configuration endpoints (`GET/POST /api/config`). It also hosts the integrated Model Context Protocol (MCP) server instance.
* **[mcp_server.py](file:///Users/andrew/ai-workspace/code/judge-read/backend/mcp_server.py)**: A standalone, dedicated MCP server script. Exposes the `search_case_law` tool to external AI agents using the Anthropic MCP SDK.
* **[data_pipeline.py](file:///Users/andrew/ai-workspace/code/judge-read/backend/data_pipeline.py)**: A multi-threaded, unified command-line pipeline. Downloads, extracts, cleans, chunks, embeds, and loads opinions into PostgreSQL.
* **[db_setup.py](file:///Users/andrew/ai-workspace/code/judge-read/backend/db_setup.py)**: Relational schema initializer. Creates table schemas (`chat_sessions`, `chat_messages`, `analytics_queries`, and `full_cases`), initializes the `pgvector` extension, and sets up Full-Text Search columns and GIN indices.
* **[test_framework.py](file:///Users/andrew/ai-workspace/code/judge-read/backend/test_framework.py)**: Diagnostic verification tool. Tests CORS policies, backend API responses, database queries, FTS rank performance, Ollama availability, and MCP tool registrations.
* **[config.json](file:///Users/andrew/ai-workspace/code/judge-read/backend/config.json)**: System configurations. Maps model availability lists, LLM hosts, active providers, API keys, and PostgreSQL database credentials.

---

## 🚀 Environment Setup

Install all python dependencies and activate the virtual environment using `uv`:
```bash
# Sync dependencies in a localized virtual environment
uv sync
```

---

## 🗄️ Setting up PostgreSQL with pgvector

Judge Read requires PostgreSQL (version >= 15 recommended) with the `pgvector` extension.

### Option 1: Docker (Recommended)
Launch a pre-configured PostgreSQL instance with pgvector:
```bash
docker run -d --name judgeread-db \
  -e POSTGRES_USER=judgeread \
  -e POSTGRES_PASSWORD=iamthelaw! \
  -e POSTGRES_DB=judgeread \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### Option 2: Native Ubuntu / Debian Installation
If you prefer a direct system installation:
```bash
sudo apt update
sudo apt install postgresql-16 postgresql-16-pgvector

# Access psql console to configure user and database
sudo -i -u postgres
psql -c "CREATE USER \"judgeread\" WITH SUPERUSER PASSWORD 'iamthelaw!';"
psql -c "CREATE DATABASE judgeread OWNER \"judgeread\";"
psql -d judgeread -c "CREATE EXTENSION vector;"
exit
```

---

## 📥 Standardized Ingestion Pipeline (`data_pipeline.py`)

The pipeline downloads bulk legal datasets, parses structured formats, cleans text, chunks documents, generates vector embeddings, and writes database tables.

### Key Ingestion Features
* **Standardized JSON Parsing**: Natively parses case metadata structures from Harvard Caselaw Access Project (CAP) or CourtListener datasets rather than loading raw files as unformatted text.
* **HTML Sanitization**: Beautiful Soup is applied selectively on specific text fields (e.g. opinion body, syllabus, summary) *before* serializing data to database, ensuring HTML is stripped safely without corrupting JSON markup.
* **Dynamic Metadata Harvesting**: Dynamically extracts fields like case name, date, court, and jurisdiction from the parsed JSON keys.
* **Precedent Citator Status Heuristic**: Automatically scans opinions for negative treatment keywords (e.g. `"overruled by"`, `"vacated"`, `"reversed"`) to classify precedents as `"overruled"`, `"caution"`, or `"good_law"`.
* **Legal Topic Categorization**: Scans texts for keyword sets to dynamically tag opinions under standard legal categories: `Criminal`, `Civil`, `Tax`, `Intellectual Property`, or `Constitutional`.
* **Embedding Payload Optimization**: Embeds and chunks the **clean plain-text representation** of the legal opinions instead of the raw JSON string, ensuring vector searches capture legal definitions without JSON tags and brackets.
* **Standardized JSON Database Storage**: Serializes cleaned fields back to the `full_cases.full_text` database column, aligning with React viewer expectations.

> [!TIP]
> **Database Credentials Loading**: `data_pipeline.py` automatically reads database settings, credentials, and model API keys/URLs directly from `config.json` by default!
> You only need to supply connection credentials via CLI flags (e.g. `--pg-host`, `--pg-user`) or the `DATABASE_URL` environment variable if you want to override the default configuration stored in `config.json`.

### Ingestion Examples

**1. Default Ingestion (Local DB, Hugging Face, OpenAI Embeddings)**:
Assumes PostgreSQL is running on `localhost:5432` with username `user` and password `password`.
```bash
uv run python data_pipeline.py --action all
```

**2. Custom Database Ingestion (e.g., using config.json credentials)**:
Pass credentials directly via command-line arguments:
```bash
uv run python data_pipeline.py \
  --action all \
  --pg-host 192.168.1.178 \
  --pg-user judgeread \
  --pg-password "iamthelaw!" \
  --pg-db judgeread \
  --limit 100
```
Alternatively, set the `DATABASE_URL` environment variable:
```bash
export DATABASE_URL="postgresql://judgeread:iamthelaw!@192.168.1.178:5432/judgeread"
uv run python data_pipeline.py --action all --limit 100
```

### Ingestion CLI Parameters Reference

* `--action`: `download`, `embed`, or `all` (default).
* `--source`: `hf` (Hugging Face harvard-lil/cold-cases, default) or `courtlistener` (CourtListener bulk opinions tarball).
* `--limit`: Number of cases to download and embed. Recommended for testing.
* `--all-cases`: Process all records (Warning: Harvester covers 8+ million opinions, 40GB+ of storage).
* `--embed-provider`: `openai` (default) or `ollama`.
* `--embed-model`: Specific model designation (defaults to `text-embedding-3-small` or `nomic-embed-text`).
* `--embed-key`: API key for OpenAI (defaults to `OPENAI_API_KEY` env var).
* `--embed-host`: Connection URL for Ollama instances (defaults to `http://localhost:11434`).
* `--drop`: Drops all existing schema tables in the database before starting ingestion (useful for clean reinstalls).

---

## 🖥️ Running the REST API Server

Uvicorn hosts the FastAPI REST application:
```bash
uv run uvicorn main:app --reload --reload-exclude ".venv"
```
* **Host Binding**: By default, the server binds to `127.0.0.1:8000`. To make it accessible over the local network (supporting devices connecting to the frontend on other IPs), bind to `0.0.0.0`:
  ```bash
  uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload --reload-exclude ".venv"
  ```

> [!NOTE]
> **Reloader Performance:** The project includes the `watchfiles` library to enable efficient, OS-native file system event watching for Uvicorn's `--reload` mode. Additionally, using `--reload-exclude ".venv"` explicitly prevents the watcher from traversing/monitoring the large virtual environment directory, ensuring minimal CPU utilization.

---

## 🤖 Model Context Protocol (MCP) Server Integration

The Judge Read database can be exposed directly as a tool context to AI agents (like Claude Desktop) using the Model Context Protocol (MCP).

### Available MCP Entrypoints
1. **Integrated Server**: Run the MCP tool registry directly out of `main.py` using:
   ```bash
   uv run mcp dev main.py:mcp
   ```
2. **Standalone Server**: Run the dedicated, lightweight script:
   ```bash
   uv run mcp dev mcp_server.py:mcp
   ```
   *(Alternatively, run Stdio transport directly: `uv run python mcp_server.py`)*

### Setup Requirements
For the MCP server to successfully execute vector searches:
1. Export your API Key:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```
2. Set the Database Connection:
   ```bash
   export DATABASE_URL="postgresql://judgeread:iamthelaw!@192.168.1.178:5432/judgeread"
   ```

---

## 🧪 Diagnostic Verification Suite (`test_framework.py`)

A automated diagnostics module tests system routing, payload deliveries, CORS configs, database connectivity, FTS, and MCP components.

> [!WARNING]
> **Host Testing Notice**: `test_framework.py` extracts database credentials from `config.json` and uses the database host (`pgHost`) as the default target IP for testing the frontend UI (port 5173) and backend API (port 8000).
> If your database is remote (e.g., `192.168.1.178`) but your frontend/backend servers are running on your local machine (`localhost`), you **must** override test targets with the `--ui-ip` and `--backend-ip` arguments.

### Diagnostic Examples

**Local Development Test Run (Remote Database)**:
```bash
uv run python test_framework.py --ui-ip localhost --backend-ip localhost
```

**Full Remote Server Test Run**:
```bash
uv run python test_framework.py
```
