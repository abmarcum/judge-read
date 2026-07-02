# Judge Read - Backend API & Data Pipeline

The orchestration logic and data ingestion pipeline for the Judge Read engine.

## Tech Stack
- **Python / uv**: Environment and dependency management.
- **FastAPI**: Provides the REST API `POST /api/search` endpoint.
- **LangChain**: Manages text splitting, embedding integrations, and LLM query orchestration.
- **PostgreSQL + pgvector**: Powers our Hybrid Search engine by combining semantic vector search with relational metadata filtering and Full-Text Search (FTS). It also stores Chat History, Telemetry analytics, and full, unchunked case law text.
- **Cohere**: Added to support Two-Stage Retrieval (Cross-Encoder Reranking) to refine the top 30 database results into the best 5 for generation.
- **LangSmith**: Used for backend LLM Tracing and monitoring.
- **BeautifulSoup**: Used to clean raw HTML from case law text.

## Project Structure
- `main.py` - The FastAPI entrypoint handling dynamic inference requests from the frontend, augmented with raw SQL for Hybrid Search, Metadata Filtering, Chat History, Telemetry logging, Cohere Reranking, and fetching full case documents via `GET /api/cases/{case_id}`.
- `data_pipeline.py` - A unified CLI tool that handles downloading raw data, extracting it into `data/`, cleaning it, assigning metadata, storing the full original document into Postgres, chunking it, and embedding it into the vector database.
- `db_setup.py` - Initializes the Postgres relational schema (`full_cases`, `chat_sessions`, `chat_messages`, `analytics_queries`) and creates the `tsvector` index for Full-Text Search.

## Setup & Running

Install dependencies and enter your virtual environment using `uv`:
```bash
uv sync
```

To run the full data extraction and embedding pipeline (defaulting to Hugging Face):
```bash
uv run python data_pipeline.py --action all
```

You can also use CLI options to limit downloads, change sources, or configure the embedding provider (OpenAI or local Ollama). For example, to download only 100 cases and embed them locally using Ollama for free:
```bash
uv run python data_pipeline.py --limit 100 --action all --source hf --embed-provider ollama --embed-model nomic-embed-text
```

**Embedding Options:**
- `--embed-provider`: `openai` (default) or `ollama`.
- `--embed-model`: The specific model to use (defaults to `text-embedding-3-small` or `nomic-embed-text`).
- `--embed-key`: API key for OpenAI (can also use `OPENAI_API_KEY` env var).
- `--embed-host`: The Ollama server host (defaults to `http://localhost:11434`).

To run the REST API:
```bash
uv run uvicorn main:app --reload
```

## MCP Server
Judge Read includes a fully functional Model Context Protocol (MCP) server powered by Anthropic's `mcp` SDK. This allows you to expose the underlying PostgreSQL vector database as an intelligent tool directly to other AI applications (like Claude Desktop).

To inspect the MCP Server or use it locally:
```bash
# Provide an OpenAI key so the server can generate embeddings for semantic search
export OPENAI_API_KEY="your-key-here"

# Run the MCP Server Inspector
uv run mcp dev mcp_server.py
```

Other AI agents can use the `search_case_law` tool exposed by this server to natively execute vector searches with all of the granular metadata filters (Year, Court, Jurisdiction, Good Law status).
