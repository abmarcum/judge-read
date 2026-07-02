# Judge Read

A Retrieval-Augmented Generation (RAG) search engine for attorneys to ground LLMs in actual US case law, preventing hallucinations.

## Architecture

This project is split into three main components:
1. **Frontend**: A modern, glassmorphism React application built with Vite for attorneys to interface with the LLM and search cases.
2. **Backend**: A Python FastAPI server that acts as the orchestration layer between the frontend UI, the embedding models, and the Vector Database. It is supercharged with Hybrid Search, Two-Stage Retrieval (Cohere Reranking), Metadata Filtering, and Telemetry/LangSmith tracing.
3. **Data Pipeline**: A unified Python CLI tool (`data_pipeline.py`) to ingest raw legal opinions from either CourtListener or Hugging Face, clean it, chunk it, embed it, and upsert it into PostgreSQL with `pgvector` and Full-Text Search indexing.

## Directories

- `/frontend` - Contains the React/Vite UI application.
- `/backend` - Contains the FastAPI server and all the data processing scripts.
- `/data` - The target directory where raw case law text and extracted datasets are stored (ignored by Git).
- `/docs` - Contains the original architectural blueprint.

## Getting Started

1. Set up the Backend (see `backend/README.md`)
2. Set up the Frontend (see `frontend/README.md`)
3. Run the Data Pipeline to populate your local `data/` folder and pgvector database!
