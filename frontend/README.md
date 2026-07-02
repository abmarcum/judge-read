# Judge Read - Frontend

The attorney-facing Search and Chat interface for Judge Read.

## Features
- **Modern Glassmorphism UI**: High-end visual aesthetics tailored for a professional legal environment using Google's `Outfit` typography.
- **Two-Stage Retrieval & Tracing**: Settings panel allows injecting Cohere API keys for reranking and LangSmith keys for LLM telemetry monitoring.
- **External Search Filters**: Contextual inputs above the chat area allow dynamic pre-filtering of search results by Court and Year.
- **Dynamic Configuration Sidebar**: A slide-out panel allowing dynamic selection of the **Embedding Model** and **LLM Engine**.
- **Session Memory**: Securely stores conversation states using Postgres, allowing users to build context over time.
- **Citator Status Tracking**: Automatically tags retrieved citations with bright red "OVERRULED" badges if the case is no longer good law.
- **Full Document Reading Pane**: Citation pills are completely interactive. Clicking on a retrieved case opens a beautiful glassmorphic full-screen modal, allowing you to read the entire, unchunked original legal document natively.
- **Adaptive Authentication**: Seamlessly transitions between a secure API Key input or a local Host URL depending on the selected provider.
- **Micro-Animations**: Smooth chat bubbles, modal transitions, and loading spinners to enhance the user experience.

## Tech Stack
- React 18
- Vite
- Axios
- Vanilla CSS
- Lucide React (Icons)

## Setup & Running

Install dependencies:
```bash
npm install
```

Start the development server:
```bash
npm run dev
```

The frontend will start at `http://localhost:5173` and route search requests to your local backend API at `http://localhost:8000/api/search`.
